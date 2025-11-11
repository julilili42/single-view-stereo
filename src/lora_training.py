import os, json, math, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torch import nn, optim
from tqdm import tqdm

# Configuration
DATA_DIR   = "/Volumes/KINGSTON/stereo_pairs"    
MODEL_ID   = "runwayml/stable-diffusion-v1-5"
HF_CACHE   = "/Volumes/KINGSTON/hf_cache"
OUTPUT_DIR = "/Volumes/KINGSTON/lora_out"

RANK       = 4
LR         = 1e-4
EPOCHS     = 5
BATCHSIZE  = 1
IMAGE_SIZE = (384, 384)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helpers
def _read_meta(meta_path):
    baseline, fov = 0.08, 60.0
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            m = json.load(f)
        if "baseline_m" in m: baseline = float(m["baseline_m"])
        if "fov_deg"   in m: fov = float(m["fov_deg"])
        if (m.get("fov_deg") is None) and "fx" in m and "width" in m:
            fov = math.degrees(2.0 * math.atan((float(m["width"])*0.5)/float(m["fx"])))
        if (m.get("baseline_m") is None) and "cam_left" in m and "cam_right" in m:
            try:
                tx_l = float(m["cam_left"]["matrix_world"][0][3])
                tx_r = float(m["cam_right"]["matrix_world"][0][3])
                baseline = abs(tx_r - tx_l)
            except Exception:
                pass
    return max(0.01, min(1.0, baseline)), max(20.0, min(120.0, fov))

# Dataset
class StereoScenesDataset(Dataset):
    def __init__(self, root_dir, size=(512, 512)):
        self.root_dir, self.size = root_dir, size
        self.samples = self._collect()

    def _collect(self):
        samples = []
        for scene in sorted(os.listdir(self.root_dir)):
            scene_dir = os.path.join(self.root_dir, scene)
            if not os.path.isdir(scene_dir): continue
            left  = os.path.join(scene_dir, f"{scene}_left.png")
            right = os.path.join(scene_dir, f"{scene}_right.png")
            meta  = os.path.join(scene_dir, f"{scene}_meta.json")
            if not (os.path.exists(left) and os.path.exists(right)):
                def find(sufs):
                    for suf in sufs:
                        p = os.path.join(scene_dir, f"{scene}{suf}")
                        if os.path.exists(p): return p
                    return None
                left  = left  if os.path.exists(left)  else find(["_left.jpg","_left.jpeg"])
                right = right if os.path.exists(right) else find(["_right.jpg","_right.jpeg"])
            if not (left and right): continue
            baseline, fov = _read_meta(meta) if os.path.exists(meta) else (0.08, 60.0)
            prompt = f"<B_{int(baseline*100)}cm> <FOV_{int(fov)}> right stereo view, scene {scene}"
            samples.append({"left": left, "right": right, "prompt": prompt})
        if not samples: raise RuntimeError(f"No scene pairs found under {self.root_dir}")
        return samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        left  = Image.open(s["left"]).convert("RGB").resize(self.size)
        right = Image.open(s["right"]).convert("RGB").resize(self.size)  # reserved
        return {"left": left, "right": right, "prompt": s["prompt"]}

# Pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    cache_dir=HF_CACHE,
    torch_dtype=DTYPE,      # wichtig: torch_dtype (nicht "dtype")
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_vae_tiling()
pipe.enable_attention_slicing()
# pipe.safety_checker = None
pipe.to(DEVICE)

# Lora
from peft import LoraConfig


for p in pipe.unet.parameters():
    p.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.vae.requires_grad_(False)


lora_cfg = LoraConfig(
    r=RANK,
    lora_alpha=RANK * 2,
    lora_dropout=0.0,
    bias="none",
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    init_lora_weights="gaussian",
)
ADAPTER_NAME = "stereo"
pipe.unet.add_adapter(lora_cfg, adapter_name=ADAPTER_NAME)
pipe.unet.set_adapters([ADAPTER_NAME])

trainable = [p for n, p in pipe.unet.named_parameters() if "lora" in n and p.requires_grad]
opt = optim.AdamW(trainable, lr=LR)


def get_prompt_embeds(pipe, prompts, device):
    try:
        enc = pipe.encode_prompt(
            prompts,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
        )
        return enc[0] if isinstance(enc, tuple) else enc
    except Exception:
        tok = pipe.tokenizer(
            prompts, padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        input_ids = tok.input_ids.to(device)
        with torch.no_grad():
            enc = pipe.text_encoder(input_ids)[0]  
        return enc

# Training
ds = StereoScenesDataset(DATA_DIR, size=IMAGE_SIZE)
dl = DataLoader(
    ds, batch_size=BATCHSIZE, shuffle=True,
    collate_fn=lambda b: b, num_workers=0
)

vae, unet = pipe.vae, pipe.unet
vae.eval(); pipe.text_encoder.eval(); unet.train()

num_steps = getattr(pipe.scheduler.config, "num_train_timesteps", 1000)

for epoch in range(EPOCHS):
    pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        images  = [s["left"]   for s in batch]
        prompts = [s["prompt"] for s in batch]

        with torch.no_grad():
            pixel_values = pipe.image_processor.preprocess(images).to(DEVICE) 
            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            t = torch.randint(0, num_steps, (latents.size(0),), device=DEVICE, dtype=torch.long)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, t)
            prompt_embeds = get_prompt_embeds(pipe, prompts, DEVICE).to(dtype=pipe.text_encoder.dtype, device=DEVICE)

        pred = unet(noisy_latents, t, encoder_hidden_states=prompt_embeds).sample
        loss = nn.functional.mse_loss(pred, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    print(f"Epoch {epoch+1} done.")

# Save
pipe.save_lora_weights(OUTPUT_DIR)
print("LoRA saved to", OUTPUT_DIR)
