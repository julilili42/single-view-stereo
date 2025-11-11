import os, torch
os.environ["HF_HOME"] = "/Volumes/KINGSTON/huggingface_cache"

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,   # <- fp32 erzwingen
        safety_checker=None
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()

    prompt = "a cat sitting on a meadow, photorealistic"
    image = pipe(prompt, num_inference_steps=20, guidance_scale=6.0,
                 height=512, width=512).images[0]
    image.save("left.png")

if __name__ == "__main__":
    main()
