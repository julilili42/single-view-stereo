import argparse
import numpy as np, cv2

def make_side_by_side(left, right, out_path="../output/depth_warp/stereo_pair.png"):
    pair = np.hstack([left, right])
    show_image("Comparison", pair / 255)
    cv2.imwrite(out_path, pair.astype(np.uint8))
    print(f"Side by side saved at: {out_path}")

def show_image(image_name, image):
    cv2.imshow(image_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, help="path for input (left) image")
    ap.add_argument("--depth", required=True, help=".npy-file of depth map")
    ap.add_argument("--out", default="../output/right_view.png", help="path for output image")
    ap.add_argument("--scale", type=float, default=0.5, help="scale factor for disparity (larger = larger shift)")
    ap.add_argument("--inpaint", action="store_true", help="if activated inpainting is used")
    args = ap.parse_args()

    imgL = cv2.imread(args.left).astype(np.float32)
    # normalize pixel values to [0,1]
    norm_imgL = imgL / 255.0
    show_image("Left Image", norm_imgL)
    # relative depth from depth-anything
    Z = np.load(args.depth).astype(np.float32)
    
    # show depth image 
    max_depth = max(map(max, Z))
    min_depth = min(map(min, Z))
    norm_depth_image = Z / max_depth
    print("Maximum depth", max_depth, "Minimum depth", min_depth)

    show_image("Depth Map", norm_depth_image)


    H, W = Z.shape
    assert norm_imgL.shape[:2] == (H, W), "proportions of depth map and image must match"

    # Calculate disparity map
    # Depth anything saves inverse depth Z = 1/(relative distance)
    # Disparity scales with d ∝ shift * Z
    disp = args.scale * Z
    min_disp = min(map(min, disp))
    max_disp = max(map(max, disp))
    median_disp = np.median(disp)
    print("Maximum disparity", max_disp, "Minimum disparity", min_disp)
    disp_image = disp / median_disp
    show_image("Disparity Map", disp_image)

    # filter out extrem disparities
    disp = np.clip(disp, 0, 0.75 * max_disp)


    # Forward Warping
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    x2 = np.floor(xs - disp).astype(np.int32)
    valid = (x2 >= 0) & (x2 < W)

    # initialize right image with zeros
    norm_imgR = np.zeros_like(norm_imgL) if args.inpaint else np.full_like(norm_imgL, (0, 0, 1))

    # shifting of x coordinate
    norm_imgR[ys[valid], x2[valid]] = norm_imgL[ys[valid], xs[valid]]
    
    imgR = (norm_imgR * 255).astype(np.uint8)

    if args.inpaint:
        # occlusion mask
        gray_scale = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        mask = (gray_scale == 0).astype(np.uint8) * 255
        show_image("Occlusion Mask", mask)

        # inpainting to fill out occlusions
        imgR = cv2.inpaint(imgR, mask, 3, cv2.INPAINT_TELEA)
        print("Inpainting done")
    else:
        print("Skipped Inpainting")

    show_image("Right image", imgR)
    cv2.imwrite(args.out, imgR)
    # comparison
    make_side_by_side(imgL, imgR)
    print(f"Saved right image at: {args.out}")
    
if __name__ == "__main__":
    main()
