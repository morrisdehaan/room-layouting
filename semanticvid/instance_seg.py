import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.signal import convolve2d
from sam2.build_sam import build_sam2
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_config_location(path):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=os.path.abspath(path), version_base='2.0')
    return

def gaussian_kernel(kernel_size, sigma):
    ax = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    x, y = np.meshgrid(ax, ax)
    kernel = np.exp(-(x**2 + y**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)

def load_image(path, unsharp_mask=True, kernel_size=5, sigma=1, sharpen_amount=3):
    image = Image.open(path).convert("RGB")
    image = np.array(image, dtype=np.float32)

    if unsharp_mask:
        gaussian = gaussian_kernel(kernel_size, sigma)
        blurred_image = np.zeros_like(image)
        for c in range(3):
            blurred_image[:,:,c] = convolve2d(image[:,:,c], gaussian, mode="same", boundary="wrap")
        image += sharpen_amount * (image - blurred_image)

    return np.clip(image, 0, 255).astype(np.uint8)

def generate_masks(img_file, path, generator, save_loc):
    image = load_image(os.path.join(path, img_file))
    save_mask_loc = os.path.join(save_loc,img_file.split(".")[0])
    os.makedirs(save_mask_loc)
    frame_data = {"frameid": img_file.split(".")[0], "masks": []}
    masks = generator.generate(image)
    for i,mask in enumerate(masks):
        np.savez_compressed(os.path.join(save_mask_loc,f"mask{i:03d}.npz"),
                mask["segmentation"].astype(bool))
        res = {"bbox": mask["bbox"],
                "stability_score": mask["stability_score"]}
        frame_data["masks"].append(res)
    with open(os.path.join(save_mask_loc, "instseg.json"), 'w') as json_file:
        json.dump(frame_data, json_file, indent=4)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="rgb images data location")
    parser.add_argument("-c", "--config", required=True, help="dir containing *sam2.1_hiera_large* model config (.yaml) and weights (.pt)")
    parser.add_argument("-s", "--start", type=int, required=False, help="start at image N", default=None)
    parser.add_argument("-e", "--end", type=int, required=False, help="end at image N (not inclusive)", default=None)
    parser.add_argument("-l", "--save_loc", required=True, help="output location")
    args = parser.parse_args()

    set_config_location(args.config)
    sam2_checkpoint = os.path.join(args.config ,"sam2.1_hiera_base_plus.pt")
    model_cfg = "sam2.1_hiera_b+.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side= 32,
        points_per_batch=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        box_nms_thresh=0.45,
    )

    save_loc = os.path.join(args.save_loc, "segmentations")
    os.makedirs(os.path.join(args.save_loc), exist_ok=False)

    img_files = sorted(os.listdir(args.data))[args.start : args.end]
    for img_file in tqdm(img_files, desc="SAM", unit="frame"):
        generate_masks(img_file, args.data, mask_generator, save_loc)


    print(f"saved to {save_loc}/")

