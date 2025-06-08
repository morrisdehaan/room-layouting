import os
import re
import clip
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from scipy.signal import convolve2d
from readers.sam_reader import samReader
from instance_seg import load_image, gaussian_kernel

device = "cuda" if torch.cuda.is_available() else "cpu"

def init_CLIP():
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    clip_model = clip_model.float()
    clip_model.eval();
    return clip_model, clip_preprocess

resize = lambda img: np.array(Image.fromarray(img).resize((224, 224), resample=Image.BILINEAR))

def crop_masks(image, mask):
    bbox = mask["bbox"]
    segm = mask["segmentation"]

    H, W, C = image.shape
    x, y, w, h = bbox
    cx = x + w / 2    
    cy = y + h / 2
    side = int(max(w, h, 224))
    left = int(np.floor(cx - (side // 2)))
    top = int(np.floor(cy - (side // 2)))
    right = left + side
    bottom = top + side

    crop_img = np.zeros((side, side, C), dtype=image.dtype)
    crop_segm = np.zeros((side, side), dtype=image.dtype)

    src_x1 = max(0, left)
    src_y1 = max(0, top)
    src_x2 = min(W, right)
    src_y2 = min(H, bottom)
    dst_x1 = src_x1 - left   
    dst_y1 = src_y1 - top
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    crop_img[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    crop_img = resize(crop_img)

    crop_segm[dst_y1:dst_y2, dst_x1:dst_x2] = segm[src_y1:src_y2, src_x1:src_x2]
    crop_segm = resize(crop_segm)
    crop_segm = crop_segm > 0

    return crop_img, crop_segm

# old
def soft_masking(crop_img, crop_mask, kernel_size=8, sigma=8, bkg_brightness=0.5):
    gauss = gaussian_kernel(kernel_size, sigma)
    blurred = np.zeros_like(crop_img)
    for c in range(3):
        blurred[:,:,c] = bkg_brightness * convolve2d(crop_img[:,:,c], gauss, mode="same", boundary="wrap")

    blurred[crop_mask] = crop_img[crop_mask]
    return blurred

def soft_masking_cuda(crop_img, crop_mask, kernel_size=25, sigma=25):
    img_tensor = torch.from_numpy(crop_img).to(device).permute(2,0,1).unsqueeze(0).float()
    gaussian = torch.from_numpy(gaussian_kernel(kernel_size, sigma)).to(device)
    gaussian = gaussian.view(1,1,kernel_size,kernel_size).repeat(3,1,1,1)
    blurred = F.conv2d(img_tensor, gaussian, padding=kernel_size//2, groups=3)
    blurred = blurred.squeeze(0).permute(1,2,0).cpu().numpy()

    blurred[crop_mask] = crop_img[crop_mask]
    blurred = np.floor(blurred).astype(np.uint8)
    return blurred

def full_masking(crop_img, crop_mask):
    masked = np.zeros_like(crop_img)
    masked[crop_mask] = crop_img[crop_mask]
    return masked

def clip_features(image, clip_model, clip_preprocess, normalized=True):
    image_tensor = clip_preprocess(Image.fromarray(image)).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
    if normalized:
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features

def pixel_embeddings(image_path, masks, clip_model, clip_preprocess, soft_mask=True, save_loc=os.getcwd()):
    image = load_image(image_path, unsharp_mask=False)
    h,w,_ = image.shape
    idx = 0
    embed_id_mat = -np.ones((h,w))  
    stability_mat = np.zeros((h,w))
    embed_mat = torch.zeros((len(masks), 512), dtype=torch.float32, device=device)
    fallback_msg = True

    for mask in masks:
        c_image, c_mask = crop_masks(image, mask)
        if soft_mask:
            try:
                masked_img = soft_masking_cuda(c_image, c_mask)
            except:
                masked_img = soft_masking(c_image, c_mask)
                if fallback_msg:
                    print("falling back to convolution with scipy")
                    fallback_msg = False
        else:
            masked_img = full_masking(c_image, c_mask)

        emb = clip_features(masked_img, clip_model=clip_model, clip_preprocess=clip_preprocess)
        embed_mat[idx] = emb

        for y,x in zip(*np.where(mask["segmentation"])):
            if mask["stability_score"] > stability_mat[y,x]:
                embed_id_mat[y,x] = idx
                stability_mat[y,x] = mask["stability_score"]
        
        idx += 1

    embed_mat = embed_mat.cpu().numpy()
    frameid = re.search(r"frame\d+", image_path).group()
    np.savez_compressed(os.path.join(save_loc,"pixel2embed",f"{frameid}.npz"), embed_id_mat)
    np.savez_compressed(os.path.join(save_loc,"embeddings",f"{frameid}.npz"), embed_mat)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="rgb images data location")
    parser.add_argument("-i", "--instseg", required=True, help="output folder of instance_seg.py")
    parser.add_argument("-s", "--start", type=int, required=False, help="start at image N", default=None)
    parser.add_argument("-e", "--end", type=int, required=False, help="end at image N (not inclusive)", default=None)
    parser.add_argument("-l", "--save_loc", required=False, help="output location", default=None)
    args = parser.parse_args()

    if args.save_loc is None:
        args.save_loc = args.instseg

    os.makedirs(os.path.join(args.save_loc,"embeddings"),exist_ok=False)
    os.makedirs(os.path.join(args.save_loc,"pixel2embed"),exist_ok=False)

    img_files = sorted(os.listdir(args.data))[args.start : args.end]
    SR = samReader(args.instseg)
    clip_model, clip_preprocess = init_CLIP()

    for i, img_file in enumerate(tqdm(img_files, desc="CLIP", unit="frame")):
        img_path = os.path.join(args.data, img_file)
        pixel_embeddings(img_path, SR[i], clip_model=clip_model, clip_preprocess=clip_preprocess,
                         soft_mask=True, save_loc=args.save_loc)

    print(f"saved to {args.save_loc}/")


