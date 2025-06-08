import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from ae_training import load_model
from readers.clip_reader import clipReader

device = "cuda" if torch.cuda.is_available() else "cpu"

def latent_frame(embed, ae_model):
    embed = torch.from_numpy(embed).to(device)
    H,W,_ = embed.shape
    embed_flat = embed.view(-1, 512)

    with torch.no_grad():
        latent = ae_model.encode(embed_flat)

    _,L = latent.shape
    latent = latent.view(H,W,L)
    latent = latent.cpu().numpy()
    return latent

def save_frame(frame, latent, location):
    np.savez_compressed(os.path.join(location,f"{frame}.npz"), latent)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="clip_images.py result location")
    parser.add_argument("-l", "--save_loc", required=False, help="directory to save model to", default=None)
    args = parser.parse_args()

    if args.save_loc is None:
        args.save_loc = os.path.join(args.data, "latentvid")
    os.makedirs(args.save_loc, exist_ok=False)

    CR = clipReader(args.data)
    ae_model,_,_,_ = load_model(os.path.join(args.data, "ae_model.pth"))

    for emb in tqdm(CR, desc="Encode to latent", unit="frame"):
        latent = latent_frame(emb["embeddings"], ae_model)
        save_frame(emb["frame"], latent, args.save_loc)
    print(f"saved to {args.save_loc}/")
