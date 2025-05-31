import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from autoencoder import Autoencoder
from readers.clip_reader import clipReader
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def init_autoenc(learning_rate=1e-4):
    encoder_dims = [256, 128, 64, 32, 3]
    decoder_dims = [16, 32, 64, 128, 256, 256, 512]
    model = Autoencoder(encoder_dims, decoder_dims)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    return model, criterion, optimizer, learning_rate

def init_dataloader(data_path, batch_size=64):
    clip_embeds = clipReader(data_path).all_embeddings()
    clip_embeds = torch.from_numpy(clip_embeds).to(device)
    dataset = TensorDataset(clip_embeds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def training_loop(model, criterion, optimizer, dataloader, epochs=500):
    for epoch in range(epochs):
        running_loss = 0.0
        for data in dataloader:
            x = data[0]
            recon = model(x)
            loss = criterion(recon, x)
            running_loss += loss.item()
            cos_sim = torch.mean(F.cosine_similarity(recon, x, dim=1)).item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch%20==0) or (epoch==epochs-1):
            epoch_display = epoch+1 if epoch == epochs-1 else epoch
            print(f"[{epoch_display}/{epochs} epochs] running loss: {running_loss:.6f} |",
                  f"cosine similarity {cos_sim:.4f}")
            
    return model, optimizer, running_loss
            
def test_reconstruction(model, dataloader):
    model.eval()
    cos_sims = []
    with torch.no_grad():
        for data in dataloader:
            x = data[0]
            encode = model.encode(x)
            decode = model.decode(encode)
            batch_cos_sim = F.cosine_similarity(decode, x, dim=1)
            cos_sims.append(torch.mean(batch_cos_sim).item())

    print(f"mean reconstruction cosine similarity: {np.mean(cos_sims):.4f}")
    return

def save_model(model, optimizer, lr, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': lr
    }, path)
    print(f"saved to {path}")

def load_model(path):
    checkpoint = torch.load(path, map_location=device)
    model, criterion, optimizer, lr = init_autoenc(learning_rate=checkpoint['learning_rate'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    
    return model, criterion, optimizer, lr


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="clip_images.py result location")
    parser.add_argument("-e", "--epochs", required=False, type=int, help="number of epochs", default=500)
    parser.add_argument("-l", "--save_loc", required=False, help="directory to save model to", default=None)
    args = parser.parse_args()

    if args.save_loc is None:
        args.save_loc = os.path.join(args.data, "ae_model.pth")

    model, criterion, optimizer, lr = init_autoenc()
    loader = init_dataloader(args.data)
    training_loop(model, criterion, optimizer, loader, epochs=args.epochs)
    save_model(model, optimizer, lr, args.save_loc)

    model2,_,_,_ = load_model(args.save_loc)
    test_reconstruction(model2, loader)

