import os
import numpy as np

class clipReader:
    def __init__(self, path):
        self.path = path
        self.files = sorted(os.listdir(os.path.join(path, "embeddings")))

    embs = lambda self,x: os.path.join(self.path,"embeddings",x)
    pixs = lambda self,x: os.path.join(self.path,"pixel2embed",x)

    def __getitem__(self, idx):
        embeddings = np.load(self.embs(self.files[idx]))["arr_0"]
        pixel2embs = np.load(self.pixs(self.files[idx]))["arr_0"].astype(int)
        embed_img = np.zeros((*pixel2embs.shape, embeddings.shape[-1]), dtype=np.float32)

        mask = pixel2embs >= 0
        embed_img[mask] = embeddings[pixel2embs[mask]]
        return {"frame":self.files[idx][:-4], "embeddings":embed_img}
    
    def __len__(self):
        return len(self.files)

    def all_embeddings(self):
        all_embeds = []
        for i in range(len(self)):
            embeddings = np.load(self.embs(self.files[i]))["arr_0"]
            emb_list = [embeddings[i] for i in range(embeddings.shape[0])]
            all_embeds.extend(emb_list)
        return np.vstack(all_embeds)


if __name__ == "__main__":
    cr = clipReader("/home/akshaysm/cv2/cv2local/res/embed/5c612198")
    print(cr.scene_embeddings().shape)