import os
import json
import numpy as np

class samReader:
    def __init__(self, path):
        self.path = os.path.join(path, "segmentations")
        self.frames = sorted(
            [i for i in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, i))]
        )

    def read_npz(self, idx):
        seg_mats = sorted(
            [i for i in os.listdir(os.path.join(self.path,self.frames[idx])) \
             if i[-4:] == ".npz"]
        )
        pth = lambda x: os.path.join(self.path,self.frames[idx], x)
        return [np.load(pth(npz)) for npz in seg_mats]

    def read_json(self, idx):
        with open(os.path.join(self.path,self.frames[idx],"instseg.json")) as j:
            return json.load(j)["masks"]

    def __getitem__ (self, idx):
        masks = []
        for js,seg in zip(self.read_json(idx), self.read_npz(idx)):
            js["segmentation"] = seg["arr_0"]
            js["frame"] = self.frames[idx]
            masks.append(js)
        return masks
    
    def __len__(self):
        return len(self.frames)
    
    
if __name__ == "__main__":
    sr = samReader("/home/akshaysm/cv2/cv2local/res/sgmnt/af8a3ce9")
    print(sr[0][0])
