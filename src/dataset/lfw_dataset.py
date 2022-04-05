import os
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms as T
import cv2


def load_image(path):
    image = cv2.imread(path)
    image = image.astype(np.float32, copy=False)
    image /= 255.0
    image = image * 2.0 - 1.0
    return image

class LFWDataset(Dataset):

    def __init__(self, cfg, train=True):
        self.train = train
        lfw_dir = os.path.join(cfg["data_dir"], "lfw")
        lfw_list_path = os.path.join(cfg["data_dir"], "lfw_pair.txt")

        with open(lfw_list_path, 'r') as f:
            img_fnames = f.readlines()

        # img_fnames = [os.path.join(lfw_dir, fname) for fname in img_fnames]
        # img_fnames = np.random.shuffle(img_fnames)
        self.img_pairs = []
        self.targets = []
        for fname in img_fnames:
            fname0, fname1, target = fname.split()
            fname0 = os.path.join(lfw_dir, fname0)
            fname1 = os.path.join(lfw_dir, fname1)
            target = np.float32(target)
            img0 = load_image(fname0)
            img1 = load_image(fname1)
            self.img_pairs.append([img0, img1])
            self.targets.append(target)


    def __getitem__(self, idx):
        rtn_dict = {}
        img_pair = self.img_pairs[idx]
        target = self.targets[idx]
        rtn_dict["input"] = img_pair
        rtn_dict["target"] = target
        return rtn_dict

    def __len__(self):
        return len(self.img_pairs)

    def collate_batch(self, batch):
        rtn_dict = {}
        for sample in batch:
            for key, val in sample.items():
                if key not in rtn_dict:
                    rtn_dict[key] = [val]
                else:
                    rtn_dict[key].append(val)

        for key, _ in rtn_dict.items():
            rtn_dict[key] = np.array(rtn_dict[key])
            
        return rtn_dict
        


if __name__ == "__main__":
    cfg = {"data_dir": "./data/LFW"}
    lfw_dataset = LFWDataset(cfg)
    for idx, (img_pair, target) in enumerate(lfw_dataset):
        pass
    print("Finish")