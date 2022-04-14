import os
import numpy as np
import numbers

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import mxnet as mx

import matplotlib.pyplot as plt
import cv2


# dataset from insightface
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/dataset.py

class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank=0):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
            #  transforms.Grayscale(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.0], std=[1.0]),
            #  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])

        # self.transform = None

        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(
            path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        # index = 1
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img, flag=0).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        # return len(self.imgidx)
        return 12800 * 4


if __name__ == "__main__":
    dataset = MXFaceDataset("./data/ms1m-retinaface-t1")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
        shuffle=False,
        # collate_fn=dataset.collate_batch,
    )

    # images = []
    labels = []

    transform = transforms.Compose(
        [transforms.ToPILImage(),
        #  transforms.Grayscale()
         ])

    sample_dir = "./samples"
    os.makedirs(sample_dir, exist_ok=True)

    for idx, batch in enumerate(dataloader):
        sample, label = batch
        # images.append(image)
        labels.append(label.item())

        if idx > 5:
            continue

        sample = sample[0].numpy().transpose(1, 2, 0)
        sample *= 255
        sample = sample.astype("uint8")
        # sample = transform(sample)
        cv2.imwrite(f"{sample_dir}/sample_{idx}.png", sample)
        # sample.save(f"{sample_dir}/sample_{idx}.png")

    labels = np.array(labels)

    plt.hist(labels, bins=100)
    plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    plt.savefig("label_dist.png")

    labels_unique = np.unique(labels)
    print(f"num_classes: {len(labels_unique)}")
