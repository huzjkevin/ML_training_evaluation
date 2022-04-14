import torch
import torchvision
from torch import optim
from torchvision import transforms

import numpy as np
import argparse
import yaml
import os
import time
import cv2
import matplotlib.pyplot as plt

from src.dataset.lfw_get_dataloader import get_test_dataloader
from src.pipeline.pipeline import Pipeline
from src.model.get_model import get_model


def run_evaluation(model, cfg):
    # test_loader = get_test_dataloader(
    #     data_path=cfg["dataset"]["data_dir"], batch_size=cfg["dataloader"]["batch_size"], num_workers=cfg["dataloader"]["num_workers"]
    # )
    # pipeline.hw_evaluate(model, test_loader, tb_prefix="TEST")
    identity_list = get_lfw_list(cfg["dataset"]["test_list"])
    img_paths = [os.path.join(cfg["dataset"]["data_dir"], each)
                 for each in identity_list]

    model.eval()
    lfw_test(model,
             img_paths,
             identity_list,
             cfg["dataset"]["test_list"],
             cfg["dataloader"]["batch_size"])


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


# def load_image(img_path):
#     image = cv2.imread(img_path, 1)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # image = cv2.imread(img_path, 0)
#     if image is None:
#         return None
#     image = np.array([image, image[:, ::-1, :]])
#     image = image.transpose((0, 3, 1, 2))
#     image = image.astype(np.float32, copy=False)
#     image = image / 255.0
#     image = image * 2.0 - 1.0
#     return image

def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def load_image_torch(img_path):
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.0], std=[1.0]),
         ])
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image_flip = np.fliplr(image)
    image = torch.unsqueeze(transform(image), dim=0)
    image_flip = torch.unsqueeze(transform(image_flip), dim=0)
    image = torch.cat((image, image_flip), dim=0)
    return image

# def get_featurs(model, test_list, batch_size=10):
#     images = None
#     features = None
#     cnt = 0
#     for i, img_path in enumerate(test_list):
#         image = load_image(img_path)
#         if image is None:
#             print('read {} error'.format(img_path))

#         if images is None:
#             images = image
#         else:
#             images = np.concatenate((images, image), axis=0)

#         if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
#             cnt += 1

#             data = torch.from_numpy(images)
#             data = data.to(torch.device("cuda"))
#             output = model(data)
#             output = output.data.cpu().numpy()

#             fe_1 = output[::2]
#             fe_2 = output[1::2]
#             feature = np.hstack((fe_1, fe_2))
#             # print(feature.shape)

#             if features is None:
#                 features = feature
#             else:
#                 features = np.vstack((features, feature))

#             images = None

#     return features, cnt


def get_featurs(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        # image = load_image(img_path)
        image = load_image_torch(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            # images = np.concatenate((images, image), axis=0)
            images = torch.cat((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            # images = torch.from_numpy(images)
            images = images.to(torch.device("cuda"))
            output = model(images)
            output = output.data.cpu().numpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            # print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)

    plt.hist(y_score, bins=100)
    plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    plt.savefig("score_dist.png")

    best_acc = 0
    # th = 0.85
    # best_th = th
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--evaluation", default=False, action="store_true")
    parser.add_argument("--cfg", type=str, required=False,
                        default="./cfgs/learn_arcface.yaml")
    parser.add_argument("--ckpt", type=str, required=False, default=None)
    parser.add_argument("--cont", default=False, action="store_true")
    args = parser.parse_args()

    # open config files
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
        cfg["pipeline"]["Logger"]["backup_list"].append(args.cfg)

    cfg["dataset"]["data_dir"] = "./data/LFW/lfw-align-128"
    cfg["dataset"]["test_list"] = "./data/LFW/lfw_pair.txt"

    model = get_model(cfg)
    model.cuda()

    pipeline = Pipeline(model, cfg["pipeline"])

    if args.ckpt:
        pipeline.load_ckpt(model, args.ckpt)

    run_evaluation(model, cfg)

# if __name__ == '__main__':

#     opt = Config()
#     if opt.backbone == 'resnet18':
#         model = resnet_face18(opt.use_se)
#     elif opt.backbone == 'resnet34':
#         model = resnet34()
#     elif opt.backbone == 'resnet50':
#         model = resnet50()

#     model = DataParallel(model)
#     # load_model(model, opt.test_model_path)
#     # model.load_state_dict(torch.load(opt.test_model_path))
#     model.to(torch.device("cuda"))

#     identity_list = get_lfw_list(opt.lfw_test_list)
#     img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

#     model.eval()
#     lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
