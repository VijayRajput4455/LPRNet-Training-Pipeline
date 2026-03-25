# -*- coding: utf-8 -*-
# /usr/bin/env/python3

"""Test a pretrained LPRNet model."""

import argparse
import math
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torch.utils.data import DataLoader

from lprnet.data.loader import CHARS, LPRDataLoader
from lprnet.model.lprnet import build_lprnet


def get_parser():
    parser = argparse.ArgumentParser(description="parameters to test net")
    parser.add_argument("--img_size", default=[94, 24], help="the image size")
    parser.add_argument("--test_img_dirs", default="./data/test", help="the test images path")
    parser.add_argument("--dropout_rate", default=0, type=float, help="dropout rate")
    parser.add_argument("--lpr_max_len", default=8, type=int, help="license plate max length")
    parser.add_argument("--test_batch_size", default=100, type=int, help="testing batch size")
    parser.add_argument("--phase_train", default=False, type=bool, help="train or test phase")
    parser.add_argument("--num_workers", default=8, type=int, help="workers used in dataloading")
    parser.add_argument("--cuda", default=True, type=bool, help="use CUDA to test model")
    parser.add_argument("--show", default=False, type=bool, help="show test image and prediction")
    parser.add_argument(
        "--pretrained_model",
        default="./weights/Final_LPRNet_model.pth",
        help="pretrained model path",
    )
    return parser.parse_args()


def collate_fn(batch):
    images = []
    labels = []
    lengths = []

    for image, label, length in batch:
        images.append(torch.from_numpy(image))
        labels.extend(label)
        lengths.append(length)

    labels = np.asarray(labels).flatten().astype(np.float32)
    return torch.stack(images, 0), torch.from_numpy(labels), lengths


def test():
    args = get_parser()

    lprnet = build_lprnet(
        lpr_max_len=args.lpr_max_len,
        phase=args.phase_train,
        class_num=len(CHARS),
        dropout_rate=args.dropout_rate,
    )
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        print("Load pretrained model successful!")
    else:
        print("[Error] Cannot find pretrained model, please check.")
        return False

    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    test_dataset = LPRDataLoader(test_img_dirs.split(","), args.img_size, args.lpr_max_len)

    try:
        Greedy_Decode_Eval(lprnet, test_dataset, args)
    finally:
        cv2.destroyAllWindows()


def Greedy_Decode_Eval(net, datasets, args):
    test_batch_size = min(int(args.test_batch_size), len(datasets))
    epoch_size = int(math.ceil(len(datasets) / float(test_batch_size)))
    batch_iterator = iter(
        DataLoader(
            datasets,
            test_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
    )

    tp = 0
    tn_len_mismatch = 0
    tn_wrong_chars = 0
    t1 = time.time()

    for _ in range(epoch_size):
        images, labels, lengths = next(batch_iterator)

        start = 0
        targets = []
        for length in lengths:
            label = labels[start : start + length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])
        raw_images = images.numpy().copy()

        images = Variable(images.cuda()) if args.cuda else Variable(images)

        probs = net(images)
        probs = probs.cpu().detach().numpy()

        pred_labels = []
        for batch_index in range(probs.shape[0]):
            prob = probs[batch_index, :, :]
            raw_label = []
            for timestep in range(prob.shape[1]):
                raw_label.append(np.argmax(prob[:, timestep], axis=0))

            decoded = []
            previous = raw_label[0]
            if previous != len(CHARS) - 1:
                decoded.append(previous)

            for c in raw_label:
                if previous == c or c == len(CHARS) - 1:
                    if c == len(CHARS) - 1:
                        previous = c
                    continue
                decoded.append(c)
                previous = c
            pred_labels.append(decoded)

        for i, label in enumerate(pred_labels):
            if args.show:
                show(raw_images[i], label, targets[i])

            if len(label) != len(targets[i]):
                tn_len_mismatch += 1
                continue

            if (np.asarray(targets[i]) == np.asarray(label)).all():
                tp += 1
            else:
                tn_wrong_chars += 1

    total = tp + tn_len_mismatch + tn_wrong_chars
    if total == 0:
        print("[Info] Test Accuracy: skipped (no evaluation samples were processed)")
        return

    acc = tp * 1.0 / total
    print(f"[Info] Test Accuracy: {acc} [{tp}:{tn_len_mismatch}:{tn_wrong_chars}:{total}]")
    t2 = time.time()
    print(f"[Info] Test Speed: {(t2 - t1) / len(datasets)}s 1/{len(datasets)}]")


def show(image, label, target):
    image = np.transpose(image, (1, 2, 0))
    image *= 128.0
    image += 127.5
    image = image.astype(np.uint8)

    predicted = "".join(CHARS[i] for i in label)
    ground_truth = "".join(CHARS[int(i)] for i in target.tolist())

    flag = "T" if predicted == ground_truth else "F"
    image = cv2ImgAddText(image, predicted, (0, 0))

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("test")
    plt.axis("off")
    print(f"target: {ground_truth} ### {flag} ### predict: {predicted}")
    plt.show()


def cv2ImgAddText(image, text, pos, textColor=(255, 0, 0), textSize=12):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=font)

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    test()
