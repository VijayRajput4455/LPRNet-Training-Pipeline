# -*- coding: utf-8 -*-
# /usr/bin/env/python3

"""PyTorch implementation for LPRNet training."""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from lprnet.data.loader import CHARS, LPRDataLoader
from lprnet.model.lprnet import build_lprnet


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def sparse_tuple_for_ctc(t_length, lengths):
    input_lengths = []
    target_lengths = []

    for length in lengths:
        input_lengths.append(t_length)
        target_lengths.append(length)

    return tuple(input_lengths), tuple(target_lengths)


def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """Set optimizer learning rate according to the epoch schedule."""
    lr = 0
    for i, epoch_boundary in enumerate(lr_schedule):
        if cur_epoch < epoch_boundary:
            lr = base_lr * (0.1 ** i)
            break

    if lr == 0:
        lr = base_lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def get_parser():
    parser = argparse.ArgumentParser(description="parameters to train net")
    parser.add_argument("--max_epoch", default=100, type=int, help="epoch to train the network")
    parser.add_argument("--img_size", default=[94, 24], help="the image size")
    parser.add_argument("--train_img_dirs", default="./data/Dataset", help="the train images path")
    parser.add_argument("--test_img_dirs", default="./data/Dataset", help="the test images path")
    parser.add_argument("--dropout_rate", default=0.5, type=float, help="dropout rate")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="base value of learning rate")
    parser.add_argument("--lpr_max_len", default=8, type=int, help="license plate number max length")
    parser.add_argument("--train_batch_size", default=128, type=int, help="training batch size")
    parser.add_argument("--test_batch_size", default=120, type=int, help="testing batch size")
    parser.add_argument("--phase_train", default=True, type=bool, help="train or test phase flag")
    parser.add_argument("--num_workers", default=8, type=int, help="workers used in dataloading")
    parser.add_argument("--cuda", default=True, type=bool, help="use CUDA to train model")
    parser.add_argument("--resume_epoch", default=0, type=int, help="resume epoch for retraining")
    parser.add_argument("--save_interval", default=2000, type=int, help="checkpoint save interval")
    parser.add_argument("--test_interval", default=2000, type=int, help="evaluation interval")
    parser.add_argument("--momentum", default=0.9, type=float, help="optimizer momentum")
    parser.add_argument("--weight_decay", default=2e-5, type=float, help="weight decay")
    parser.add_argument("--lr_schedule", default=[4, 8, 12, 14, 16], help="learning-rate schedule")
    parser.add_argument("--save_folder", default="./weights/", help="location to save checkpoint models")
    parser.add_argument("--pretrained_model", default="", help="pretrained model path")
    return parser.parse_args()


def resolve_input_dirs(path_string):
    resolved_dirs = []

    for raw_path in path_string.split(","):
        raw_path = os.path.expanduser(raw_path.strip())
        if not raw_path:
            continue

        candidate_paths = []
        if os.path.isabs(raw_path):
            candidate_paths.append(raw_path)
        else:
            candidate_paths.extend(
                [
                    os.path.abspath(raw_path),
                    os.path.abspath(os.path.join(SCRIPT_DIR, raw_path)),
                    os.path.abspath(os.path.join(PROJECT_ROOT, raw_path)),
                ]
            )

        normalized_paths = []
        for candidate in candidate_paths:
            normalized = os.path.normpath(candidate)
            if normalized not in normalized_paths:
                normalized_paths.append(normalized)

        matched = next((candidate for candidate in normalized_paths if os.path.exists(candidate)), None)
        resolved_dirs.append(matched or normalized_paths[0])

    return resolved_dirs


def collate_fn(batch):
    images = []
    labels = []
    lengths = []

    for image, label, length in batch:
        images.append(torch.from_numpy(image))
        labels.extend(label)
        lengths.append(length)

    labels = np.asarray(labels).flatten().astype(np.int64)
    return torch.stack(images, 0), torch.from_numpy(labels), lengths


def train():
    args = get_parser()

    t_length = 18
    epoch = args.resume_epoch

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

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
        def weights_init(module):
            for key in module.state_dict():
                if key.split(".")[-1] == "weight":
                    if "conv" in key:
                        nn.init.kaiming_normal_(module.state_dict()[key], mode="fan_out")
                    if "bn" in key:
                        nn.init.xavier_uniform_(module.state_dict()[key])
                elif key.split(".")[-1] == "bias":
                    module.state_dict()[key][...] = 0.01

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("Initialize net weights successful!")

    optimizer = optim.RMSprop(
        lprnet.parameters(),
        lr=args.learning_rate,
        alpha=0.9,
        eps=1e-08,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    train_img_dirs = resolve_input_dirs(args.train_img_dirs)
    test_img_dirs = resolve_input_dirs(args.test_img_dirs)
    train_dataset = LPRDataLoader(train_img_dirs, args.img_size, args.lpr_max_len)
    test_dataset = LPRDataLoader(test_img_dirs, args.img_size, args.lpr_max_len)

    if len(train_dataset) == 0:
        raise ValueError(f"No training images found. Checked: {', '.join(train_img_dirs)}")
    if len(test_dataset) == 0:
        raise ValueError(f"No test images found. Checked: {', '.join(test_img_dirs)}")

    train_batch_size = min(args.train_batch_size, len(train_dataset))
    test_batch_size = min(args.test_batch_size, len(test_dataset))
    epoch_size = int(math.ceil(len(train_dataset) / float(train_batch_size)))
    max_iter = args.max_epoch * epoch_size

    print(
        "[Info] Train samples: {} | Test samples: {} | Train batch size: {} | "
        "Test batch size: {} | Steps per epoch: {} | Epochs: {}".format(
            len(train_dataset),
            len(test_dataset),
            train_batch_size,
            test_batch_size,
            epoch_size,
            args.max_epoch,
        )
    )

    ctc_loss = nn.CTCLoss(blank=len(CHARS) - 1, reduction="mean")
    start_iter = args.resume_epoch * epoch_size if args.resume_epoch > 0 else 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            batch_iterator = iter(
                DataLoader(
                    train_dataset,
                    train_batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    collate_fn=collate_fn,
                )
            )
            epoch += 1

        if iteration != 0 and iteration % args.save_interval == 0:
            torch.save(lprnet.state_dict(), f"{args.save_folder}LPRNet__iteration_{iteration}.pth")

        if (iteration + 1) % args.test_interval == 0:
            Greedy_Decode_Eval(lprnet, test_dataset, args)

        start_time = time.time()
        images, labels, lengths = next(batch_iterator)
        input_lengths, target_lengths = sparse_tuple_for_ctc(t_length, lengths)
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)

        if args.cuda:
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(labels, requires_grad=False).cuda()
        else:
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

        logits = lprnet(images)
        log_probs = logits.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()

        optimizer.zero_grad()
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)

        if loss.item() == np.inf:
            continue

        loss.backward()
        optimizer.step()

        end_time = time.time()
        log_interval = 1 if epoch_size <= 20 else 20
        if iteration % log_interval == 0 or iteration == max_iter - 1:
            print(
                f"Epoch: {epoch} || epochiter: {iteration % epoch_size}/{epoch_size} || "
                f"Total iter: {iteration} || Loss: {loss.item():.4f} || "
                f"Batch time: {end_time - start_time:.4f} sec || LR: {lr:.8f}"
            )

    print("Final test Accuracy:")
    Greedy_Decode_Eval(lprnet, test_dataset, args)
    torch.save(lprnet.state_dict(), f"{args.save_folder}Final_LPRNet_model.pth")


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


if __name__ == "__main__":
    train()
