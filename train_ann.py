'''
Script for learning residual cartpole dynamics with a basic ANN
Author: rzfeng
'''
import os
import argparse
import yaml
import pickle

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from src.ann import ResidualDynamicsANN
from src.dataset import MultistepDataset


DATA_PATH = "data/training/piecewise.pkl"
OUT_PATH = "models/residual_nn.pth"
ANN_CFG_PATH = "ann_cfg.yaml"


def main():
    parser = argparse.ArgumentParser(description="Cartpole ANN Dynamics Learning")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-e", "--max_epochs", type=int, default=-1, help="maximum training epochs")
    parser.add_argument("-s", "--max_steps", type=int, default=-1, help="maximum training steps")
    parser.add_argument("-t", "--training_split", type=float, default=0.8, help="training data fraction")
    parser.add_argument("-g", "--gpu", action="store_true", help="use gpu")
    parser.add_argument("-c", "--cores", type=int, default=1, help="number of cpu cores")
    opts = parser.parse_args()
    ann_cfg = yaml.safe_load(open(ANN_CFG_PATH))

    if opts.max_epochs == -1 and opts.max_steps == -1:
        print("[Train ANN] Error! Please set max steps with -s or max epochs with -e.")
        return

    if opts.max_steps != -1:
        opts.max_steps = int(opts.max_steps / opts.batch_size)

    if os.path.exists(OUT_PATH):
        print("[Train ANN] Error! Model output path {} exists!".format(OUT_PATH))
        return

    data = pickle.load(open(DATA_PATH, "rb"))
    dataset = MultistepDataset(data, ann_cfg["n_steps"])
    train_size = int(len(dataset) * opts.training_split)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_set, batch_size=opts.batch_size, shuffle=True, num_workers=opts.cores)
    val_dataloader = DataLoader(val_set, batch_size=opts.batch_size)

    device = "cuda:0" if opts.gpu else "cpu"
    model = ResidualDynamicsANN(data[0]["states"].size(1), data[0]["controls"].size(1)).to(device)

    if opts.gpu:
        trainer = pl.Trainer(accelerator="gpu", devices=1, max_steps=opts.max_steps, max_epochs=opts.max_epochs,
                             log_every_n_steps=1)
    else:
        trainer = pl.Trainer(max_steps=opts.max_steps, max_epochs=opts.max_epochs,
                             log_every_n_steps=1)
    trainer.fit(model, train_dataloader)

    metrics = trainer.test(model, val_dataloader)
    for metrics_i in metrics:
        for k, v in metrics_i.items():
            print(f"{k:20s}: {v:0.6f}")

    torch.save(model.state_dict(), OUT_PATH)


if __name__ == "__main__":
    main()