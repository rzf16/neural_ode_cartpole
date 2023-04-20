'''
Script for model comparison
Author: rzfeng
'''
import argparse
import time
import pickle
import yaml

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.dataset import MultistepDataset
from src.ann import ResidualDynamicsANN
from src.neural_ode import DynamicsNeuralODE


DATA_PATH = "data/training/piecewise.pkl"
CFG_PATH = "cfg.yaml"
COMPARE_CFG_PATH = "compare_cfg.yaml"


def main():
    parser = argparse.ArgumentParser(description="Cartpole Model Comparison")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("-n", "--n_trajs", type=int, default=200, help="number of trajectories to evaluate with")
    parser.add_argument("-t", "--time_horizon", type=int, default=5, help="timesteps to predict")
    opts = parser.parse_args()
    cfg = yaml.safe_load(open(CFG_PATH))
    compare_cfg = yaml.safe_load(open(COMPARE_CFG_PATH))

    data = pickle.load(open(DATA_PATH, "rb"))[-opts.n_trajs:]
    dataset = MultistepDataset(data, opts.time_horizon)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False)

    model_metrics = {}
    for model_info in compare_cfg["models"]:
        if model_info["type"] == "residual_nn":
            model = ResidualDynamicsANN(4,1)
            model.load_state_dict(torch.load(model_info["path"]))
        elif model_info["type"] == "neural_ode":
            model = DynamicsNeuralODE(4, 1, cfg["dt"],
                                      model_info["parameters"]["n_layers"],
                                      model_info["parameters"]["width"],
                                      method=model_info["parameters"]["method"],
                                      rtol=model_info["parameters"]["rtol"],
                                      atol=model_info["parameters"]["atol"],
                                      options=model_info["parameters"]["options"],)
            model.load_state_dict(torch.load(model_info["path"]))
        else:
            print("[Model Comparison] Error! Unrecognized model type.")
            return

        trainer = pl.Trainer(enable_model_summary=False, enable_progress_bar=False)
        tic = time.time()
        metrics = trainer.test(model, dataloader, verbose=False)
        toc = time.time()
        metrics.append({"traj inference time": (toc - tic) / len(dataloader)})
        model_metrics[model_info["name"]] = metrics

    print("\n")
    for name, metrics in model_metrics.items():
        print(f"{name} performance:")
        for metrics_i in metrics:
            for k, v in metrics_i.items():
                print(f"{k:20s}: {v:0.6f}")
        print("\n")


if __name__ == "__main__":
    main()