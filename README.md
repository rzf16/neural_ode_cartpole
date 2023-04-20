# Neural ODEs for Cartpole Dynamics
This repository contains the code for rzfeng's Robot Learning final project. It implements a simulated cartpole system, residual neural network and neural ODE dynamics models, and MPPI for swing-up control.

## Dependencies
The following Python3 packages are required:
```
torch
numpy
matplotlib
torchdiffeq
seaborn
pyyaml
lightning
tqdm
```
All dependencies can be installed with `pip3 install <package>`.

## Running the Code
The cartpole swing-up task can be run with `python3 main.py`. Task, dynamics, control, and visualization parameters can be tweaked in `cfg.yaml`.

## Learning the Dynamics
Pre-trained models are available in `models/`. To train your own model, first collect some data with `python3 collect_data.py`. Parameters for the data collection process can be seen in `data_collection_cfg.yaml` and by running `python3 collect_data.py -h`. Then, we can train a residual model with `python3 train_ann.py` (see `ann_cfg.yaml` and `python3 train_ann.py -h` for parameters) or a neural ODE with `python3 train_ode.py` (see `ode_cfg.yaml` and `python3 train_ode.py -h` for parameters). Finally, we can compare our model against other models by running `python3 compare_models.py` (see `compare_cfg.yaml` and `compare_models train_ode.py -h` for parameters) or use it for control by setting the dynamics model parameters in `cfg.yaml` and running `python3 main.py`!