'''
Data collection script for cartpole
Author: rzfeng
'''
from typing import Optional
import argparse
import os
import pickle
import yaml

import torch
from torch.distributions.uniform import Uniform

from tqdm import tqdm

from src.cartpole import Cartpole
from src.recorder import Trajectory

CFG_PATH = "cfg.yaml"
DATA_COLLECTION_CFG_PATH = "data_collection_cfg.yaml"


def main():
    parser = argparse.ArgumentParser(description="Cartpole Data Collection")
    parser.add_argument("-n", "--n_trajs", type=int, default=1000, help="number of trajectories to collect")
    parser.add_argument("-l", "--traj_length", type=int, default=50, help="trajectory length")
    parser.add_argument("-d", "--discrete", action="store_true", help="use discrete dynamics")
    opts = parser.parse_args()
    cfg = yaml.safe_load(open(CFG_PATH))
    data_collection_cfg = yaml.safe_load(open(DATA_COLLECTION_CFG_PATH))

    filename = data_collection_cfg["filename"]
    if os.path.exists(filename):
        print(f"[Data Collection] Error! File {filename} already exists D:")
        return

    cartpole = Cartpole(torch.zeros(Cartpole.state_dim()), cfg["system"]["m_cart"], cfg["system"]["m_pole"],
                        cfg["system"]["l"], cfg["system"]["vis_params"])
    data = collect_data(cartpole,
                        opts.n_trajs,
                        opts.traj_length,
                        cfg["dt"],
                        torch.tensor(cfg["system"]["u_min"]), torch.tensor(cfg["system"]["u_max"]),
                        torch.tensor(cfg["system"]["s_min"]), torch.tensor(cfg["system"]["s_max"]),
                        torch.tensor(data_collection_cfg["s0_min"]), torch.tensor(data_collection_cfg["s0_max"]),
                        (not opts.discrete))
    pickle.dump(data, open(filename, "wb"))


# Collects data for a cartpole system
# @input cartpole [Cartpole]: Cartpole object
# @input n_trajs [int]: number of trajectories to collect
# @input traj_length [int]: trajectory length
# @input dt [float]: time step
# @input u_min [torch.tensor (control_dim)]: minimum action
# @input u_max [torch.tensor (control_dim)]: maximum action
# @input s_min [torch.tensor (state_dim)]: minimum state
# @input s_max [torch.tensor (state_dim)]: maximum state
# @input s0_min [Optional[torch.tensor (state_dim)]]: minimum start state
# @input s0_max [Optional[torch.tensor (state_dim)]]: maximum start state
# @input continuous [bool]: use continuous dynamics
# @output [List[Dict(str: Trajectory)]]: list of collected state and control trajectories
def collect_data(cartpole: Cartpole, n_trajs: int, traj_length: int, dt: float,
                 u_min: torch.tensor, u_max: torch.tensor,
                 s_min: torch.tensor, s_max: torch.tensor,
                 s0_min: Optional[torch.tensor] = None, s0_max: Optional[torch.tensor] = None, continuous: bool = True):
    data = []
    u_dist = Uniform(u_min, u_max)
    s0_dist = Uniform(s0_min, s0_max) if (s0_min is not None and s0_max is not None) else Uniform(s_min, s_max)

    for i in tqdm(range(n_trajs)):
        while True:
            state_traj = Trajectory([],[])
            control_traj = Trajectory([],[])

            # Reset cartpole
            s0 = s0_dist.sample()
            cartpole.set_state(s0)
            # state_traj.log_datum(cartpole.get_state(), 0.0)

            # Collect trajectory in a batched fashion
            control = u_dist.sample((traj_length,))
            control_traj.log_data_batch(control, dt*torch.arange(0,traj_length))
            if continuous:
                states, state_times = cartpole.apply_control(control, (0.0,dt*traj_length), t_eval=dt*torch.arange(0,traj_length+1))
            else:
                state_traj.log_datum(s0, 0.0)
                discrete_dynamics = cartpole.generate_discrete_dynamics(dt)
                states = discrete_dynamics(torch.tensor([0.0]), s0.unsqueeze(0), control.unsqueeze(0)).squeeze(0)
                state_times = dt*torch.arange(1,traj_length+1)
            state_traj.log_data_batch(states, state_times)

            success = torch.all(states >= s_min) and torch.all(states <= s_max)

            if success:
                states, state_times = state_traj.as_torch()
                controls, control_times = control_traj.as_torch()
                datum = {
                    "states": states,
                    "state_times": state_times,
                    "controls": controls,
                    "control_times": control_times
                }
                data.append(datum)
                break

    return data


if __name__ == "__main__":
    main()