'''
Data recording and visualization
Author: rzfeng
'''
import os
import pickle
import shutil
import yaml
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch

from src.cartpole import Cartpole


LOG_PATH = "data/logs/"
TRAINING_DATA_PATH = "data/training/"
MEDIA_DIR = "media/"


# Class for a generic trajectory
@dataclass
class Trajectory:
    data: List[torch.tensor]
    timestamps: List[float]

    def __len__(self):
        return len(self.data)

    def log_datum(self, datum: torch.tensor, time: float):
        self.data.append(datum)
        self.timestamps.append(time)

    def log_data_batch(self, data: torch.tensor, times: torch.tensor):
        self.data.extend(list(data))
        self.timestamps.extend(times.tolist())

    def as_np(self) -> Tuple:
        return torch.stack(self.data).numpy(), np.array(self.timestamps)

    def as_torch(self) -> Tuple:
        return torch.stack(self.data), torch.tensor(self.timestamps)


# Class for data recording and visualization
class DataRecorder:
    # @input keys [List[str]]: list of data keys
    def __init__(self, keys: List[str]):
        self.state_trajs = {key: Trajectory([],[]) for key in keys}
        self.control_traj = Trajectory([],[])

    # Logs a batch of states
    # @input key [str]: data key
    # @input s [torch.tensor (T x state_dim)]: batch of states to add
    # @input t [torch.tensor (T)]: timestamps
    def log_state(self, key: str, s: torch.tensor, t: torch.tensor):
        self.state_trajs[key].log_data_batch(s,t)

    # Logs a batch of controls
    # @input u [torch.tensor (T x control_dim)]: batch of controls to add
    # @input t [torch.tensor (T)]: timestamps
    def log_control(self, u: torch.tensor, t: torch.tensor):
        self.control_traj.log_data_batch(u,t)

    # Writes recorded data to disk as NumPy arrays
    # @input cfg_path [str]: path to configuration file
    # @input prefix [str]: data directory prefix (e.g. "run" -> run001, run002, etc.)
    def write_data(self, cfg_path: str, prefix: str = "run"):
        # Get the current run number and make the corresponding directory
        idx = 1
        write_dir = os.path.join(LOG_PATH, prefix+f"{idx:03}")
        while os.path.exists(write_dir):
            idx += 1
            write_dir = os.path.join(LOG_PATH, prefix+f"{idx:03}")
        os.makedirs(write_dir)

        # Copy the configuration info
        shutil.copy(cfg_path, os.path.join(write_dir, "cfg.yaml"))

        # Write data as NumPy arrays
        controls, control_times = self.control_traj.as_np()
        np.save(os.path.join(write_dir, "controls.npy"), controls)
        np.save(os.path.join(write_dir, "control_times.npy"), control_times)

        for key, state_traj in self.state_trajs.items():
            states, state_times = state_traj.as_np()
            np.save(os.path.join(write_dir, f"{key}_states.npy"), states)
            np.save(os.path.join(write_dir, f"{key}_state_times.npy"), state_times)

        print(f"[Recorder] Data written to {write_dir}!")

    # Loads trajectory data from a directory
    # @input dir [str]: directory to load from (EXCLUDING "data/")
    def from_np(self, dir: str):
        load_dir = os.path.join(LOG_PATH, dir)
        cfg = yaml.safe_load(open(os.path.join(load_dir, "cfg.yaml")))
        keys = [key for key in cfg["data_keys"]] # Is this robust? Maybe use substrings instead?

        # Load data from NumPy arrays
        controls = np.load(os.path.join(load_dir, "controls.npy"))
        control_times = np.load(os.path.join(load_dir, "control_times.npy"))
        controls = [torch.from_numpy(control) for control in controls]
        control_times = control_times.tolist()
        self.control_traj = Trajectory(controls, control_times)

        for key in keys:
            states = np.load(os.path.join(load_dir, f"{key}_states.npy"))
            state_times = np.load(os.path.join(load_dir, f"{key}_state_times.npy"))
            states = [torch.from_numpy(state) for state in states]
            state_times = state_times.tolist()
            self.state_trajs[key] = Trajectory(states, state_times)

    # Loads trajectory data from a pickle file
    # @input filename [str]: filename to load from (EXCLUDING "data/training_data/")
    # @input key [str]: data key to store with
    def from_pkl(self, filename: str, key: str):
        # Load data
        # data = pickle.load(open(os.path.join(TRAINING_DATA_PATH, filename), "rb"))
        data = [pickle.load(open(os.path.join(TRAINING_DATA_PATH, filename), "rb"))[4]]
        states = torch.cat([traj["states"] for traj in data], dim=0)
        state_times = torch.cat([traj["state_times"] for traj in data])
        self.state_trajs[key] = Trajectory(list(states), state_times.tolist())
        controls = torch.cat([traj["controls"] for traj in data], dim=0)
        control_times = torch.cat([traj["control_times"] for traj in data])
        self.control_traj = Trajectory(list(controls), control_times.tolist())

    # Plots the state trajectory
    def plot_state_trajs(self):
        layout = Cartpole.get_state_plot_layout()
        description = Cartpole.get_state_description()

        fig, ax = plt.subplots(nrows=layout.shape[0], ncols=layout.shape[1], sharex=True, squeeze=False)
        fig.tight_layout()
        fig.supxlabel("time")
        fig.suptitle("cartpole state trajectory")
        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                if layout[i,j] >= 0:
                    for key, state_traj in self.state_trajs.items():
                        states, times = state_traj.as_np()
                        col.plot(times, states[:,layout[i,j]], label=key)
                        col.set_ylabel(description[layout[i,j]].name)

        handles, labels = ax[-1,-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper right")
        plt.show()

    # Plots the control trajectory
    def plot_control_traj(self):
        controls, times = self.control_traj.as_np()
        layout = Cartpole.get_control_plot_layout()
        description = Cartpole.get_control_description()

        fig, ax = plt.subplots(nrows=layout.shape[0], ncols=layout.shape[1], sharex=True, squeeze=False)
        fig.tight_layout()
        fig.supxlabel("time")
        fig.suptitle("cartpole control trajectory")
        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                if layout[i,j] >= 0:
                    col.plot(times, controls[:,layout[i,j]])
                    col.set_ylabel(description[layout[i,j]].name)

        plt.show()

    # Animates the 2D trajectory of vehicles
    # @input cartpole [Cartpole]: Cartpole object
    # @input key [str]: key to visualize
    # @input n_frames [Optional[int]]: number of frames to animate
    # @input fps [int]: frames per second
    # @input end_wait [float]: seconds to wait at the end before finishing the animation
    # @input write [Optional[str]]: filename to write the animation to (EXCLUDING "media/""); None indicates not to write
    def animate2d(self, cartpole: Cartpole, key: str, n_frames: Optional[int] = None, fps: int = 5, end_wait: float = 1.0, write: Optional[str] = None):
        fig = plt.figure()
        ax = plt.axes(aspect="equal")

        states, times = self.state_trajs[key].as_np()

        # Plot all patches to get the axis limits
        for state in states:
            cartpole.add_vis2d(ax, torch.from_numpy(state))
        ax.autoscale_view()
        # Now clear the axes and set the axis limits
        lims = (ax.get_xlim(), ax.get_ylim())
        ax.clear()
        ax.set_xlim(lims[0])
        ax.set_ylim((-4.0*cartpole.l, cartpole.vis_params["cart_height"] + 4.0 * cartpole.l))
        ax.autoscale(False)

        anim_artists = []
        # Animation function for Matplotlib
        def anim_fn(t):
            # Get the appropriate index for the time point
            n = np.searchsorted(times, t)
            if n >= times.size:
                return []

            # Replace previous drawings
            for artist in anim_artists:
                artist.remove()
            anim_artists.clear()
            anim_artists.extend(cartpole.add_vis2d(ax, torch.from_numpy(states[n,:])))

            return anim_artists

        max_t = times.max()
        n_frames = times.size if n_frames is None else n_frames
        end_buffer = int(np.ceil(end_wait * fps))
        frame_iter = np.append(np.linspace(0.0, max_t, n_frames), max_t * np.ones(end_buffer))
        anim = animation.FuncAnimation(fig, anim_fn, frames=frame_iter, interval = 1000.0 / fps, blit=True)

        ax.set_title("cartpole animation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if write is not None:
            write_path = os.path.join(MEDIA_DIR, write)
            overwrite = True
            if os.path.exists(write_path):
                overwrite = input(f"[Recorder] Write path {write_path} exists. Overwrite? (y/n) ") == "y"
            if overwrite:
                anim.save(os.path.join(MEDIA_DIR, write))

        plt.show()