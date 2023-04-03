'''
Data recording and visualization
Author: rzfeng
'''
import os
import shutil
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch

from src.cartpole import Cartpole


DATA_PATH = "data/"
MEDIA_DIR = "media/"


# Class for a generic trajectory
@dataclass
class Trajectory:
    data: List[torch.tensor]
    timestamps: List[float]

    def as_np(self) -> Tuple:
        return torch.stack(self.data).numpy(), np.array(self.timestamps)


# Class for system data
@dataclass
class SystemTrajectory:
    state_traj: Trajectory
    control_traj: Trajectory


# Class for data recording and visualization
class DataRecorder:
    def __init__(self):
        self.data = SystemTrajectory(Trajectory([],[]),Trajectory([],[]))

    # Logs a batch of states
    # @input s [torch.tensor (T x state_dim)]: batch of states to add
    # @input t [torch.tensor (T)]: timestamps
    def log_state(self, s: torch.tensor, t: torch.tensor):
        self.data.state_traj.data.extend(list(s))
        self.data.state_traj.timestamps.extend(t.tolist())

    # Logs a batch of controls
    # @input u [torch.tensor (T x control_dim)]: batch of controls to add
    # @input t [torch.tensor (T)]: timestamps
    def log_control(self, u: torch.tensor, t: torch.tensor):
        self.data.control_traj.data.extend(list(u))
        self.data.control_traj.timestamps.extend(t.tolist())

    # Writes recorded data to disk as NumPy arrays
    # @input cfg_path [str]: path to configuration file
    # @input prefix [str]: data directory prefix (e.g. "run" -> run001, run002, etc.)
    def write_data(self, cfg_path: str, prefix: str = "run"):
        # Get the current run number and make the corresponding directory
        idx = 1
        write_dir = os.path.join(DATA_PATH, prefix+f"{idx:03}")
        while os.path.exists(write_dir):
            idx += 1
            write_dir = os.path.join(DATA_PATH, prefix+f"{idx:03}")
        os.makedirs(write_dir)

        # Copy the configuration info
        shutil.copy(cfg_path, os.path.join(write_dir, "cfg.yaml"))

        # Write data as NumPy arrays
        states, state_times = self.data.state_traj.as_np()
        controls, control_times = self.data.state_traj.as_np()
        np.save(os.path.join(write_dir, "states.npy"), states)
        np.save(os.path.join(write_dir, "state_times.npy"), state_times)
        np.save(os.path.join(write_dir, "controls.npy"), controls)
        np.save(os.path.join(write_dir, "control_times.npy"), control_times)

        print(f"[Recorder] Data written to {write_dir}!")

    # Loads trajectory data from a directory
    # @input dir [str]: directory to load from (EXCLUDING "data/")
    def from_data(self, dir: str):
        load_dir = os.path.join(DATA_PATH, dir)

        # Load data from NumPy arrays
        states = np.load(os.path.join(load_dir, "states.npy"))
        state_times = np.load(os.path.join(load_dir, "state_times.npy"))
        controls = np.load(os.path.join(load_dir, "controls.npy"))
        control_times = np.load(os.path.join(load_dir, "control_times.npy"))

        states = [torch.from_numpy(state) for state in states]
        state_times = state_times.tolist()
        controls = [torch.from_numpy(control) for control in controls]
        control_times = control_times.tolist()

        self.data = SystemTrajectory(Trajectory(states, state_times), Trajectory(controls, control_times))

    # Plots the state trajectory
    def plot_state_traj(self):
        states, times = self.data.state_traj.as_np()
        layout = Cartpole.get_state_plot_layout()
        description = Cartpole.get_state_description()

        fig, ax = plt.subplots(nrows=layout.shape[0], ncols=layout.shape[1], sharex=True, squeeze=False)
        fig.tight_layout()
        fig.supxlabel("time")
        fig.suptitle("cartpole state trajectory")
        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                if layout[i,j] >= 0:
                    col.plot(times, states[:,layout[i,j]])
                    col.set_ylabel(description[layout[i,j]].name)

        plt.show()

    # Plots the control trajectory
    def plot_control_traj(self):
        controls, times = self.data.control_traj.as_np()
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
    # @input n_frames [Optional[int]]: number of frames to animate
    # @input fps [int]: frames per second
    # @input end_wait [float]: seconds to wait at the end before finishing the animation
    # @input write [Optional[str]]: filename to write the animation to (EXCLUDING "media/""); None indicates not to write
    def animate2d(self, cartpole: Cartpole, n_frames: Optional[int] = None, fps: int = 5, end_wait: float = 1.0, write: Optional[str] = None):
        fig = plt.figure()
        ax = plt.axes(aspect="equal")

        states, times = self.data.state_traj.as_np()

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