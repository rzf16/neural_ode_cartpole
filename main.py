'''
Main procedure for Robot Learning final project
Author: rzfeng
'''
import torch

import seaborn
seaborn.set()

from src.cartpole import Cartpole
from src.recorder import DataRecorder


def main():
    s0 = torch.tensor([0.0, 0.0, torch.pi/2.0, 0.0])
    cartpole = Cartpole(s0, 1.0, 0.1, 0.5,
                        {"cart_width": 1.0, "cart_height": 0.5, "pole_thickness": 2, "cart_color": "blue", "pole_color": "red"})
    recorder = DataRecorder()

    tf = 5.0
    u = torch.zeros((50, 1))
    states, times = cartpole.apply_control(u, (0.0, tf))
    recorder.log_control(u, torch.linspace(0.0, tf, u.size(0)))
    recorder.log_state(states, times)

    recorder.animate2d(cartpole, fps=20)


if __name__ == "__main__":
    main()