'''
Multi-step dataset for dynamics data
Author: rzfeng
'''
from typing import List
import numpy as np
from torch.utils.data import Dataset


class MultistepDataset(Dataset):
    # @input data [List[Dict]]: list of dictionaries with tensors of states (T x state_dim), state_times (T),
    #                           actions (T-1 x action_dim), and action_times (T-1)
    # @input n_steps [int]: training sample horizon
    def __init__(self, data: List, n_steps: int):
        super().__init__()
        self.data = data
        self.traj_length = self.data[0]["controls"].size(0) - n_steps + 1
        self.n_steps = n_steps

    def __len__(self):
        return len(self.data) * (self.traj_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    # Returns a training sample
    # @input i [int]: index
    # @output [torch.tensor (state_dim)]: initial state
    # @output [torch.tensor (n_steps x action_dim)]: actions
    # @output [torch.tensor (n_steps x state_dim)]: next states
    def __getitem__(self, i):
        traj_idx = int(np.floor(i / self.traj_length))
        transition_idx = i % self.traj_length

        state = self.data[traj_idx]["states"][transition_idx,:]
        actions = self.data[traj_idx]["controls"][transition_idx : transition_idx+self.n_steps, :]
        next_states = self.data[traj_idx]["states"][transition_idx+1 : transition_idx+self.n_steps+1, :]
        return state, actions, next_states

class SinglestepDataset(Dataset):
    # @input data [List[Dict]]: list of dictionaries with tensors of states (T x state_dim), state_times (T),
    #                           actions (T-1 x action_dim), and action_times (T-1)
    # @input n_steps [int]: training sample horizon
    def __init__(self, data: List):
        super().__init__()
        self.data = data
        self.traj_length = self.data[0]["controls"].size(0)

    def __len__(self):
        return len(self.data) * (self.traj_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    # Returns a training sample
    # @input i [int]: index
    # @output [torch.tensor (state_dim)]: initial state
    # @output [torch.tensor (action_dim)]: action
    # @output [torch.tensor (state_dim)]: next state
    def __getitem__(self, i):
        traj_idx = int(np.floor(i / self.traj_length))
        transition_idx = i % self.traj_length

        state = self.data[traj_idx]["states"][transition_idx,:]
        action = self.data[traj_idx]["controls"][transition_idx,:]
        next_state = self.data[traj_idx]["states"][transition_idx+1,:]
        return state, action, next_state