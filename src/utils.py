'''
Handy utility functions
Author: rzfeng
'''
import torch


# Wraps angles (in radians) to (-pi, pi]
# @input theta [torch.tensor (B)]: angles in radians
# @output [torch.tensor (B)]: wrapped angles
def wrap_radians(theta: torch.tensor) -> torch.tensor:
    wrapped = torch.remainder(theta, 2*torch.pi)
    wrapped[wrapped > torch.pi] -= 2*torch.pi
    return wrapped


# Wraps a one-step dynamics model into a recursive dynamics function
class RecursiveDynamics:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    # Rolls out an action trajectory
    # @input times [torch.tensor (B)]: times
    # @input states [torch.tensor (B x state_dim)]: initial states
    # @input actions [torch.tensor (B x T x control_dim)]: control trajectory
    # output [torch.tensor (B x T x state_dim)]: next states
    def __call__(self, times: torch.tensor, states: torch.tensor, actions: torch.tensor) -> torch.tensor:
        T = actions.size(1)
        curr_state = states.clone().detach()
        next_states = []
        for t in range(T):
            pred_next_state = self.model(curr_state, actions[:,t,:])
            next_states.append(pred_next_state)
            curr_state = pred_next_state
        next_states = torch.stack(next_states, dim=1).detach()
        return next_states