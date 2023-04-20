'''
Neural ODE for cartpole dynamics
Author: rzfeng
'''
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchdiffeq import odeint_adjoint

from src.utils import wrap_radians


class ODEFunc(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, n_layers: int, width: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        layers.append(nn.Linear(state_dim+action_dim, width))
        layers.append(nn.Tanh())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, state_dim+action_dim))
        self.layers = nn.Sequential(*layers)
    
    # Predicts the state derivative
    # @input t [torch.tensor (B)]: time
    # @input state_action [torch.tensor (B x state_dim + action_dim)]: concatenated state-action
    # @output [troch.tensor (b x state_dim + action_dim)]: predicted state derivative + filler
    def forward(self, t: torch.tensor, state_action: torch.tensor):
        return self.layers(state_action)


class DynamicsNeuralODE(pl.LightningModule):
    def __init__(self, state_dim: int, action_dim: int, dt: float, n_layers: int, width: int,
                 method: str = "dopri5", rtol: float = 1e-7, atol: float = 1e-9, options: Dict = {}, device: str = "cpu"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dt = dt
        self.tspan = torch.tensor([0.0, self.dt]).to(device)
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.options = options

        self.ode = ODEFunc(state_dim, action_dim, n_layers, width)

    # Predicts the next state
    # @input state [torch.tensor (B x state_dim)]: batch of initial states
    # @input action [torch.tensor (B x action_dim)]: batch of actions
    # @output [torch.tensor (B x state_dim)]: batch of predicted next states
    def forward(self, state: torch.tensor, action: torch.tensor):
        out = odeint_adjoint(self.ode, torch.cat((state, action), dim=-1), self.tspan,
                             method=self.method, rtol=self.rtol, atol=self.atol, options=self.options)[1,:,:self.state_dim]
        out_wrapped = out.clone()
        out_wrapped[:,2] = wrap_radians(out[:,2])
        return out_wrapped

    # Computes a multi-step recursive MSE loss
    # @input state [torch.tensor (B x state_dim)]: batch of initial states
    # @input action [torch.tensor (B x T x action_dim)]: batch of actions
    # @input state [torch.tensor (B x T x state_dim)]: batch of next states
    # @output [float]: loss
    def multistep_mse(self, state: torch.tensor, actions: torch.tensor, next_states: torch.tensor):
        T = actions.size(1)
        loss = 0.0
        discount_factor = 1.0
        curr_state = state.clone().detach()
        for t in range(T):
            pred_next_state = self(curr_state, actions[:,t,:])
            e = pred_next_state - next_states[:,t,:]
            e_wrapped = e.clone()
            e_wrapped[:,2] = wrap_radians(e[:,2])
            loss += discount_factor * e_wrapped.square().mean()
            # loss += discount_factor * F.mse_loss(pred_next_state, next_states[:,t,:])
            discount_factor *= 0.9
            curr_state = pred_next_state
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self.multistep_mse(*train_batch)
        self.log("training loss", loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.multistep_mse(*val_batch)
        self.log("validation loss", loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss = self.multistep_mse(*test_batch)
        self.log("test loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)