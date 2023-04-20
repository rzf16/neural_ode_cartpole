'''
Basic feedforward NN for cartpole dynamics
Author: rzfeng
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.utils import wrap_radians


class ResidualDynamicsANN(pl.LightningModule):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        layers.append(nn.Linear(state_dim+action_dim, 100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100, 100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100, 100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100, state_dim))
        self.layers = nn.Sequential(*layers)

    # Predicts the next state
    # @input state [torch.tensor (B x state_dim)]: batch of initial states
    # @input action [torch.tensor (B x action_dim)]: batch of actions
    # @output [torch.tensor (B x state_dim)]: batch of predicted next states
    def forward(self, state: torch.tensor, action: torch.tensor):
        # return state + self.layers(torch.cat((state, action), dim=-1))
        out = state + self.layers(torch.cat((state, action), dim=-1))
        out[:,2] = wrap_radians(out[:,2])
        return out

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
            pred_next_state[:,2] = wrap_radians(pred_next_state[:,2])
            e = pred_next_state - next_states[:,t,:]
            e[:,2] = wrap_radians(e[:,2])
            loss += discount_factor * e.square().mean()
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