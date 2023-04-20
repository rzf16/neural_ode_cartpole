'''
Main procedure for Robot Learning final project
Author: rzfeng
'''
import time
import yaml

import torch

import seaborn
seaborn.set()

from src.cartpole import Cartpole
from src.recorder import DataRecorder
from src.mppi import MPPI
from src.ann import ResidualDynamicsANN
from src.neural_ode import DynamicsNeuralODE
from src.utils import RecursiveDynamics


CFG_PATH = "cfg.yaml"


def main():
    cfg = yaml.safe_load(open(CFG_PATH))
    cartpole = Cartpole(torch.tensor(cfg["task"]["s0"]), cfg["system"]["m_cart"], cfg["system"]["m_pole"], cfg["system"]["l"],
                        cfg["system"]["vis_params"])
    recorder = DataRecorder(["gt", "model"])

    if cfg["mppi"]["dynamics_type"] == "gt":
        dynamics = cartpole.generate_discrete_dynamics(cfg["dt"])
    elif cfg["mppi"]["dynamics_type"] == "residual_nn":
        model = ResidualDynamicsANN(4,1)
        model.load_state_dict(torch.load(cfg["mppi"]["dynamics_path"]))
        dynamics = RecursiveDynamics(model)
    elif cfg["mppi"]["dynamics_type"] == "neural_ode":
        model = DynamicsNeuralODE(4, 1, cfg["dt"],
                                  cfg["mppi"]["dynamics_params"]["n_layers"],
                                  cfg["mppi"]["dynamics_params"]["width"],
                                  method=cfg["mppi"]["dynamics_params"]["method"],
                                  rtol=cfg["mppi"]["dynamics_params"]["rtol"],
                                  atol=cfg["mppi"]["dynamics_params"]["atol"],
                                  options=cfg["mppi"]["dynamics_params"]["options"],)
        model.load_state_dict(torch.load(cfg["mppi"]["dynamics_path"]))
        dynamics = RecursiveDynamics(model)
    else:
        print("[Main] Error! Unrecognized dynamics type D:")
        return

    goal = torch.tensor(cfg["task"]["goal"])
    tolerance =  torch.tensor(cfg["task"]["tolerance"])
    cost = generate_quad_cost(goal, torch.diag(torch.tensor(cfg["mppi"]["Q"])))
    controller = MPPI(dynamics, cost, 4, 1, torch.tensor(cfg["mppi"]["sigma"]), cfg["dt"], cfg["mppi"]["horizon"],
                      u_min=torch.tensor(cfg["system"]["u_min"]), u_max=torch.tensor(cfg["system"]["u_max"]))
    controller.warm_start(cartpole.get_state(), cfg["mppi"]["n_warm_start_steps"])

    goal_reached = False
    mppi_times = []
    for i in range(cfg["mppi"]["max_steps"]):
        if i % cfg["vis"]["model_reset_freq"] == 0:
            pred_state = cartpole.get_state()

        tic = time.time()
        a = controller.get_command(cartpole.get_state())
        toc = time.time()
        mppi_times.append(toc - tic)
        cartpole.apply_control(a.unsqueeze(0).repeat(2,1), (0.1*i, 0.1*(i+1)))

        pred_state = model(pred_state.unsqueeze(0), a.unsqueeze(0)).squeeze().detach()

        recorder.log_state("gt", cartpole.get_state().unsqueeze(0), torch.tensor([0.1*(i+1)]))
        recorder.log_state("model", pred_state.unsqueeze(0), torch.tensor([0.1*(i+1)]))
        recorder.log_control(a.unsqueeze(0), torch.tensor([0.1*i]))

        if cfg["mppi"]["stop_on_goal"]:
            if torch.all(torch.abs(cartpole.get_state() - goal) <= tolerance):
                dt = cfg["dt"]
                print(f"MPPI reached the goal after {i+1} steps ({dt*(i+1)} seconds) :D")
                goal_reached = True
                break

    if not goal_reached and cfg["mppi"]["stop_on_goal"]:
        dt = cfg["dt"]
        max_steps = cfg["mppi"]["max_steps"]
        print(f"MPPI failed to reach the goal after {max_steps} steps ({dt*max_steps} seconds) D:")

    print(f"Average MPPI compute time: {sum(mppi_times) / len(mppi_times)} s")

    if cfg["vis"]["plot_state_traj"]:
        recorder.plot_state_trajs()
    
    if cfg["vis"]["plot_control_traj"]:
        recorder.plot_control_traj()

    if cfg["vis"]["animate"]:
        recorder.animate2d(cartpole, "gt", fps=20)


# Generates an MPPI cost function for a goal-reaching objective
# @input goal [torch.tensor (state_dim)]: goal state
# @input Q [torch.tensor (state_dim x state_dim)]: state weight matrix
# output [function(torch.tensor (B), torch.tensor (B x T x state_dim), torch.tensor (B x T x control_dim)) -> torch.tensor (B)]:
#       MPPI cost function
def generate_quad_cost(goal: torch.tensor, Q: torch.tensor):
    # Goal-reaching cost function for MPPI, computing the cost for a batch of state and control trajectories
    # @input times [torch.tensor (B)]: batch of initial timesteps
    # @input states [torch.tensor (B x T x state_dim)]: batch of state trajectories
    # @input actions [torch.tensor (B x T x control_dim)]: batch of control trajectories
    # @output [torch.tensor (B)]: batch of costs
    def quad_cost(times, states, actions):
        B = states.size(0)
        T = states.size(1)
        state_dim = states.size(2)
        batch_Q = Q.repeat(B*T,1,1)
        diffs = goal - states

        cost = torch.bmm(diffs.reshape(B*T,1,state_dim), torch.bmm(batch_Q, diffs.reshape(B*T, state_dim, 1))).reshape(B,T).sum(dim=1)
        return cost
    return quad_cost


if __name__ == "__main__":
    main()