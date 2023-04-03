'''
Simulated cartpole system
Author: rzfeng
'''
from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple, Dict

import torch
import numpy as np
from torchdiffeq import odeint

from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.patches import Rectangle

from src.utils import wrap_radians
from src.integration import ExplicitEulerIntegrator


# Class for describing a state or control variable
@dataclass
class VarDescription:
    name: str
    var_type: str # real or circle
    description: str
    unit: str


class Cartpole:
    # @input m_cart [float]: cart mass
    # @input m_pole [float]: pole mass
    # @input l [float]: pole length (to point mass)
    # @input vis_params [Dict]: visualization parameters ("cart_width", "cart_height", "pole_thickness", "cart_color", "pole_color")
    # @input g [float]: acceleration from gravity
    def __init__(self, s0: torch.tensor, m_cart: float, m_pole: float, l: float, vis_params: Dict, g: float = 9.81):
        assert(s0.dim() == 1)
        assert(s0.size(0) == self.state_dim())
        self._state = s0
        self.m_cart = m_cart
        self.m_pole = m_pole
        self.m_total = m_cart + m_pole
        self.l = l
        self.ml_pole = m_pole * l
        self.g = g
        self.vis_params = vis_params

    @classmethod
    def get_state_description(cls) -> List[VarDescription]:
        return [
            VarDescription("x", "real", "cart position", "m"),
            VarDescription("dx", "real", "cart velocity", "m/s"),
            VarDescription("theta", "circle", "pole angle", "rad"),
            VarDescription("dtheta", "real", "pole angular velocity", "rad/s")
        ]

    @classmethod
    def get_control_description(cls) -> List[VarDescription]:
        return [
            VarDescription("F", "real", "linear force", "N")
        ]

    # Returns the layout for state trajectory plotting
    # @output [np.ndarray (AxB)]: layout for state trajectory plotting, where AxB >= state_dim and
    #                             each element is the index to a state variable
    def get_state_plot_layout(cls) -> np.ndarray:
        return np.array([
            [0,2],
            [1,3]
        ])

    # Returns the layout for control trajectory plotting
    # @output [np.ndarray (AxB)]: layout for control trajectory plotting, where AxB >= control_dim and
    #                             each element is the index to a control variable
    def get_control_plot_layout(cls) -> np.ndarray:
        return np.array([[0]])

    @classmethod
    def state_dim(cls) -> int:
        return len(cls.get_state_description())

    @classmethod
    def control_dim(cls) -> int:
        return len(cls.get_control_description())

    def set_state(self, s: torch.tensor):
        self._state = s
        # Wrap any angles
        state_description = self.get_state_description()
        for i in range(self.state_dim()):
            if state_description[i].var_type == "circle":
                self._state[i] = wrap_radians(torch.tensor([self._state[i]])).squeeze()

    def get_state(self):
        return self._state.clone()

    # Computes the state derivatives for a batch of states and controls
    # @input t [torch.tensor (B)]: time points
    # @input s [torch.tensor (B x state_dim)]: batch of states
    # @input u [torch.tensor (B x control_dim)]: batch of controls
    # @output [torch.tensor (B x state_dim)]: batch of state derivatives
    def continuous_dynamics(self, t: torch.tensor, s: torch.tensor, u: torch.tensor) -> torch.tensor:
        x = s[:,0]
        dx = s[:,1]
        theta = s[:,2]
        dtheta = s[:,3]
        sintheta = torch.sin(theta)
        costheta = torch.cos(theta)

        temp = (u + self.ml_pole * dtheta.square() * sintheta) / self.m_total

        ds = torch.zeros_like(s)
        ds[:,0] = dx
        ds[:,2] = dtheta
        ds[:,3] = (self.g * sintheta - costheta * temp) / \
                  (self.l * (4.0/3.0 - (self.m_pole * costheta.square()) / self.m_total))
        ds[:,1] = temp - (self.ml_pole * ds[:,3] * costheta) / self.m_total
        return ds

    # Generates a discrete dynamics rollout function
    # @input dt [float]: time step length
    # @output [function(torch.tensor (B), torch.tensor (B x state_dim), torch.tensor (B x T x control_dim)) -> torch.tensor (B x T x state_dim)]:
    #       dynamics rollout function
    def generate_discrete_dynamics(self, dt: float) -> Callable[[torch.tensor, torch.tensor, torch.tensor], torch.tensor]:
        # Discrete dynamics rollout function, rolling out a batch of initial states and times using a batch of control trajectories
        # @input t [torch.tensor (B)]: batch of initial times
        # @input s0 [torch.tensor (B x state_dim)]: batch of initial states
        # @input u [torch.tensor (B x T x control_dim)]: batch of control trajectories
        # @output [torch.tensor (B x T x state_dim)]: batch of state trajectories
        def discrete_dynamics(t: torch.tensor, s0: torch.tensor, u: torch.tensor) -> torch.tensor:
            B = t.size(0)
            T = u.size(1)
            state_dim = s0.size(1)

            state_traj = torch.zeros((B, T, state_dim))
            integrator = ExplicitEulerIntegrator(dt, self.continuous_dynamics)
            curr_state = s0.clone()
            for t_idx in range(T):
                curr_state = integrator(t+t_idx*dt, curr_state.unsqueeze(1), u[:,t_idx,:].unsqueeze(1)).squeeze(1)
                state_traj[:,t_idx,:] = curr_state
            return state_traj
        return discrete_dynamics

    # Forward-integrates the dynamics of the system given a start state and a control trajectory
    # @input s0 [torch.tensor (state_dim)]: initial state
    # @input u [torch.tensor (T x control_dim)]: linear spline control trajectory
    # @input t_span [Tuple(float, float)]: time span
    # @input t_eval Optional[torch.tensor (N)]: evaluation time points
    # @output [torch.tensor (N x state_dim)]: state trajectory
    def integrate(self, s0: torch.tensor, u: torch.tensor, t_span: Tuple[float, float], t_eval: Optional[torch.tensor] = None) -> torch.tensor:
        t_spaced = torch.linspace(t_span[0], t_span[1], u.size(0))
        t_eval = t_spaced if t_eval is None else t_eval

        # Wraps the continuous dynamics into the form required by the ODE solver
        # @input t [torch.tensor ()]: time point
        # @input s [torch.tensor (state_dim)]: state
        # @output [torch.tensor (state_dim)]: state derivative
        def ode_dynamics(t: torch.tensor, s: torch.tensor):
            u_t = torch.tensor([np.interp(t, t_spaced, u[:,i].numpy()) for i in range(self.control_dim())])
            ds_t = self.continuous_dynamics(t.unsqueeze(0), s.unsqueeze(0), u_t.unsqueeze(0))
            return ds_t.squeeze(0)

        sol = odeint(ode_dynamics, s0, t_eval, method="rk4") # Dopri5 was giving some issues with step size
        return sol, t_eval

    # Applies a control sequence to the system
    # @input u [torch.tensor (T x control_dim)]: linear spline control trajectory
    # @input t_span [Tuple(float, float)]: time duration
    # @input t_eval Optional[torch.tensor (N)]: evaluation time points
    # @output [torch.tensor (N x state_dim)]: state trajectory
    # @output [torch.tensor(N)]: timestamps of state trajectory
    def apply_control(self, u: torch.tensor, t_span: Tuple[float, float], t_eval: Optional[torch.tensor] = None) -> Tuple[torch.tensor, torch.tensor]:
        state_traj, timestamps = self.integrate(self._state, u, t_span, t_eval)
        # Wrap any angles
        state_description = self.get_state_description()
        for i in range(self.state_dim()):
            if state_description[i].var_type == "circle":
                state_traj[:,i] = wrap_radians(state_traj[:,i])
        self.set_state(state_traj[-1,:])
        return state_traj, timestamps

    # Adds the 2D visualization to Matplotlib axes
    # @input ax [Axes]: axes to visualize on
    # @input s [torch.tensor (state_dim)]: state to visualize
    # @output [List[Artist])]: Matplotlib artists for the visualization
    def add_vis2d(self, ax: Axes, s: torch.tensor) -> List[Artist]:
        artists = []

        # Draw ground
        artists.append(ax.axhline(color="k"))

        # Draw cart
        x = s[0]
        anchor = np.array([x - 0.5*self.vis_params["cart_width"], 0.0])
        cart = Rectangle(anchor, self.vis_params["cart_width"], self.vis_params["cart_height"], color=self.vis_params["cart_color"])
        artists.append(ax.add_patch(cart))

        # Draw pole
        theta = s[2]
        pole_start = np.array([x, self.vis_params["cart_height"]])
        pole_end = pole_start + np.array([np.sin(theta) * 2.0 * self.l, np.cos(theta) * 2.0 * self.l])
        xs = np.array([pole_start[0], pole_end[0]])
        ys = np.array([pole_start[1], pole_end[1]])
        artists.append(ax.plot(xs, ys, linewidth=self.vis_params["pole_thickness"], color=self.vis_params["pole_color"])[0])

        return artists