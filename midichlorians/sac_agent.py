import torch
import numpy as np
import numpy.random as npr

from midichlorians.models.sac import Critic, GaussianPolicy
from midichlorians.models.equivariant_sac import EquivariantCritic, EquivariantGaussianPolicy

class SACAgent(object):
  '''
  Soft Actor-Critic Agent.

  Args:
    config (dict): Task config.
    device (torch.Device): Device to use for inference (cpu or gpu)
    training (bool): Flag indicating if the agent is being trained or not. Defaults to False.
    dx (double):
    dy (double):
    dz (double):
    dr (double):
  '''
  def __init__(self, config, device):
    self.config = config
    self.device = device

    self.p_range = torch.tensor([0, 1])
    self.dx_range = torch.tensor([-self.config.dpos, self.config.dpos])
    self.dy_range = torch.tensor([-self.config.dpos, self.config.dpos])
    self.dz_range = torch.tensor([-self.config.dpos, self.config.dpos])
    self.dtheta_range = torch.tensor([-self.config.drot, self.config.drot])
    self.action_shape = 5

    self.actor = EquivariantGaussianPolicy(self.config.obs_channels, self.config.action_dim)
    self.actor.to(self.device)
    self.actor.eval()

    self.critic = EquivariantCritic(self.config.obs_channels, self.config.action_dim)
    self.critic.to(self.device)
    self.critic.eval()

  def getAction(self, obs, evaluate=False):
    '''
    Get the action from the policy.

    Args:
      obs (numpy.array): The current observation
      evalute (bool):

    Returns:
      (numpy.array, double) : (Action, Q-Value)
    '''
    obs = torch.Tensor(obs.astype(np.float32)).view(1, 1, 128, 128).to(self.device)

    with torch.no_grad():
      if evaluate:
        _, _, action = self.actor.sample(obs)
      else:
        action, _, _ = self.actor.sample(obs)
      value = self.critic(obs, action)

    return self.decodeAction(*[action[:,i] for i in range(self.action_shape)])[1], value

  def getRandomAction(self, obs):
    '''
    Get a random action from the environment.

    Args:
      obs (numpy.array): The current observation

    Returns:
      (numpy.array, double) : (Action, Q-Value)
    '''
    obs = torch.Tensor(obs.astype(np.float32)).view(1, 1, 128, 128).to(self.device)

    action = torch.rand(1, self.action_shape)
    with torch.no_grad():
      value = self.critic(obs, action.to(self.device))

    return self.decodeAction(*[action[:,i] for i in range(self.action_shape)])[1], value

  def decodeAction(self, unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta):
    '''
    Convert action from model to environment action.

    Args:
      unscaled_p (double):
      unscaled_dx (double):
      unscaled_dy (double):
      unscaled_dz (double):
      unscaled_dtheta (double):

    Returns:
      (torch.Tensor, torch.Tensor) : Unscaled actions, scaled actions
    '''
    p = 0.5 * (unscaled_p + 1) * (self.p_range[1] - self.p_range[0]) + self.p_range[0]
    dx = 0.5 * (unscaled_dx + 1) * (self.dx_range[1] - self.dx_range[0]) + self.dx_range[0]
    dy = 0.5 * (unscaled_dy + 1) * (self.dy_range[1] - self.dy_range[0]) + self.dy_range[0]
    dz = 0.5 * (unscaled_dz + 1) * (self.dz_range[1] - self.dz_range[0]) + self.dz_range[0]

    dtheta = 0.5 * (unscaled_dtheta + 1) * (self.dtheta_range[1] - self.dtheta_range[0]) + self.dtheta_range[0]
    actions = torch.stack([p, dx, dy, dz, dtheta], dim=1)
    unscaled_actions = torch.stack([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta], dim=1)

    return unscaled_actions, actions

  def convertPlanAction(self, plan_action):
    '''
    Convert actions from planner to agent actions by unscalling/scalling them.

    Args:
      plan_action (numpy.array): Action received from planner

    Returns:
      (torch.Tensor, torch.Tensor) : Unscaled actions, scaled actions
    '''
    p = plan_action[0].clamp(*self.p_range)
    dx = plan_action[1].clamp(*self.dx_range)
    dy = plan_action[2].clamp(*self.dy_range)
    dz = plan_action[3].clamp(*self.dz_range)
    dtheta = plan_action[4].clamp(*self.dtheta_range)

    return self.decodeAction(
      self.getUnscaledAction(p, self.p_range).view(1,1),
      self.getUnscaledAction(dx, self.dx_range).view(1,1),
      self.getUnscaledAction(dy, self.dy_range).view(1,1),
      self.getUnscaledAction(dz, self.dz_range).view(1,1),
      self.getUnscaledAction(dtheta, self.dtheta_range).view(1,1)
    )

  def getUnscaledAction(self, action, action_range):
    '''
    Convert action to the unscalled version using the given range.

    Args:
      action (double): Action
      action_range (list[double]): Min and max range for the given action

    Returns:
      double: The unscalled action
    '''
    return 2 * (action - action_range[0]) / (action_range[1] - action_range[0]) - 1

  def setWeights(self, weights):
    '''
    Load given weights into the actor and critic

    Args:
      weights (dict, dict): Model weights to load
    '''
    self.actor.load_state_dict(weights[0])
    self.critic.load_state_dict(weights[1])