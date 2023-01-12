import torch
import numpy as np
import numpy.random as npr
from functools import partial

from midichlorians import torch_utils
from midichlorians.models.encoders.equiv_sensor_fusion import EquivariantSensorFusion
from midichlorians.models.equivariant_fusion_sac import EquivariantFusionCritic, EquivariantFusionGaussianPolicy

class SACAgent(object):
  '''
  Soft Actor-Critic Agent.

  Args:
    config (dict): Task config.
    device (torch.Device): Device to use for inference (cpu or gpu)
  '''
  def __init__(self, config, device, encoder_1=None, actor=None, encoder_2=None, critic=None, initialize_models=True):
    self.config = config
    self.device = device

    self.normalizeForce = partial(torch_utils.normalizeForce, max_force=self.config.max_force)

    self.p_range = torch.tensor([0, 1])
    self.dx_range = torch.tensor([-self.config.dpos, self.config.dpos])
    self.dy_range = torch.tensor([-self.config.dpos, self.config.dpos])
    self.dz_range = torch.tensor([-self.config.dpos, self.config.dpos])
    self.dtheta_range = torch.tensor([-self.config.drot, self.config.drot])
    self.action_shape = 5

    if encoder_1:
      self.encoder_1 = encoder_1
    else:
      self.encoder_1 = EquivariantSensorFusion(deterministic=self.config.deterministic)
      self.encoder_1.to(self.device)
      self.encoder_1.train()

    if actor:
      self.actor = actor
    else:
      self.actor = EquivariantFusionGaussianPolicy(self.config.action_dim, initialize=initialize_models)
      self.actor.to(self.device)
      self.actor.train()

    if encoder_2:
      self.encoder_2 = encoder_2
    else:
      self.encoder_2 = EquivariantSensorFusion(deterministic=self.config.deterministic)
      self.encoder_2.to(self.device)
      self.encoder_2.train()

    if critic:
      self.critic = critic
    else:
      self.critic = EquivariantFusionCritic(self.config.action_dim, initialize=initialize_models)
      self.critic.to(self.device)
      self.critic.train()

  def getAction(self, state, obs, force, proprio, evaluate=False):
    '''
    Get the action from the policy.

    Args:
      state (int): The current gripper state
      obs (numpy.array): The current observation
      evalute (bool):

    Returns:
      (numpy.array, double) : (Action, Q-Value)
    '''
    obs = torch.Tensor(obs.astype(np.float32)).view(len(state), 1, self.config.obs_size, self.config.obs_size).to(self.device)
    state = torch.Tensor(state).view(len(state), 1, 1, 1).to(self.device)
    state_tile = state.repeat(1, 1, obs.size(2), obs.size(3))
    obs = torch.cat((obs, state_tile), dim=1)
    force = torch.Tensor(torch_utils.normalizeForce(force, self.config.max_force)).view(len(state), self.config.force_history, self.config.force_dim).to(self.device)
    proprio = torch.Tensor(proprio).view(len(state), 4).to(self.device)

    with torch.no_grad():
      if self.config.deterministic:
        z = self.encoder_1((obs, force, proprio))
      else:
        z, z_mu, z_var, prior_mu, prior_var = self.encoder_1((obs, force, proprio))

      if evaluate:
        _, _, action = self.actor.sample(z)
      else:
        action, _, _ = self.actor.sample(z)

    action = action.cpu()
    action_idx, action = self.decodeActions(*[action[:,i] for i in range(self.action_shape)])
    with torch.no_grad():
      z = self.encoder_2((obs, force, proprio))
      value = self.critic(z, action_idx.to(self.device))

    value = torch.min(torch.hstack((value[0], value[1])), dim=1)[0]
    return action_idx, action, value

  def decodeActions(self, unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta):
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
    p = plan_action[:, 0].clamp(*self.p_range)
    dx = plan_action[:, 1].clamp(*self.dx_range)
    dy = plan_action[:, 2].clamp(*self.dy_range)
    dz = plan_action[:, 3].clamp(*self.dz_range)
    dtheta = plan_action[:, 4].clamp(*self.dtheta_range)

    return self.decodeActions(
      self.getUnscaledActions(p, self.p_range),
      self.getUnscaledActions(dx, self.dx_range),
      self.getUnscaledActions(dy, self.dy_range),
      self.getUnscaledActions(dz, self.dz_range),
      self.getUnscaledActions(dtheta, self.dtheta_range)
    )

  def getUnscaledActions(self, action, action_range):
    '''
    Convert action to the unscalled version using the given range.

    Args:
      action (double): Action
      action_range (list[double]): Min and max range for the given action

    Returns:
      double: The unscalled action
    '''
    return 2 * (action - action_range[0]) / (action_range[1] - action_range[0]) - 1

  def getWeights(self):
    '''
    '''
    return (self.encoder_1.state_dict(),
            self.actor.state_dict(),
            self.encoder_2.state_dict(),
            self.critic.state_dict())

  def setWeights(self, weights):
    '''
    Load given weights into the actor and critic

    Args:
      weights (dict, dict): (actor weights, critic weights)
    '''
    if weights is not None:
      self.encoder_1.load_state_dict(weights[0])
      self.actor.load_state_dict(weights[1])
      self.encoder_2.load_state_dict(weights[2])
      self.critic.load_state_dict(weights[3])
