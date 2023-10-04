import torch
import numpy as np
import numpy.random as npr
from functools import partial

from svfl import torch_utils
from svfl.models.sac import Critic, GaussianPolicy

class Agent(object):
  '''
  Soft Actor-Critic Agent.

  Args:
    config (dict): Task config.
    device (torch.Device): Device to use for inference (cpu or gpu)
  '''
  def __init__(self, config, device, actor=None, critic=None, initialize_models=True):
    self.config = config
    self.device = device

    self.normalizeForce = partial(torch_utils.normalizeForce, max_force=self.config.max_force)

    self.p_range = torch.tensor([0, 1])
    self.dx_range = torch.tensor([-self.config.dpos, self.config.dpos])
    self.dy_range = torch.tensor([-self.config.dpos, self.config.dpos])
    self.dz_range = torch.tensor([-self.config.dpos, self.config.dpos])
    self.dtheta_range = torch.tensor([-self.config.drot, self.config.drot])
    self.action_shape = 5

    if actor:
      self.actor = actor
    else:
      self.actor = GaussianPolicy(
        self.config.vision_size,
        self.config.action_dim,
        z_dim=self.config.z_dim,
        encoder=self.config.encoder,
        initialize=initialize_models,
        equivariant=self.config.equivariant
      )
      self.actor.to(self.device)
      self.actor.eval()

    if critic:
      self.critic = critic
    else:
      self.critic = Critic(
        self.config.vision_size,
        self.config.action_dim,
        z_dim=self.config.z_dim,
        encoder=self.config.encoder,
        initialize=initialize_models,
        equivariant=self.config.equivariant
      )
      self.critic.to(self.device)
      self.critic.eval()

  def getAction(self, pose, force, proprio, evaluate=False):
    '''
    Get the action from the policy.

    Args:
      evalute (bool):

    Returns:
      (numpy.array, double) : (Action, Q-Value)
    '''
    #vision = torch.Tensor(vision.astype(np.float32)).view(vision.shape[0], vision.shape[1], vision.shape[2], vision.shape[3]).to(self.device)
    #vision = torch_utils.centerCrop(vision, out=self.config.vision_size)
    pose = torch.Tensor(torch_utils.normalizePose(pose, self.config.workspace)).view(pose.shape[0], self.config.pose_dim).to(self.device)
    force = torch.Tensor(torch_utils.normalizeForce(force, self.config.max_force)).view(pose.shape[0], self.config.force_history, self.config.force_dim).to(self.device)
    proprio = torch.Tensor(torch_utils.normalizeProprio(proprio, self.config.workspace)).view(pose.shape[0], self.config.proprio_dim).to(self.device)

    with torch.no_grad():
      if evaluate:
        _, _, action = self.actor.sample((pose, force, proprio))
      else:
        action, _, _ = self.actor.sample((pose, force, proprio))

    action = action.cpu()
    norm_action, action = self.unnormalizeActions(*[action[:,i] for i in range(self.action_shape)])
    with torch.no_grad():
      value = self.critic((pose, force, proprio), norm_action.to(self.device))

    value = torch.min(torch.hstack((value[0], value[1])), dim=1)[0]
    return norm_action, action, value

  def unnormalizeActions(self, norm_p, norm_dx, norm_dy, norm_dz, norm_dtheta):
    '''
    Convert action from model (normalized) to environment action (unnormalized).
    '''
    p = torch_utils.unnormalize(norm_p, *self.p_range)
    dx = torch_utils.unnormalize(norm_dx, *self.dx_range)
    dy = torch_utils.unnormalize(norm_dy, *self.dy_range)
    dz = torch_utils.unnormalize(norm_dz, *self.dz_range)
    dtheta = torch_utils.unnormalize(norm_dtheta, *self.dtheta_range)

    actions = torch.stack([p, dx, dy, dz, dtheta], dim=1)
    norm_actions = torch.stack([norm_p, norm_dx, norm_dy, norm_dz, norm_dtheta], dim=1)

    return norm_actions, actions

  def convertPlanAction(self, plan_action):
    '''
    Convert actions from planner to agent actions by normalizing/unnormalizing them.
    '''
    p = plan_action[:, 0].clamp(*self.p_range)
    dx = plan_action[:, 1].clamp(*self.dx_range)
    dy = plan_action[:, 2].clamp(*self.dy_range)
    dz = plan_action[:, 3].clamp(*self.dz_range)
    dtheta = plan_action[:, 4].clamp(*self.dtheta_range)

    return self.unnormalizeActions(
      torch_utils.normalize(p, *self.p_range),
      torch_utils.normalize(dx, *self.dx_range),
      torch_utils.normalize(dy, *self.dy_range),
      torch_utils.normalize(dz, *self.dz_range),
      torch_utils.normalize(dtheta, *self.dtheta_range)
    )

  def getWeights(self):
    '''
    '''
    return (self.actor.state_dict(),
            self.critic.state_dict())

  def setWeights(self, weights):
    '''
    Load given weights into the actor and critic

    Args:
      weights (dict, dict): (actor weights, critic weights)
    '''
    if weights is not None:
      self.actor.eval()
      self.critic.eval()

      self.actor.load_state_dict(weights[0])
      self.critic.load_state_dict(weights[1])
