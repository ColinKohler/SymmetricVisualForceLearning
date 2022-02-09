import torch
import numpy as np
import numpy.random as npr

from models.sac import Critic, GaussianPolicy

class SACAgent(object):
  '''
  Soft Actor-Critic Agent.

  Args:
    config (dict): Task config.
    device (torch.Device): Device to use for inference (cpu or gpu)
    training (bool): Flag indicating if the agent is being trained or not. Defaults to False.
  '''
  def __init__(self, config, device):
    self.config = config
    self.device = device

    self.actor = GaussianPolicy()
    self.actor.to(self.device)
    self.actor.test()

    self.critic = Critic()
    self.critic.to(self.device)
    self.critic.test()

  def selectAction(self, obs, evaluate=False):
    '''
    Get the action from the policy.

    Args:
      obs (numpy.array): The current observation

    Returns:
      numpy.array : The action generated by the policy
    '''
    obs = obs.to(self.device)

    with torch.no_grad():
      if evaluate:
        _, _, action = self.actor.sample(obs)
      else:
        action, _, _ = self.actor.sample(obs)

    return action

  def loadModel(self, checkpoint):
    '''
    Load the model from the given checkpoint.

    Args:
      checkpoint (dict): Checkpoint dictionary containing weights
    '''
    self.actor.set_weights(copy.deepcopy(checkpoint['weights'][0]))
    self.critic.set_weights(copy.deepcopy(checkpoint['weights'][1]))
