import torch
import numpy as np
import numpy.random as npr

class SACAgent(object):
  '''
  Soft Actor-Critic Agent.

  Args:
    config (dict): Task config.
    device (torch.Device): Device to use for inference (cpu or gpu)
    training (bool): Flag indicating if the agent is being trained or not. Defaults to False.
  '''
  def __init__(self, config, device, training=False):
    pass

  def selectAction(self, obs):
    '''
    Get the action from the policy.

    Args:
      obs (numpy.array): The current observation.

    Returns:
      numpy.array : The action generated by the policy.
    '''
    pass
