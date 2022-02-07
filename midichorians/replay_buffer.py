import ray
import torch
import numpy as np
import numpy.random as npr

@ray.remote
class ReplayBuffer(object):
  '''

  '''
  def __init__(self, inital_checkpoint, initial_buffer, config):
    self.config = config
    npr.seed(self.config.seed)

    self.buffer = copy.deepcopy(initial_buffer)
    self.num_eps = initial_checkpoint['num_eps']
    self.num_steps = initial_checkpoint['num_steps']
    self.total_samples = None

    self.model = None

  def getBuffer(self):
    '''
    Get the replay buffer.

    Returns:
      list[EpisodeHistory] : The replay buffer
    '''
    return self.buffer

  def add(self, eps_history, shared_storage=None):
    '''
    Add a new episode to the replay buffer.

    Args:
      eps_history (EpisodeHistory): The episode to add to the buffer.
      shared_storage (ray.Worker): Shared storage worker. Defaults to None.
    '''
    pass

  def sample(self, shared_storage):
    '''
    Sample a batch from the replay buffer.

    Args:
      shared_storage (ray.Worker): Shared storage worker.

    Returns:
      (list[int], list[numpy.array], list[numpy.array], list[double], list[double]) : (Index, Observation, Action, Reward, Weight)
    '''
    pass

  def updatePriorities(self, td_errors, idx_info):
    '''
    Update the priorities for each sample in the batch.

    Args:
      td_errors (numpy.array): The TD error for each sample in the batch
      idx_info (numpy.array): The episode and step for each sample in the batch
    '''
    pass

  def resetPriorities(self):
    '''
    Uniformly reset the priorities for all samples in the buffer.
    '''
    pass

  def updateTargetNetwork(self, shared_storage):
    '''
    Update the weights of the model used to generate the targets.

    Args:
      shared_storage (ray.Worker): Shared storage worker.
    '''
    pass
