import ray
import torch
import numpy as np
import numpy.random as npr

@ray.remote
class DataGenerator(object):
  '''
  Ray worker that generates data samples.

  Args:
    initial_checkpoint (dict): Checkpoint to initalize training with.
    config (dict): Task config.
    seed (int): Random seed to use for random number generation
  '''
  def __init__(self, initial_checkpoint, config, seed):
    self.seed = seed
    self.config = config
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    self.env = None
    self.agent = None

    npr.seed(self.seed)
    torch.manual_seed(self.seed)

  def continuousDataGen(self, shared_storage, replay_buffer, test_mode=False):
    '''
    Continuously generates data samples according to the policy specified in the config.

    Args:
      replay_buffer (ray.worker): Replay buffer worker containing data samples.
      shared_storage (ray.worker): Shared storage worker, shares data across workers.
      test_mode (bool): Flag indicating if we are using this worker for data generation or testing.
        Defaults to data generation (False).
    '''
    while (
      ray.get(shared_storage.getInfo.remote('training_step')) < self.config.training_steps and \
      not ray.get(shared_storage.getInfo.remote('terminate'))
    ):
      pass

  def generateEpisode(self, test_mode):
    '''
    Generate a single episode.

    Args:
      test_mode (bool): Flag indicating if we are using this worker for data generation or testing.

    Returns:
      None : Episode history
    '''
    pass
