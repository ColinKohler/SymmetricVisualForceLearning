import copy
import os
import ray
import torch
import pickle

@ray.remote
class SharedStorage(object):
  '''
  Remote ray worker class used to share data betweeen ray workers.

  Args:
    checkpoint (dict): Training checkpoint.
    config (dict): Task configuration.
  '''
  def __init__(self, checkpoint, config):
    self.config = config
    self.current_checkptoint = copy.deepcopy(checkpoint)

  def saveCheckpoint(self, path=None):
    '''
    Save the checkpoint to file.

    Args:
      path (str): The path to save the checkpoint to. Defaults to None.
        When set to None, defaults to results path in config.
    '''
    if not path:
      path = os.path.join(self.config.results_path, 'model.checkpoint')
    torch.save(self.current_checkpoint, path)

  def getCheckpoint(self):
    '''
    Get the current checkpoint.

    Returns:
      dict: Current checkpoint.
    '''
    return copy.deepcopy(self.current_checkpoint)

  def saveReplayBuffer(self, replay_buffer):
    '''
    Save the replay buffer to file.

    Args:
     replay_buffer (list): The replay buffer data.
    '''
    pickle.dump(
      {
        'buffer' : replay_buffer,
        'num_eps' : self.current_checkpoint['num_eps'],
        'num_steps' : self.current_checkpoint['num_steps']
      },
      open(os.path.join(self.config.results_path, 'replay_buffer.pkl'), 'wb')
    )

  def logEpsRewared(self, reward):
    '''
    Add the episode reward to the checkpoint for logging.

    Args:
      reward (float): The episode reward.
    '''
    self.current_checkpoint['eps_reward'].append(reward)

  def getInfo(self, keys):
    '''
    Get data from the current checkpoint for the desired keys.

    Args:
      keys (list[str]): Keys to get data from.

    Returns:
      dict: The key-value pairs desired.
    '''
    return {key: self.current_checkpoint[key] for key in keys}

  def setInfo(self, keys, values):
    '''
    Update the current checkpoint to the new key-value pairs.

    Args:
      keys (list[str]): Keys to update.
      values (list[str]): Values to update.
    '''
    [self.current_checkpoint[key] = value for key, value in zip(keys, values)]
