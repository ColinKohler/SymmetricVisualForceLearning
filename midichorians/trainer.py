import time
import ray
import torch
import numpy as np
import numpy.random as npr

@ray.remote
class Trainer(object):
  '''
  Ray worker that cordinates training of our model.

  Args:
    initial_checkpoint (dict): Checkpoint to initalize training with.
    config (dict): Task config.
  '''
  def __init__(self, initial_checkpoint, config):
    self.config = config
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    self.model = None
    self.training_step = inital_checkpoint['training_step']

    npr.seed(self.config.seed)
    torch.manual_seed(self.config.seed)

  def continuousUpdateWeights(self, replay_buffer, shared_storage):
    '''
    Continuously sample batches from the replay buffer and perform weight updates.
    This continuous until the desired number of training steps has been reached.

    Args:
      replay_buffer (ray.worker): Replay buffer worker containing data samples.
      shared_storage (ray.worker): Shared storage worker, shares data across workers.
    '''
    while ray.get(shared_storage.getInfo.remote('num_steps')) == 0k:
      time.sleep(0.1)

    next_batch = replay_buffer.sample.remote(shared_storage)
    while self.training_step < self.config.training_steps and \
          not ray.get(shared_storage.getInfo.remote('terminate')):
      idx_batch, batch = ray.get(next_batch)
      next_batch = replay_buffer.sample.remote(shared_storage)

      # TODO: Training loop

      replay_buffer.updatePriorities.remote(priorities, idx_batch)
      self.training_step += 1

      if self.training_step % self.config.checkpoint_interval == 0:
        shared_storage.setInfo.remote(
          {
            'weights' : copy.deepcopy(weights),
            'optimizer_state' : copy.deepcopy(optimizer_state)
          }
        )
        replay_buffer.updateTargetNetwork.remote(shared_storage)
        if self.config.save_model:
          shared_storage.saveReplayBuffer.remote(replay_buffer.getBuffer.remote())
          shared_storage.saveCheckpoint.remote()

    shared_storage.setInfo.remote(
      {
        'training_step' : self.training_step,
        'lr' : self.lr,
        'loss' : loss
      }
    )

    if self.config.training_delay:
      time.sleep(self.config.training_delay)
