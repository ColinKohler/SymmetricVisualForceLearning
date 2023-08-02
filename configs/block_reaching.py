import os
import datetime
import numpy as np

from configs.config import Config

class BlockReachingConfig(Config):
  '''
  Task config for block reaching.

  Args:
    num_gpus (int):
    results_path (str):
  '''
  def __init__(self, equivariant=True, vision_size=64, encoder='vision+force+proprio', num_gpus=1, results_path=None):
    super().__init__(equivariant=equivariant, vision_size=vision_size, encoder=encoder, num_gpus=num_gpus)
    self.seed = None

    # Env
    self.max_steps = 50
    self.max_force = 100

    # Training
    if results_path:
      self.results_path = os.path.join(self.root_path,
                                       'block_reaching',
                                       results_path)
    else:
      self.results_path = os.path.join(self.root_path,
                                       'block_reaching',
                                       datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))
    self.save_model = True
    self.pre_training_steps = 0
    self.training_steps = 10000
    self.training_steps_per_action = 1
    self.batch_size = 64
    self.target_update_interval = 1
    self.checkpoint_interval = 100
    self.init_temp = 1e-2
    self.tau = 1e-2
    self.discount = 0.99
    self.init_expert_weight = 0.0
    self.end_expert_weight = 0.0

    # LR schedule
    self.actor_lr_init = 1e-3
    self.critic_lr_init = 1e-3
    self.lr_decay = 0.95
    self.lr_decay_interval = 500

    # Replay Buffer
    self.replay_buffer_size = 100000
    self.per_alpha = 0.6
    self.init_per_beta = 0.4
    self.end_per_beta = 1.0
    self.per_eps = 1e-6
