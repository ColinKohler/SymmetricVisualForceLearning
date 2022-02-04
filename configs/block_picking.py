import os
import datetime

from config import Config

class BlockPickingConfig(Config):
  '''
  Task config for block picking.

  Args:
    num_gpus (int):
    results_path (str):
  '''
  def __init__(self, num_gpus=1, results_path=None):
    super().__init__(num_gpus=num_gpus)
    self.seed = 0

    # Env
    self.env_type = 'block_picking'
    self.max_steps = 10

    # Data Gen
    self.num_agent_workers = None
    self.discount = 0.95

    # Exploration

    # Training
    if results_path:
      self.results_path = os.path.join(self.results_path,
                                       'block_stacking',
                                       results_path)
    else:
      self.results_path = os.path.join(self.results_path,
                                       'block_stacking',
                                       datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))
    self.save_model = True
    self.training_steps = 1000
    self.batch_size = 64
    self.checkpoint_interval = 100

    # LR schedule
    self.lr_init = 1e-3
    self.weight_decay = 1e-5
    self.lr_decay = 0.95
    self.decay_lr_interval = 50

    # Replay Buffer
    self.replay_buffer_size = 100
    self.per_alpha = 0.6
    self.init_per_beta = 0.
    self.end_per_beta = 1.0
    self.per_eps = 1e-6

    # Wait times
    self.data_gen_delay = 0
    self.training_delay = 0
    self.train_data_ratio = 0

  def getEnvConfig(self):
    '''
    Gets the environment config required by the simulator for this task.

    Returns:
      dict: The env config
    '''
    return {
      'workspace' : self.workspace,
      'max_steps' : self.max_steps,
      'obs_size' : self.obs_size,
      'in_hand_size' : self.hand_obs_size,
      'physics_mode' : 'slow',
      'robot' : self.robot,
      'num_objects' : 1,
      'object_scale_range' : (0.8, 0.8)
    }
