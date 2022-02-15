import os
import datetime

from configs.config import Config

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
    self.env_type = 'close_loop_block_picking'
    self.max_steps = 200

    # Data Gen
    self.num_data_gen_workers = 4
    self.num_expert_episodes = 100
    self.discount = 0.95

    # Exploration
    self.init_eps = 1.0
    self.end_eps = 0.0
    self.eps_anneal_steps = 1000

    # Training
    if results_path:
      self.results_path = os.path.join(self.root_path,
                                       'block_picking',
                                       results_path)
    else:
      self.results_path = os.path.join(self.root_path,
                                       'block_picking',
                                       datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))
    self.save_model = True
    self.training_steps = 10000
    self.batch_size = 64
    self.target_update_interval = 100
    self.checkpoint_interval = 100
    self.init_temp = 1e-2
    self.tau = 1e-2

    # LR schedule
    self.actor_lr_init = 1e-3
    self.actor_weight_decay = 1e-5
    self.critic_lr_init = 1e-3
    self.critic_weight_decay = 1e-5
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

  def getEnvConfig(self, render=False):
    '''
    Gets the environment config required by the simulator for this task.

    Args:
      render (bool): Render the PyBullet env. Defaults to False

    Returns:
      dict: The env config
    '''
    return {
      'workspace' : self.workspace,
      'max_steps' : self.max_steps,
      'obs_size' : self.obs_size,
      'physics_mode' : 'slow',
      'action_sequence' : self.action_sequence,
      'robot' : self.robot,
      'num_objects' : 1,
      'object_scale_range' : (1.0, 1.0),
      'random_orientation' : self.random_orientation,
      'workspace_check' : 'point',
      'reward_type' : self.reward_type,
      'view_type' : self.view_type,
      'obs_type' : self.obs_type,
      'render': render
    }

  def getPlannerConfig(self):
    '''

    '''
    return {
      'random_orientation': True,
      'dpos' : self.dpos,
      'drot' : self.drot
    }
