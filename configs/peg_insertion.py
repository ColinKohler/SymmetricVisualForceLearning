import os
import datetime
import numpy as np

from configs.config import Config

class PegInsertionConfig(Config):
  '''
  Task config for peg insertion.

  Args:
    num_gpus (int):
    results_path (str):
  '''
  def __init__(self, equivariant=True, vision_size=64, encoder='vision+force+proprio', num_gpus=1, results_path=None):
    super().__init__(equivariant=equivariant, vision_size=vision_size, encoder=encoder, num_gpus=num_gpus)
    self.seed = None

    # Env
    self.robot = 'panda'
    self.env_type = 'close_loop_peg_insertion'
    self.max_steps = 50
    self.dpos = 0.025
    self.drot = np.pi / 16
    self.max_force = 500

    # Data Gen
    self.num_data_gen_envs = 5
    self.num_expert_episodes = 100

    # Training
    if results_path:
      self.results_path = os.path.join(self.root_path,
                                       'peg_insertion',
                                       results_path)
    else:
      self.results_path = os.path.join(self.root_path,
                                       'peg_insertion',
                                       datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))
    self.save_model = True
    self.training_steps = 35000
    self.batch_size = 64
    self.target_update_interval = 1
    self.checkpoint_interval = 500
    self.init_temp = 1e-2
    self.tau = 1e-2
    self.discount = 0.99

    # Eval
    self.num_eval_envs = 5
    self.num_eval_episodes = 100
    self.eval_interval = 500
    self.num_eval_intervals = int(self.training_steps / self.eval_interval)

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
      'fast_mode' : True,
      'physics_mode' : 'force',
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
      'random_orientation': self.random_orientation,
      'dpos' : self.dpos,
      'drot' : self.drot,
    }
