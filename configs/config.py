import os
import numpy as np

class Config(object):
  '''
  Base task config.
  '''

  def __init__(self, num_gpus=1):
    # Env
    self.obs_size = 128
    self.obs_channels = 2
    self.obs_type = 'pixel'
    self.force_dim = 2
    self.force_history = 4

    self.action_sequence = 'pxyzr'
    self.action_dim =  len(self.action_sequence)

    self.workspace = np.array([[0.25, 0.65], [-0.2, 0.2], [0.01, 0.25]])
    self.view_type = 'camera_center_xyz'
    self.random_orientation = True
    self.robot = 'panda'
    self.reward_type = 'sparse'

    self.dpos = 1e-3
    self.drot = np.pi / self.dpos

    # Training
    self.root_path = '/home/kohler.c/scratch/midichlorians/'
    self.num_gpus = num_gpus
    self.gen_data_on_gpu = False
    self.per_beta_anneal_steps = None

  def getPerBeta(self, step):
    if self.per_beta_anneal_steps:
      anneal_steps = self.per_beta_anneal_steps
    else:
      anneal_steps = self.training_steps

    r  max((anneal_steps - step) / anneal_steps, 0)
    return (self.init_per_beta - self.end_per_beta) * r + self.end_per_beta

  def getEps(self, step):
    if self.eps_anneal_steps:
      anneal_steps = self.eps_anneal_steps
    else:
      anneal_steps = self.training_steps

    r = max((anneal_steps - step) / anneal_steps, 0)
    return (self.init_eps - self.end_eps) * r + self.end_eps
