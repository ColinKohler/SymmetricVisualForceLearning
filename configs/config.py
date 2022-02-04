import os
import numpy as np

class Config(object):
  '''
  Base task config.
  '''

  def __init__(self, num_gpus=1):
    # Env
    self.obs_size = 128
    self.hand_obs_size = 24

    self.workspace = np.array([[0.2, 0.6], [-0.2, 0.2], [0, 0.4]])

    # Training
    self.root_path = None
    self.num_gpus = num_gpus
    self.gen_data_on_gpu = False
    self.per_beta_anneal_steps = None

  def getPerBeta(self, step):
    if self.per_beta_anneal_steps:
      anneal_steps = self.per_beta_anneal_steps
    else:
      anneal_steps = self.training_steps

    r = max((anneal_steps - step) / anneal_steps, 0)
    return (self.init_per_beta - self.end_per_beta) * r + self.end_per_beta
