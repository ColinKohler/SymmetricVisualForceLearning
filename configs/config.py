import os
import numpy as np

class Config(object):
  '''
  Base task config.
  '''

  def __init__(self, equivariant=True, vision_size=64, encoder='vision+force+proprio', num_gpus=1):
    # Env
    self.obs_type = ['vision', 'force', 'proprio']
    self.vision_size = vision_size
    if vision_size == 64 or vision_size == 32:
      self.obs_size = vision_size + 12
    elif vision_size == 16:
      self.obs_size = vision_size + 4
    elif vision_size == 8:
      self.obs_size = vision_size + 2
    self.vision_channels = 1
    self.force_dim = 6
    self.force_history = 64
    self.max_force = 100
    self.proprio_dim = 5

    self.action_sequence = 'pxyzr'
    self.action_dim =  len(self.action_sequence)

    self.workspace = np.array([[-0.15, 0.15], [0.35, 0.65], [0.10, 0.45]])

    self.dpos = 0.025
    self.drot = np.pi / 16

    # Model
    self.equivariant = equivariant
    self.z_dim = 64
    self.encoder = encoder.split('+')

    # Training
    self.root_path = '/home/helpinghands/workspace/data/'
    self.num_gpus = num_gpus
    self.per_beta_anneal_steps = None

  def getPerBeta(self, step):
    if self.per_beta_anneal_steps:
      anneal_steps = self.per_beta_anneal_steps
    else:
      anneal_steps = self.training_steps

    r = max((anneal_steps - step) / anneal_steps, 0)
    return (self.init_per_beta - self.end_per_beta) * r + self.end_per_beta
