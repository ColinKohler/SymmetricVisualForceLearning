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
    if vision_size == 128:
      self.obs_size = vision_size + 12
    elif vision_size == 64 or vision_size == 32:
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

    self.action_sequence = 'pxyz'
    self.action_dim =  len(self.action_sequence)

    #self.workspace = np.array([[-0.15, 0.15], [0.40, 0.70], [0.12, 0.32]]) # Larger workspace
    self.workspace = np.array([[-0.08, 0.08], [0.47, 0.63], [0.12, 0.25]]) # Small workspace for testing

    #self.dpos = 0.025
    #self.drot = np.pi / 16
    self.dpos = 0.05
    self.drot = np.pi / 8

    # Model
    self.equivariant = equivariant
    self.z_dim = 64
    self.encoder = encoder.split('+')

    # Training
    self.pre_training_steps = 0
    self.training_steps_per_action = 1
    self.root_path = '/home/helpinghands/workspace/data/'
    self.num_gpus = num_gpus
    self.expert_weight_anneal_steps = None
    self.per_beta_anneal_steps = None

  def getExpertWeight(self, step):
    if self.expert_weight_anneal_steps:
      anneal_steps = self.expert_weight_anneal_steps
    else:
      anneal_steps = self.training_steps

    r = max((anneal_steps - step) / anneal_steps, 0)
    return (self.init_expert_weight - self.end_expert_weight) * r + self.end_expert_weight

  def getPerBeta(self, step):
    if self.per_beta_anneal_steps:
      anneal_steps = self.per_beta_anneal_steps
    else:
      anneal_steps = self.training_steps

    r = max((anneal_steps - step) / anneal_steps, 0)
    return (self.init_per_beta - self.end_per_beta) * r + self.end_per_beta
