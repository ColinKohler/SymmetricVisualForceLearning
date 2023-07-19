import time
import torch
import numpy as np
import numpy.random as npr

from midichlorians import torch_utils

class EpisodeHistory(object):
  '''
  Class containing the history of an episode.
  '''
  def __init__(self, is_expert=False):
    self.vision_history = list()
    self.force_history = list()
    self.proprio_history = list()
    self.action_history = list()
    self.value_history = list()
    self.reward_history = list()
    self.done_history = list()
    self.is_expert = is_expert

    self.priorities = None
    self.eps_priority = None
    self.is_expert = is_expert

  def logStep(self, vision, force, proprio, action, value, reward, done, max_force):
    self.vision_history.append(vision)
    self.force_history.append(
      torch_utils.normalizeForce(force, max_force)
    )
    self.proprio_history.append(proprio)
    self.action_history.append(action)
    self.value_history.append(value)
    self.reward_history.append(reward)
    self.done_history.append(done)
