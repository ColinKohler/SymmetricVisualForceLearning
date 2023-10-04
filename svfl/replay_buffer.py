import ray
import copy
import torch
import random
import numpy as np
import numpy.random as npr
from random import sample

from svfl import torch_utils
from svfl.segment_tree import SumSegmentTree, MinSegmentTree
from functools import partial

class Sample(object):
  def __init__(self, obs, action, reward, obs_, done, is_expert, timeout):
    super().__init__()

    self.obs = obs
    self.action = action
    self.reward = reward
    self.obs_ = obs_
    self.done = done if not timeout else False
    self.is_expert = is_expert
    self.timeout = timeout

  def normalize(self, config):
    self.obs[0] = torch_utils.normalizePose(self.obs[0].reshape(1, -1), config.workspace).squeeze()
    self.obs[1] = torch_utils.normalizeForce(self.obs[1].reshape(1, -1), config.max_force).squeeze()
    self.obs[2] = torch_utils.normalizeProprio(self.obs[2].reshape(1, -1), config.workspace).squeeze()

    self.obs_[0] = torch_utils.normalizePose(self.obs_[0].reshape(1, -1), config.workspace).squeeze()
    self.obs_[1] = torch_utils.normalizeForce(self.obs_[1].reshape(1, -1), config.max_force).squeeze()
    self.obs_[2] = torch_utils.normalizeProprio(self.obs_[2].reshape(1, -1), config.workspace).squeeze()

class QLearningBuffer(object):
  def __init__(self, size):
    super().__init__()

    self._storage = []
    self._max_size = size
    self._next_idx = 0

  def __len__(self):
    return len(self._storage)

  def __getitem__(self, key):
    return self._storage[key]

  def __setitem__(self, key, value):
    self._storage[key] = value

  def add(self, data):
    if self._next_idx >= len(self._storage):
      self._storage.append(data)
    else:
      self._storage[self._next_idx] = data
    self._next_idx = (self._next_idx + 1) % self._max_size

  def sample(self, batch_size):
    batch_indexes = npr.choice(self.__len__(), batch_size).tolist()
    batch = [self._storage[idx] for idx in batch_indexes]
    return batch

  def getSaveState(self):
    return {
      'storage': self._storage,
      'max_size': self._max_size,
      'next_idx': self._next_idx
    }

  def loadFromState(self, save_state):
    self._storage = save_state['storage']
    self._max_size = save_state['max_size']
    self._next_idx = save_state['next_idx']

class QLearningBufferExpert(QLearningBuffer):
  def __init__(self, size):
    super().__init__(size)
    self._expert_idx = []

  def add(self, data):
    if self._next_idx >= len(self._storage):
      self._storage.append(data)
      idx = len(self._storage)-1
      self._next_idx = (self._next_idx + 1) % self._max_size
    else:
      self._storage[self._next_idx] = data
      idx = copy.deepcopy(self._next_idx)
      self._next_idx = (self._next_idx + 1) % self._max_size
      while self._storage[self._next_idx].expert:
        self._next_idx = (self._next_idx + 1) % self._max_size
    if data.is_expert:
      self._expert_idx.append(idx)

  def sample(self, batch_size):
    if len(self._expert_idx) < batch_size/2 or len(self._storage) - len(self._expert_idx) < batch_size/2:
      return super().sample(batch_size)
    expert_indexes = npr.choice(self._expert_idx, int(batch_size / 2)).tolist()
    non_expert_mask = np.ones(self.__len__(), dtype=np.bool)
    non_expert_mask[np.array(self._expert_idx)] = 0
    non_expert_indexes = npr.choice(np.arange(self.__len__())[non_expert_mask], int(batch_size/2)).tolist()
    batch_indexes = expert_indexes + non_expert_indexes
    batch = [self._storage[idx] for idx in batch_indexes]
    return batch

  def getSaveState(self):
    save_state = super().getSaveState()
    save_state['expert_idx'] = self._expert_idx
    return save_state

  def loadFromState(self, save_state):
    super().loadFromState(save_state)
    self._expert_idx = save_state['expert_idx']

@ray.remote
class ReplayBuffer(object):
  def __init__(self, initial_checkpoint, initial_buffer, config):
    self.config = config
    if self.config.seed:
      npr.seed(self.config.seed)

    it_capacity = 1
    while it_capacity < self.config.replay_buffer_size:
        it_capacity *= 2

    self._it_sum = SumSegmentTree(it_capacity)
    self._it_min = MinSegmentTree(it_capacity)
    self._max_priority = 1.0

    # TODO: Need to load checkpoint into buffer
    self.buffer = QLearningBufferExpert(self.config.replay_buffer_size)
    self.num_eps = 0
    self.num_steps = 0

  def getBuffer(self):
    return self.buffer

  def __len__(self):
    return len(self.buffer)

  def __getitem__(self, key):
    return self.buffer[key]

  def __setitem__(self, key, value):
    self.buffer[key] = value

  def add(self, sample, shared_storage):
    idx = self.buffer._next_idx
    self.buffer.add(sample)
    self._it_sum[idx] = self._max_priority ** self.config.per_alpha
    self._it_min[idx] = self._max_priority ** self.config.per_alpha

    if shared_storage:
      shared_storage.setInfo.remote('num_eps', self.num_eps)
      shared_storage.setInfo.remote('num_steps', self.num_steps)

  def _sample_proportional(self):
    res = []
    for _ in range(self.config.batch_size):
      mass = random.random() * self._it_sum.sum(0, len(self.buffer) - 1)
      idx = self._it_sum.find_prefixsum_idx(mass)
      res.append(idx)
    return res

  def sample(self, shared_storage):
    training_step = ray.get(shared_storage.getInfo.remote('training_step'))
    beta = self.config.getPerBeta(training_step)
    assert beta > 0

    idxs = self._sample_proportional()

    weights = []
    p_min = self._it_min.min() / self._it_sum.sum()
    max_weight = (p_min * len(self.buffer)) ** (-beta)

    (pose_batch,
     force_batch,
     proprio_batch,
     next_pose_batch,
     next_force_batch,
     next_proprio_batch,
     action_batch,
     reward_batch,
     done_batch,
     is_expert_batch,
    ) = [list() for _ in range(10)]

    for idx in idxs:
      p_sample = self._it_sum[idx] / self._it_sum.sum()
      weight = (p_sample * len(self.buffer)) ** (-beta)
      weights.append(weight / max_weight)

      sample = self.buffer._storage[idx]
      pose_batch.append(sample.obs[0])
      force_batch.append(sample.obs[1])
      proprio_batch.append(sample.obs[2])
      next_pose_batch.append(sample.obs_[0])
      next_force_batch.append(sample.obs_[1])
      next_proprio_batch.append(sample.obs_[2])
      action_batch.append(sample.action)
      reward_batch.append(sample.reward)
      done_batch.append(sample.done)
      is_expert_batch.append(sample.is_expert)

    pose_batch = torch.tensor(np.stack(pose_batch)).float()
    force_batch = torch.tensor(np.stack(force_batch)).float()
    proprio_batch = torch.tensor(np.stack(proprio_batch)).float()
    next_pose_batch = torch.tensor(np.stack(next_pose_batch)).float()
    next_force_batch = torch.tensor(np.stack(next_force_batch)).float()
    next_proprio_batch = torch.tensor(np.stack(next_proprio_batch)).float()
    action_batch = torch.tensor(np.stack(action_batch)).float()
    reward_batch = torch.tensor(reward_batch).float()
    done_batch = torch.tensor(done_batch).int()
    non_final_mask_batch = (done_batch ^ 1).float()
    is_expert_batch = torch.tensor(is_expert_batch).long()
    weights = torch.tensor(weights).float()

    batch = (
      (pose_batch, force_batch, proprio_batch),
      (next_pose_batch, next_force_batch, next_proprio_batch),
      action_batch,
      reward_batch,
      non_final_mask_batch,
      is_expert_batch,
    )

    return batch, weights, idxs

  def updatePriorities(self, idxs, td_error):
    expert_priorirties = np.stack([self.buffer._storage[idx].is_expert for idx in idxs]) * self.config.per_expert_eps
    priorities = np.abs(td_error) + expert_priorirties + self.config.per_eps

    assert len(idxs) == len(priorities)
    for idx, priority in zip(idxs, priorities):
      if priority <= 0:
        print("Invalid priority:", priority)
        print("All priorities:", priorities)

      assert priority > 0
      assert 0 <= idx < len(self.buffer)
      self._it_sum[idx] = priority ** self.config.per_alpha
      self._it_min[idx] = priority ** self.config.per_alpha

      self._max_priority = max(self._max_priority, priority)

  def augmentTransitionSO2(self, vision, vision_):
    ''''''
    vision_aug, vision_aug_, transform_params = torch_utils.perturb(
      vision.copy(),
      vision_.copy(),
      set_trans_zero=True
    )

    vision = vision_aug.reshape(*vision.shape)
    vision_ = vision_aug_.reshape(*vision_.shape)

    return vision, vision_

  def crop(self, vision, vision_):
    s = vision.shape[-1]

    crop_max = s - self.config.vision_size + 1
    w1 = npr.randint(0, crop_max)
    w2 = npr.randint(0, crop_max)
    vision = vision[:, w1:w1+self.config.vision_size, w2:w2+self.config.vision_size]
    vision_ = vision_[:, w1:w1+self.config.vision_size, w2:w2+self.config.vision_size]

    return vision, vision_
