import ray
import torch
import numpy as np
import numpy.random as npr

@ray.remote
class ReplayBuffer(object):
  '''

  '''
  def __init__(self, inital_checkpoint, initial_buffer, config):
    self.config = config
    npr.seed(self.config.seed)

    self.buffer = copy.deepcopy(initial_buffer)
    self.num_eps = initial_checkpoint['num_eps']
    self.num_steps = initial_checkpoint['num_steps']
    self.total_samples = sum([len(eps_history.obs_history) for eps_history in self.buffer.values()])

    self.model = None

  def getBuffer(self):
    '''
    Get the replay buffer.

    Returns:
      list[EpisodeHistory] : The replay buffer
    '''
    return self.buffer

  def add(self, eps_history, shared_storage=None):
    '''
    Add a new episode to the replay buffer. If the episode already has priorities
    those are used, otherwise we calculate them in the standard TD error fashion:
    td_error = |V(s,a) - (V(s', a') * R(s,a) ** gamma)| + eps
    priority = td_error ** alpha

    Args:
      eps_history (EpisodeHistory): The episode to add to the buffer.
      shared_storage (ray.Worker): Shared storage worker. Defaults to None.
    '''
    if eps_history.priorities is None:
      priorities = list()
      for i, value in enumerate(eps_history.value_history):
        if (i + 1) < len(eps_history.value_history):
          priority = np.abs(value - (eps_history.reward_history[i] + self.config.discount * eps_history.value_history[i+1])) + self.config.per_eps
        else:
          priority = np.abs(value - eps_history.reward_history[i]) + self.config.per_eps
        priorities.append(priority ** self.config.per_alpha)

      eps_history.priorities = np.array(priorities, dytpe=np.float32)
      eps_history.eps_priority = np.max(eps_history.priorities)

    # Add to buffer
    self.buffer[self.num_Eps] = copy.deepcopy(eps_history)
    self.num_eps += 1
    self.num_steps += len(eps_history.obs_history)
    self.total_samples += len(eps_history.obs_history)

    # Delete the oldest episode if the buffer is full
    if self.config.replay_buffer_size < len(self.buffer):
      del_id = self.num_eps - len(self.buffer)
      self.total_samples -= len(self.buffer[del_id].obs_history)
      del self.buffer[del_id]

    if shared_storage:
      shared_storage.setInfo.remote('num_eps', self.num_eps)
      shared_storage.setInfo.remote('num_steps', self.num_steps)

  def sample(self, shared_storage):
    '''
    Sample a batch from the replay buffer.

    Args:
      shared_storage (ray.Worker): Shared storage worker.

    Returns:
      (list[int], list[numpy.array], list[numpy.array], list[double], list[double]) : (Index, Observation, Action, Reward, Weight)
    '''
    (index_batch,
     obs_batch,
     action_batch,
     reward_batch,
     q_value_batch
     weight_batch
    ) = [list() for _ in range(6)]

    for _ in range(self.config.batch_size):
      eps_id, eps_history, eps_prob = self.sampleEps()
      eps_step, step_prob = self.sampleStep(eps_history)

      idx = [eps_id, eps_step]

      training_step = ray.get(shared_storage.getInfo.remote('training_steps'))
      weight = (1 / (self.total_samples * eps_prob * step_prob)) ** self.config.getPerBeta(training_step)

  def updatePriorities(self, td_errors, idx_info):
    '''
    Update the priorities for each sample in the batch.

    Args:
      td_errors (numpy.array): The TD error for each sample in the batch
      idx_info (numpy.array): The episode and step for each sample in the batch
    '''
    for i in range(len(idx_info)):
      eps_id, eps_step = idx_info[i]

      if next(iter(self.buffer)) <= eps_id:
        td_error = td_errors[i,:]
        start_idx = eps_step
        end_idx = min(eps_step + len(td_error), len(self.buffer[eps_id].priorities))

        self.buffer[eps_id].priorities[start_idx:end_idx] = (td_error[:end_idx-start_idx] + self.config.per_eps) ** self.config.per_alpha
        self.buffer[eps_id].eps_priority = np.max(self.buffer[eps_id].priorities)

  def resetPriorities(self):
    '''
    Uniformly reset the priorities for all samples in the buffer.
    '''
    for eps_history in self.buffer.values():
      eps_history.eps_priority = 1.0
      eps_history.priorities = np.array([1.0] * len(eps_history.priorities))

  def updateTargetNetwork(self, shared_storage):
    '''
    Update the weights of the model used to generate the targets.

    Args:
      shared_storage (ray.Worker): Shared storage worker.
    '''
    pass
