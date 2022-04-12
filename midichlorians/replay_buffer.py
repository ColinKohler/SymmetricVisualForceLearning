import ray
import copy
import torch
import numpy as np
import numpy.random as npr

from midichlorians import torch_utils

@ray.remote
class ReplayBuffer(object):
  '''

  '''
  def __init__(self, initial_checkpoint, initial_buffer, config):
    self.config = config
    if self.config.seed:
      npr.seed(self.config.seed)

    self.buffer = copy.deepcopy(initial_buffer)
    self.num_eps = initial_checkpoint['num_eps']
    self.num_steps = initial_checkpoint['num_steps']
    self.total_samples = sum([len(eps_history.obs_history) for eps_history in self.buffer.values()])

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

      eps_history.priorities = np.array(priorities, dtype=np.float32)
      eps_history.eps_priority = np.max(eps_history.priorities)

    # Add to buffer
    self.buffer[self.num_eps] = copy.deepcopy(eps_history)
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
     state_batch,
     obs_batch,
     force_batch,
     next_state_batch,
     next_obs_batch,
     next_force_batch,
     action_batch,
     reward_batch,
     done_batch,
     weight_batch
    ) = [list() for _ in range(11)]

    for _ in range(self.config.batch_size):
      eps_id, eps_history, eps_prob = self.sampleEps(uniform=True)
      eps_step, step_prob = self.sampleStep(eps_history, uniform=True)

      force_stack = eps_history.force_history[max(0, eps_step-3):eps_step+1]
      if len(force_stack) == 4:
        force_stack = np.array(force_stack)
      else:
        force_stack = np.pad(force_stack, ((4 - len(force_stack) % 4, 0), (0, 0)))

      force_stack_ = eps_history.force_history[max(0, eps_step-2):eps_step+2]
      if len(force_stack_) == 4:
        force_stack_ = np.array(force_stack_)
      else:
        force_stack_ = np.pad(force_stack_, ((4 - len(force_stack_) % 4, 0), (0, 0)))

      obs, force_stack, obs_, force_stack_, action = self.augmentTransitionSO2(
        eps_history.obs_history[eps_step],
        force_stack,
        eps_history.obs_history[eps_step+1],
        force_stack_,
        eps_history.force_history[eps_step+1],
        eps_history.action_history[eps_step+1]
      )
      obs = torch_utils.unnormalizeObs(obs)
      obs_ = torch_utils.unnormalizeObs(obs_)

      index_batch.append([eps_id, eps_step])
      state_batch.append(eps_history.state_history[eps_step])
      obs_batch.append(obs)
      force_batch.append(force_stack)
      next_state_batch.append(eps_history.state_history[eps_step+1])
      next_obs_batch.append(obs_)
      next_force_batch.append(force_stack_)
      action_batch.append(action)
      reward_batch.append(eps_history.reward_history[eps_step+1])
      done_batch.append(eps_history.done_history[eps_step+1])

      training_step = ray.get(shared_storage.getInfo.remote('training_step'))
      weight = (1 / (self.total_samples * eps_prob * step_prob)) ** self.config.getPerBeta(training_step)

    state_batch = torch.tensor(state_batch).long()
    obs_batch = torch.tensor(np.stack(obs_batch)).float()
    force_batch = torch.tensor(np.stack(force_batch)).float()
    next_state_batch = torch.tensor(next_state_batch).long()
    next_obs_batch = torch.tensor(np.stack(next_obs_batch)).float()
    next_force_batch = torch.tensor(np.stack(next_force_batch)).float()
    action_batch = torch.tensor(np.stack(action_batch)).float()
    reward_batch = torch.tensor(reward_batch).float()
    done_batch = torch.tensor(done_batch).int()
    non_final_mask_batch = (done_batch ^ 1).float()
    weight_batch = torch.tensor(weight_batch).float()

    state_tile = state_batch.reshape(state_batch.size(0), 1, 1, 1).repeat(1, 1, obs_batch.size(2), obs_batch.size(3))
    obs_batch = torch.cat([obs_batch, state_tile], dim=1)

    state_tile_ = next_state_batch.reshape(next_state_batch.size(0), 1, 1, 1).repeat(1, 1, next_obs_batch.size(2), next_obs_batch.size(3))
    next_obs_batch = torch.cat([next_obs_batch, state_tile_], dim=1)

    return (
      index_batch,
      (
        (obs_batch, force_batch),
        (next_obs_batch, next_force_batch),
        action_batch,
        reward_batch,
        non_final_mask_batch,
        weight_batch
      )
    )

  def sampleEps(self, uniform=False):
    '''
    Sample a episode from the buffer using the priorities

    Returns:
      (int, EpisodeHistory, double) : (episode ID, episode, episode probability)
    '''
    if uniform:
      eps_idx = npr.choice(len(self.buffer))
      eps_prob = 1.0
    else:
      eps_probs = np.array([eps_history.eps_priority for eps_history in self.buffer.values()], dtype=np.float32)
      eps_probs /= np.sum(eps_probs)

      eps_idx = npr.choice(len(self.buffer), p=eps_probs)
      eps_prob = eps_probs[eps_idx]

    eps_id = self.num_eps - len(self.buffer) + eps_idx
    return eps_id, self.buffer[eps_id], eps_prob

  def sampleStep(self, eps_history, uniform=False):
    '''
    Sample a step from the given episode using the step priorities

    Args:
      eps_history (EpisodeHistory): The episode to sample a step from

    Returns:
      (int, double) : (step index, step probability)
    '''
    if uniform:
      step_idx = npr.choice(len(eps_history.priorities[:-1]))
      step_prob = 1.0
    else:
      step_probs = eps_history.priorities[:-1] / sum(eps_history.priorities[:-1])
      step_idx = npr.choice(len(step_probs), p=step_probs)
      step_prob = step_probs[step_idx]

    return step_idx, step_prob

  def augmentTransitionSO2(self, obs, force, obs_, force_, action):
    ''''''
    obs, fxy_1, fxy_2, obs_, fxy_1_, fxy_2_, dxy, transform_params = torch_utils.perturb(
      obs[0].copy(),
      force[:,:2].copy(),
      force[:,3:5].copy(),
      obs_[0].copy(),
      force_[:,:2].copy(),
      force_[:,3:5].copy(),
      action[1:3].copy(),
      set_trans_zero=True
    )

    obs = obs.reshape(1, *obs.shape)
    force = force.copy()
    force[:,0] = fxy_1[:,0]
    force[:,1] = fxy_1[:,1]
    force[:,3] = fxy_2[:,0]
    force[:,4] = fxy_2[:,1]

    obs_ = obs_.reshape(1, *obs_.shape)
    force_ = force_.copy()
    force_[:,0] = fxy_1_[:,0]
    force_[:,1] = fxy_1_[:,1]
    force_[:,3] = fxy_2_[:,0]
    force_[:,4] = fxy_2_[:,1]

    action = action.copy()
    action[1] = dxy[0]
    action[2] = dxy[1]

    return obs, force, obs_, force_, action

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
        td_error = td_errors[i]

        self.buffer[eps_id].priorities[eps_step] = (td_error + self.config.per_eps) ** self.config.per_alpha
        self.buffer[eps_id].eps_priority = np.max(self.buffer[eps_id].priorities)

  def resetPriorities(self):
    '''
    Uniformly reset the priorities for all samples in the buffer.
    '''
    for eps_history in self.buffer.values():
      eps_history.eps_priority = 1.0
      eps_history.priorities = np.array([1.0] * len(eps_history.priorities))
