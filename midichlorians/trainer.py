import time
import copy
import ray
import torch
import torch.nn.functional as F
import numpy as np
import numpy.random as npr

from midichlorians.models.sac import Critic, GaussianPolicy
from midichlorians.models.equivariant_sac import EquivariantCritic, EquivariantGaussianPolicy
from midichlorians import torch_utils

@ray.remote
class Trainer(object):
  '''
  Ray worker that cordinates training of our model.

  Args:
    initial_checkpoint (dict): Checkpoint to initalize training with.
    config (dict): Task config.
  '''
  def __init__(self, initial_checkpoint, config):
    self.config = config
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    self.alpha = self.config.init_temp

    # Initialize actor and critic models
    self.actor = EquivariantGaussianPolicy(self.config.obs_channels, self.config.action_dim)
    self.actor.load_state_dict(initial_checkpoint['weights'][0])
    self.actor.to(self.device)
    self.actor.train()

    self.critic = EquivariantCritic(self.config.obs_channels, self.config.action_dim)
    self.critic.load_state_dict(initial_checkpoint['weights'][1])
    self.critic.to(self.device)
    self.critic.train()

    self.critic_target = EquivariantCritic(self.config.obs_channels, self.config.action_dim)
    self.critic_target.load_state_dict(initial_checkpoint['weights'][1])
    self.critic_target.to(self.device)

    self.training_step = initial_checkpoint['training_step']

    # Initialize optimizer
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                            lr=self.config.actor_lr_init,
                                            weight_decay=self.config.actor_weight_decay)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                             lr=self.config.critic_lr_init,
                                             weight_decay=self.config.critic_weight_decay)

    if initial_checkpoint['optimizer_state'] is not None:
      self.actor_optimizer.load_state_dict(
        copy.deepcopy(initial_checkpoint['optimizer_state'][0])
      )
      self.critic_optimizer.load_state_dict(
        copy.deepcopy(initial_checkpoint['optimizer_state'][1])
      )

    # Set random number generator seed
    npr.seed(self.config.seed)
    torch.manual_seed(self.config.seed)

  def continuousUpdateWeights(self, replay_buffer, shared_storage):
    '''
    Continuously sample batches from the replay buffer and perform weight updates.
    This continuous until the desired number of training steps has been reached.

    Args:
      replay_buffer (ray.worker): Replay buffer worker containing data samples.
      shared_storage (ray.worker): Shared storage worker, shares data across workers.
    '''
    while ray.get(shared_storage.getInfo.remote('num_eps')) == 0:
      time.sleep(0.1)

    next_batch = replay_buffer.sample.remote(shared_storage)
    while self.training_step < self.config.training_steps and \
          not ray.get(shared_storage.getInfo.remote('terminate')):
      idx_batch, batch = ray.get(next_batch)
      next_batch = replay_buffer.sample.remote(shared_storage)

      priorities, loss = self.updateWeights(batch)
      replay_buffer.updatePriorities.remote(priorities.cpu(), idx_batch)
      self.training_step += 1

      # Update target critic towards current critic
      if self.training_step % self.config.target_update_interval == 0:
        self.softTargetUpdate()

      # Save to shared storage
      if self.training_step % self.config.checkpoint_interval == 0:
        actor_weights = torch_utils.dictToCpu(self.actor.state_dict())
        critic_weights = torch_utils.dictToCpu(self.critic.state_dict())
        actor_optimizer_state = torch_utils.dictToCpu(self.actor_optimizer.state_dict())
        critic_optimizer_state = torch_utils.dictToCpu(self.critic_optimizer.state_dict())

        shared_storage.setInfo.remote(
          {
            'weights' : (copy.deepcopy(actor_weights), copy.deepcopy(critic_weights)),
            'optimizer_state' : (copy.deepcopy(actor_optimizer_state), copy.deepcopy(critic_optimizer_state))
          }
        )

        if self.config.save_model:
          shared_storage.saveReplayBuffer.remote(replay_buffer.getBuffer.remote())
          shared_storage.saveCheckpoint.remote()

      shared_storage.setInfo.remote(
        {
          'training_step' : self.training_step,
          'lr' : (self.config.actor_lr_init, self.config.critic_lr_init),
          'loss' : loss
        }
      )

      if self.config.training_delay:
        time.sleep(self.config.training_delay)

  def updateWeights(self, batch):
    '''
    Perform one training step.

    Args:
      () : The current batch.

    Returns:
      (numpy.array, double) : (Priorities, Batch Loss)
    '''
    obs_batch, next_obs_batch, action_batch, reward_batch, done_batch, weight_batch = batch

    obs_batch = obs_batch.to(self.device)
    next_obs_batch = next_obs_batch.to(self.device)
    action_batch = action_batch.to(self.device)
    reward_batch = reward_batch.to(self.device)
    done_batch = done_batch.to(self.device)
    weight_batch = weight_batch.to(self.device)

    # Critic Update
    with torch.no_grad():
      next_state_action, next_state_log_pi, _ = self.actor.sample(next_obs_batch)
      next_state_log_pi = next_state_log_pi.squeeze()

      qf1_next_target, qf2_next_target = self.critic_target(next_obs_batch, next_state_action)
      qf1_next_target = qf1_next_target.squeeze()
      qf2_next_target = qf2_next_target.squeeze()

      min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
      next_q_value = reward_batch + torch.abs(1 - done_batch) * self.config.discount * min_qf_next_target

    qf1, qf2 = self.critic(obs_batch, action_batch)
    qf1 = qf1.squeeze()
    qf2 = qf2.squeeze()

    qf1_loss = F.mse_loss(qf1, next_q_value)
    qf2_loss = F.mse_loss(qf2, next_q_value)
    critic_loss = qf1_loss + qf2_loss

    with torch.no_grad():
      td_error = 0.5 * (torch.abs(qf2 - next_q_value) + torch.abs(qf1 - next_q_value))

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Actor update
    pi, log_pi, _ = self.actor.sample(obs_batch)

    qf1_pi, qf2_pi = self.critic(obs_batch, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)

    actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    return td_error, (actor_loss.item(), critic_loss.item())

  def updateLR(self):
    '''
    Update the learning rate.
    '''
    lr = self.config.lr_init * self.config.lr_decay_rate ** (
      self.training_step / self.config.lr_decay_steps
    )

    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr

  def softTargetUpdate(self):
    '''
    Update the target critic model to the current critic model.
    '''
    for t_param, l_param in zip(self.critic_target.parameters(), self.critic.parameters()):
      t_param.data.copy_(self.config.tau * l_param.data + (1.0 - self.config.tau) * t_param.data)