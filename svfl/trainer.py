import gc
import time
import copy
import ray
import torch
import torch.nn.functional as F
import numpy as np
import numpy.random as npr

from escnn import nn as enn

from svfl.agent import Agent
from svfl.data_generator import DataGenerator, EvalDataGenerator
from svfl.models.sac import Critic, GaussianPolicy
from svfl import torch_utils

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
    self.target_entropy = -self.config.action_dim
    self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
    self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3)

    # Initialize actor, and critic models
    self.actor = GaussianPolicy(
      self.config.vision_size,
      self.config.action_dim,
      z_dim=self.config.z_dim,
      encoder=self.config.encoder,
      equivariant=self.config.equivariant
    )
    self.actor.train()
    self.actor.to(self.device)

    self.critic = Critic(
      self.config.vision_size,
      self.config.action_dim,
      z_dim=self.config.z_dim,
      encoder=self.config.encoder,
      equivariant=self.config.equivariant
    )
    self.critic.train()
    self.critic.to(self.device)

    self.critic_target = Critic(
      self.config.vision_size,
      self.config.action_dim,
      z_dim=self.config.z_dim,
      encoder=self.config.encoder,
      equivariant=self.config.equivariant
    )
    self.critic_target.train()
    self.critic_target.to(self.device)
    torch_utils.softUpdate(self.critic_target, self.critic, 1.0)
    for param in self.critic_target.parameters():
      param.requires_grad = False

    self.training_step = initial_checkpoint['training_step']

    # Initialize optimizer
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                            lr=self.config.actor_lr_init)
    self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer,
                                                                  self.config.lr_decay)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                             lr=self.config.critic_lr_init)
    self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer,
                                                                   self.config.lr_decay)

    if initial_checkpoint['optimizer_state'] is not None:
      self.actor_optimizer.load_state_dict(
        copy.deepcopy(initial_checkpoint['optimizer_state'][0])
      )
      self.critic_optimizer.load_state_dict(
        copy.deepcopy(initial_checkpoint['optimizer_state'][1])
      )

    # Initialize data generator
    self.agent = Agent(self.config, self.device, actor=self.actor, critic=self.critic)
    self.data_generator = DataGenerator(self.agent, self.config, self.config.seed)

    # Set random number generator seed
    if self.config.seed:
      npr.seed(self.config.seed)
      torch.manual_seed(self.config.seed)

  def generateExpertData(self, replay_buffer, shared_storage, logger):
    '''
    Generate the amount of expert data defined in the task config.

    Args:
      replay_buffer (ray.worker): Replay buffer worker containing data samples.
      shared_storage (ray.worker): Shared storage worker, shares data across workers.
      logger (ray.worker): Logger worker, logs training data across workers.
    '''
    num_expert_eps = 0
    self.data_generator.resetEnvs(is_expert=True)
    while num_expert_eps < self.config.num_expert_episodes:
      self.data_generator.stepEnvsAsync(shared_storage, replay_buffer, logger, expert=True)
      complete_eps = self.data_generator.stepEnvsWait(shared_storage, replay_buffer, logger, expert=True)
      num_expert_eps += complete_eps

  def generateData(self, num_eps, replay_buffer, shared_storage, logger):
    '''

    Args:
      replay_buffer (ray.worker): Replay buffer worker containing data samples.
      shared_storage (ray.worker): Shared storage worker, shares data across workers.
      logger (ray.worker): Logger worker, logs training data across workers.
    '''
    current_eps = 0
    self.data_generator.resetEnvs(is_expert=False)
    while current_eps < num_eps:
      self.data_generator.stepEnvsAsync(shared_storage, replay_buffer, logger)
      complete_eps = self.data_generator.stepEnvsWait(shared_storage, replay_buffer, logger)
      current_eps += complete_eps

  def continuousUpdateWeights(self, replay_buffer, shared_storage, logger):
    '''
    Continuously sample batches from the replay buffer and perform weight updates.
    This continuous until the desired number of training steps has been reached.

    Args:
      replay_buffer (ray.worker): Replay buffer worker containing data samples.
      shared_storage (ray.worker): Shared storage worker, shares data across workers.
      logger (ray.worker): Logger worker, logs training data across workers.
    '''
    self.data_generator.resetEnvs(is_expert=False)

    next_batch = replay_buffer.sample.remote(shared_storage)
    while self.training_step < self.config.training_steps and \
          not ray.get(shared_storage.getInfo.remote('terminate')):

      # Pause training if we need to wait for eval interval to end
      while ray.get(shared_storage.getInfo.remote('pause_training')):
        time.sleep(0.1)

      self.actor.eval()
      self.critic.eval()
      self.data_generator.stepEnvsAsync(shared_storage, replay_buffer, logger)

      idx_batch, batch = ray.get(next_batch)
      next_batch = replay_buffer.sample.remote(shared_storage)

      self.actor.train()
      self.critic.train()
      td_error, loss = self.updateWeights(batch)
      replay_buffer.updatePriorities.remote(td_error.cpu(), idx_batch)
      self.training_step += 1

      self.data_generator.stepEnvsWait(shared_storage, replay_buffer, logger)

      # Update target critic towards current critic
      torch_utils.softUpdate(self.critic_target, self.critic, self.config.tau)

      # Update LRs
      if self.training_step > 0 and self.training_step % self.config.lr_decay_interval == 0:
        self.actor_scheduler.step()
        self.critic_scheduler.step()

      # Save to shared storage
      if self.training_step % self.config.checkpoint_interval == 0:
        self.actor.eval()
        self.critic.eval()
        actor_weights = torch_utils.dictToCpu(self.actor.state_dict())
        critic_weights = torch_utils.dictToCpu(self.critic.state_dict())
        actor_optimizer_state = torch_utils.dictToCpu(self.actor_optimizer.state_dict())
        critic_optimizer_state = torch_utils.dictToCpu(self.critic_optimizer.state_dict())

        shared_storage.setInfo.remote(
          {
            'weights' : copy.deepcopy((actor_weights, critic_weights)),
            'optimizer_state' : (copy.deepcopy(actor_optimizer_state), copy.deepcopy(critic_optimizer_state))
          }
        )

        if self.config.save_model:
          shared_storage.saveReplayBuffer.remote(replay_buffer.getBuffer.remote())
          shared_storage.saveCheckpoint.remote()

      # Logger/Shared storage updates
      shared_storage.setInfo.remote(
        {
          'training_step' : self.training_step,
          'run_eval_interval' : self.training_step > 0 and self.training_step % self.config.eval_interval == 0
        }
      )

      # Wait until evaluation has started
      while ray.get(shared_storage.getInfo.remote('run_eval_interval')):
        time.sleep(0.1)

      logger.updateScalars.remote(
        {
          '3.Loss/4.Actor_lr' : self.actor_optimizer.param_groups[0]['lr'],
          '3.Loss/5.Critic_lr' : self.critic_optimizer.param_groups[0]['lr'],
          '3.Loss/6.Entropy' : loss[3],
          '3.Loss/7.TD_error' : torch.mean(td_error.cpu()),
        }
      )
      logger.logTrainingStep.remote(
        {
          'Actor' : loss[0],
          'Critic' : loss[1],
          'Alpha' : loss[2],
        }
      )

  def updateWeights(self, batch):
    '''
    Perform one training step.

    Args:
      () : The current batch.

    Returns:
      (numpy.array, double) : (Priorities, Batch Loss)
    '''
    obs_batch, next_obs_batch, action_batch, reward_batch, non_final_mask_batch, is_expert_batch, weight_batch = batch

    obs_batch = (obs_batch[0].to(self.device), obs_batch[1].to(self.device), obs_batch[2].to(self.device))
    next_obs_batch = (next_obs_batch[0].to(self.device), next_obs_batch[1].to(self.device), next_obs_batch[2].to(self.device))
    action_batch = action_batch.to(self.device)
    reward_batch = reward_batch.to(self.device)
    non_final_mask_batch = non_final_mask_batch.to(self.device)
    weight_batch = weight_batch.to(self.device)

    # Critic Update
    with torch.no_grad():
      next_action, next_log_pi, _ = self.actor.sample(next_obs_batch)
      next_q1, next_q2 = self.critic_target(next_obs_batch, next_action)
      next_log_pi, next_q1, next_q2 = next_log_pi.squeeze(), next_q1.squeeze(), next_q2.squeeze()

      next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
      target_q = reward_batch + non_final_mask_batch * self.config.discount * next_q

    curr_q1, curr_q2 = self.critic(obs_batch, action_batch)
    curr_q1, curr_q2 = curr_q1.squeeze(), curr_q2.squeeze()

    critic_loss = F.mse_loss(curr_q1, target_q) + F.mse_loss(curr_q2, target_q)

    with torch.no_grad():
      td_error = 0.5 * (torch.abs(curr_q1 - target_q) + torch.abs(curr_q2 - target_q))

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Actor update
    action, log_pi, _ = self.actor.sample(obs_batch)
    q1, q2 = self.critic(obs_batch, action)

    actor_loss = torch.mean((self.alpha * log_pi) - torch.min(q1, q2))
    if is_expert_batch.sum():
      expert_weight = self.config.getExpertWeight(self.training_step)
      actor_loss += expert_weight * F.mse_loss(action[is_expert_batch], action_batch[is_expert_batch])

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # Alpha update
    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

    self.alpha_optimizer.zero_grad()
    alpha_loss.backward()
    self.alpha_optimizer.step()

    with torch.no_grad():
      entropy = -log_pi.detach().mean()
      self.alpha = self.log_alpha.exp()

    return td_error, (actor_loss.item(), critic_loss.item(), alpha_loss.item(), entropy.item())
