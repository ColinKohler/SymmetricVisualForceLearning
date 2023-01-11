import time
import copy
import ray
import torch
import torch.nn.functional as F
import numpy as np
import numpy.random as npr

from midichlorians.sac_agent import SACAgent
from midichlorians.data_generator import DataGenerator, EvalDataGenerator
from midichlorians.models.equivariant_sensor_fusion import EquivariantSensorFusion
from midichlorians.models.equivariant_fusion_sac import EquivariantFusionCritic, EquivariantFusionGaussianPolicy
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
    self.target_entropy = -self.config.action_dim
    self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
    self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3)

    # Initialize encoder, actor, and critic models
    self.encoder = EquivariantSensorFusion(deterministic=self.config.deterministic)
    self.encoder.train()
    self.encoder.load_state_dict(initial_checkpoint['weights'][0])
    self.encoder.to(self.device)

    self.actor = EquivariantFusionGaussianPolicy(self.config.action_dim)
    self.actor.train()
    self.actor.load_state_dict(initial_checkpoint['weights'][1])
    self.actor.to(self.device)

    self.critic = EquivariantFusionCritic(self.config.action_dim)
    self.critic.train()
    self.critic.load_state_dict(initial_checkpoint['weights'][2])
    self.critic.to(self.device)

    self.critic_target = EquivariantFusionCritic(self.config.action_dim)
    self.critic_target.train()
    self.critic_target.load_state_dict(initial_checkpoint['weights'][2])
    self.critic_target.to(self.device)

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
    self.agent = SACAgent(self.config, self.device, encoder=self.encoder, actor=self.actor, critic=self.critic)
    self.data_generator = DataGenerator(self.agent, self.config, self.config.seed)
    self.data_generator.resetEnvs()

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
    self.data_generator.resetEnvs()

    next_batch = replay_buffer.sample.remote(shared_storage)
    while self.training_step < self.config.training_steps and \
          not ray.get(shared_storage.getInfo.remote('terminate')):

      # Pause training if we need to wait for eval interval to end
      if ray.get(shared_storage.getInfo.remote('pause_training')):
        time.sleep(0.5)
        continue

      self.data_generator.stepEnvsAsync(shared_storage, replay_buffer, logger)

      idx_batch, batch = ray.get(next_batch)
      next_batch = replay_buffer.sample.remote(shared_storage)

      priorities, loss = self.updateWeights(batch)
      replay_buffer.updatePriorities.remote(priorities.cpu(), idx_batch)
      self.training_step += 1

      self.data_generator.stepEnvsWait(shared_storage, replay_buffer, logger)

      # Update target critic towards current critic
      #if self.training_step % self.config.target_update_interval == 0:
      self.softTargetUpdate()

      # Update LRs
      if self.training_step > 0 and self.training_step % self.config.lr_decay_interval == 0:
        self.actor_scheduler.step()
        self.critic_scheduler.step()

      # Save to shared storage
      if self.training_step % self.config.checkpoint_interval == 0:
        encoder_weights = torch_utils.dictToCpu(self.encoder.state_dict())
        actor_weights = torch_utils.dictToCpu(self.actor.state_dict())
        critic_weights = torch_utils.dictToCpu(self.critic.state_dict())
        actor_optimizer_state = torch_utils.dictToCpu(self.actor_optimizer.state_dict())
        critic_optimizer_state = torch_utils.dictToCpu(self.critic_optimizer.state_dict())

        shared_storage.setInfo.remote(
          {
            'weights' : copy.deepcopy((encoder_weights, actor_weights, critic_weights)),
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
      logger.updateScalars.remote(
        {
          '3.Loss/3.Actor_lr' : self.actor_optimizer.param_groups[0]['lr'],
          '3.Loss/4.Critic_lr' : self.critic_optimizer.param_groups[0]['lr']
        }
      )
      logger.logTrainingStep.remote(
        {
          'Actor' : loss[0],
          'Critic' : loss[1]
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
    obs_batch, next_obs_batch, action_batch, reward_batch, non_final_mask_batch, weight_batch = batch

    obs_batch = (obs_batch[0].to(self.device), obs_batch[1].to(self.device), obs_batch[2].to(self.device))
    next_obs_batch = (next_obs_batch[0].to(self.device), next_obs_batch[1].to(self.device), next_obs_batch[2].to(self.device))
    action_batch = action_batch.to(self.device)
    reward_batch = reward_batch.to(self.device)
    non_final_mask_batch = non_final_mask_batch.to(self.device)
    weight_batch = weight_batch.to(self.device)

    # Critic Update
    with torch.no_grad():
      if self.config.deterministic:
        next_z = self.encoder(next_obs_batch)
      else:
        next_z, mu_z, var_z, mu_prior, var_prior = self.encoder(next_obs_batch)

      next_action, next_log_pi = self.actor.sample(next_z)
      next_q1, next_q2, z = self.critic_target(next_z, next_action)
      next_log_pi, next_q1, next_q2 = next_log_pi.sqeueeze(), next_q1.squeeze(), next_q2.squeeze()

      next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
      target_q = reward_batch + non_final_mask_batch * self.config.discount * next_q

    z = self.encoder(obs_batch)
    curr_q1, curr_q2 = self.critic(z, action_batch)
    curr_q1, curr_q2 = curr_q1.squeeze(), curr_q2.squeeze()

    critic_loss = F.mse_loss(curr_q1, target_q) + F.mse_loss(curr_q2, target_q)
    #kl_loss = 0.0 * torch.mean(
    #  torch_utils.klNormal(mu_z, var_z, mu_prior.squeeze(0), var_prior.squeeze(0))
    #)
    #critic_loss = qf1_loss + qf2_loss + kl_loss

    with torch.no_grad():
      td_error = 0.5 * (torch.abs(curr_q1 - target_q) + torch.abs(curr_q2 - target_q))

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Actor update
    action, log_pi = self.actor.sample(z)
    q1, q2 = self.critic(z, action)

    actor_loss = -torch.mean(torch.min(q1, q2) - self.alpha * log_pi)
    #kl_loss = 0.0 * torch.mean(
    #  torch_utils.klNormal(mu_z, var_z, mu_prior.squeeze(0), var_prior.squeeze(0))
    #)
    #actor_loss += kl_loss

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # Alpha update
    with torch.no_grad():
      entropy = -log_pi.detach().mean()
    alpha_loss = -self.log_alpha * (self.target_entropy - entropy)

    self.alpha_optimizer.zero_grad()
    alpha_loss.backward()
    self.alpha_optimizer.step()

    with torch.no_grad():
      self.alpha = self.log_alpha.exp()

    return td_error, (actor_loss.item(), critic_loss.item())

  def softTargetUpdate(self):
    '''
    Update the target critic model to the current critic model.
    '''
    for t_param, l_param in zip(self.critic_target.parameters(), self.critic.parameters()):
      t_param.data.copy_(self.config.tau * l_param.data + (1.0 - self.config.tau) * t_param.data)
