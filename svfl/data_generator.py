import ray
import time
import torch
import numpy as np
import numpy.random as npr

from svfl.agent import Agent
from svfl import torch_utils

from bulletarm import env_factory

@ray.remote
class EvalDataGenerator(object):
  '''

  '''
  def __init__(self, config, seed):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    agent = Agent(config, device)

    self.config = config
    self.data_generator = DataGenerator(agent, config, seed, evaluate=True)

  def generateEpisodes(self, num_eps, shared_storage, replay_buffer, logger):
    ''''''
    shared_storage.setInfo.remote(
      {
        'generating_eval_eps' : True,
        'run_eval_interval' : False,
      }
    )
    self.data_generator.agent.setWeights(ray.get(shared_storage.getInfo.remote('weights')))
    self.data_generator.resetEnvs()

    gen_eps = 0
    while gen_eps < num_eps:
      self.data_generator.stepEnvsAsync(shared_storage, replay_buffer, logger)
      complete_eps = self.data_generator.stepEnvsWait(shared_storage, replay_buffer, logger)
      gen_eps += complete_eps

    # Write log before moving onto the next eval interval (w/o this log for current interval may not get written)
    logger.writeLog.remote()
    logger_state = ray.get(logger.getSaveState.remote())
    if logger_state['num_eval_intervals'] < self.config.num_eval_intervals:
      logger.logEvalInterval.remote()
    shared_storage.setInfo.remote('generating_eval_eps', False)

class DataGenerator(object):
  '''
  RL Env wrapper that generates data

  Args:
    agent (midiclorians.SACAgent): Agent used to generate data
    config (dict): Task config.
    seed (int): Random seed to use for random number generation
    eval (bool): Are we generating training or evaluation data. Defaults to False
  '''
  def __init__(self, agent, config, seed, evaluate=False):
    self.seed = seed
    self.eval = evaluate
    self.config = config
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if seed:
      npr.seed(self.seed)
      torch.manual_seed(self.seed)

    self.agent = agent

    self.num_envs = self.config.num_data_gen_envs if not self.eval else self.config.num_eval_envs
    env_config = self.config.getEnvConfig()
    planner_config = self.config.getPlannerConfig()
    self.envs = env_factory.createEnvs(
      self.num_envs,
      self.config.env_type,
      env_config,
      planner_config
    )
    self.obs = None
    self.current_epsiodes = None

  def resetEnvs(self, is_expert=False):
    self.current_episodes = [EpisodeHistory(is_expert) for _ in range(self.num_envs)]
    self.obs = self.envs.reset()
    for i, eps_history in enumerate(self.current_episodes):
      eps_history.logStep(self.obs[0][i], self.obs[1][i], self.obs[2][i], np.array([0,0,0,0,0]), 0, 0, 0, self.config.max_force)

  def stepEnvsAsync(self, shared_storage, replay_buffer, logger, expert=False):
    '''
    Continuously generates data samples according to the policy specified in the config.

    Args:
      replay_buffer (ray.worker): Replay buffer worker containing data samples.
      shared_storage (ray.worker): Shared storage worker, shares data across workers.
      logger (ray.worker): Logger worker, logs training data across workers.
      expert (bool): Flag indicating if we are generating expert data or agent data. Defaults to
        False.
    '''
    if expert:
      expert_actions = torch.tensor(self.envs.getNextAction()).float()
      self.action_idxs, self.actions = self.agent.convertPlanAction(expert_actions)
      self.values = torch.zeros(self.config.num_data_gen_envs)
    else:
      self.action_idxs, self.actions, self.values = self.agent.getAction(
        self.obs[0],
        self.obs[1],
        self.obs[2],
        evaluate=self.eval
      )

    self.envs.stepAsync(self.actions, auto_reset=False)

  def stepEnvsWait(self, shared_storage, replay_buffer, logger, expert=False):
    '''
    Continuously generates data samples according to the policy specified in the config.

    Args:
      replay_buffer (ray.worker): Replay buffer worker containing data samples.
      shared_storage (ray.worker): Shared storage worker, shares data across workers.
      logger (ray.worker): Logger worker, logs training data across workers.
      expert (bool): Flag indicating if we are generating expert data or agent data. Defaults to
        False.
    '''
    obs_, rewards, dones = self.envs.stepWait()
    obs_ = list(obs_)

    for i, eps_history in enumerate(self.current_episodes):
      eps_history.logStep(
        obs_[0][i],
        obs_[1][i],
        obs_[2][i],
        self.action_idxs[i].squeeze().numpy(),
        self.values[i].item(),
        rewards[i],
        dones[i],
        self.config.max_force
      )

    done_idxs = np.nonzero(dones)[0]
    if len(done_idxs) != 0:
      new_obs_ = self.envs.reset_envs(done_idxs)

      for i, done_idx in enumerate(done_idxs):
        if not self.eval:
          replay_buffer.add.remote(self.current_episodes[done_idx], shared_storage)

        if not expert and not self.eval:
          logger.logTrainingEpisode.remote(self.current_episodes[done_idx].reward_history)

        if not expert and self.eval:
          logger.logEvalEpisode.remote(self.current_episodes[done_idx].reward_history,
                                       self.current_episodes[done_idx].value_history)

        self.current_episodes[done_idx] = EpisodeHistory(expert)
        self.current_episodes[done_idx].logStep(
          new_obs_[0][i],
          new_obs_[1][i],
          new_obs_[2][i],
          np.array([0,0,0,0,0]),
          0,
          0,
          0,
          self.config.max_force
        )

        obs_[0][done_idx] = new_obs_[0][i]
        obs_[1][done_idx] = new_obs_[1][i]
        obs_[2][done_idx] = new_obs_[2][i]

    self.obs = obs_
    return len(done_idxs)

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
