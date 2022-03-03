import time
import torch
import numpy as np
import numpy.random as npr

from midichlorians.sac_agent import SACAgent
from midichlorians import torch_utils

from helping_hands_rl_envs import env_factory

@ray.remote
class EvalDataGenerator(object):
  '''

  '''
  def __init__(self, agent, config, seed, evaluate=False):
    self.data_generator = DataGenerator(agent, config, seed, evaluate=evaluate)

  def generateEpisodes(self, num_eps, shared_storage, replay_buffer, logger):
    self.data_generator.agent.setWeights(ray.get(shared_storage_worker.getInfo('weights')))
    self.shared_storage.logEvalInterval.remote()
    self.data_generator.resetEnvs()

    while ray.get(shared_storage.getInfo('num_eval_eps')) < num_eps:
      self.data_generator.stepEnvs(shared_storage, replay_buffer, logger)

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

    npr.seed(self.seed)
    torch.manual_seed(self.seed)

    self.agent = agent

    env_config = self.config.getEnvConfig()
    planner_config = self.config.getPlannerConfig()
    self.envs = env_factory.createEnvs(
      self.config.num_data_gen_envs if self.eval else self.config.num_eval_envs,
      self.config.env_type,
      env_config,
      planner_config
    )
    self.obs = None
    self.current_epsiodes = None

  def resetEnvs(self):
    self.current_episodes = [EpisodeHistory for _ in range(self.config.num_data_gen_envs)]
    self.obs = self.envs.reset()
    for i, eps_history in enumerate(self.current_episodes):
      eps_history.logStep(
        self.obs[i,0],
        self.obs[i,2],
        np.array([0,0,0,0,0]),
        0,
        0,
        0
      )

  def stepEnvs(self, shared_storage, replay_buffer, logger, expert=False):
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
      action_idxs, actions = self.agent.convertPlanAction(expert_actions)
    else:
      action_idxs, actions, values = self.agent.getAction(
        self.obs[0],
        self.obs[2],
        evaluate=self.eval
      )

    self.envs.stepAsync(actions, auto_reset=False)
    obs_, rewards, dones = self.envs.stepWait()

    for i, eps_history in enumerate(self.current_episodes):
      eps_history.logStep(
        obs_[i,0],
        obs_[i,2],
        action_idxs[i].squeeze().numpy(),
        values[i,0].item(),
        rewards[i],
        dones[i]
      )

    done_idxs = torch.nonzero(dones).squeeze(1)
    if done_idxs.shape[0] != 0:
      new_obs_ = self.envs.reset_envs(done_idxs)
      obs_[done_idxs] = new_obs_

      for done_idx in done_indxs:
        replay_buffer.add.remote(self.current_epsodes[done_idx], shared_storage)
        self.current_episodes[done_idx] = EpisodeHistory()
        self.current_episodes[done_idx].logStep(
          obs_[done_idx,0],
          obs_[done_idx,2],
          np.array([0,0,0,0,0]),
          0,
          0,
          0
        )

        if not expert and not self.eval:
          shared_storage.logEpsReward.remote(sum(self.current_episodes[done_idx].reward_history))
          logger.logTrainingEpisode.remote(self.current_episodes[done_idx].reward_history)

        if self.eval:
          shared_storage.logEvalEpisode.remote(self.current_episodes[done_idx])
          logger.logEvalEpisode.remote(self.current_episodes[done_idx].reward_history,
                                       self.current_episodes[done_idx].value_history)

    self.obs = obs_

class EpisodeHistory(object):
  '''
  Class containing the history of an episode.
  '''
  def __init__(self):
    self.state_history = list()
    self.obs_history = list()
    self.action_history = list()
    self.value_history = list()
    self.reward_history = list()
    self.done_history = list()

    self.priorities = None
    self.eps_priority = None

  def logStep(self, state, obs, action, value, reward, done):
    self.state_history.append(state)
    self.obs_history.append(
      torch_utils.normalizeObs(obs)
    )
    self.action_history.append(action)
    self.value_history.append(value)
    self.reward_history.append(reward)
    self.done_history.append(done)
