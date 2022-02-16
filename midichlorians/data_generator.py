import ray
import torch
import numpy as np
import numpy.random as npr

from midichlorians.sac_agent import SACAgent

from helping_hands_rl_envs import env_factory

@ray.remote
class DataGenerator(object):
  '''
  Ray worker that generates data samples.

  Args:
    initial_checkpoint (dict): Checkpoint to initalize training with.
    config (dict): Task config.
    seed (int): Random seed to use for random number generation
    render (bool): Render the PyBullet env. Defaults to False
  '''
  def __init__(self, initial_checkpoint, config, seed):
    self.seed = seed
    self.config = config
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    npr.seed(self.seed)
    torch.manual_seed(self.seed)

    env_config = self.config.getEnvConfig()
    planner_config = self.config.getPlannerConfig()
    self.env = env_factory.createEnvs(0, self.config.env_type, env_config, planner_config)

    self.agent = SACAgent(self.config, self.device)
    self.agent.setWeights(initial_checkpoint['weights'])

  def continuousDataGen(self, shared_storage, replay_buffer, test_mode=False):
    '''
    Continuously generates data samples according to the policy specified in the config.

    Args:
      replay_buffer (ray.worker): Replay buffer worker containing data samples.
      shared_storage (ray.worker): Shared storage worker, shares data across workers.
      test_mode (bool): Flag indicating if we are using this worker for data generation or testing.
        Defaults to data generation (False).
    '''
    while (
      ray.get(shared_storage.getInfo.remote('training_step')) < self.config.training_steps and \
      not ray.get(shared_storage.getInfo.remote('terminate'))
    ):
      self.agent.setWeights(ray.get(shared_storage.getInfo.remote('weights')))

      if not test_mode:
        if ray.get(shared_storage.getInfo.remote('num_eps')) < self.config.num_expert_episodes:
          eps_history = self.generateExpertEpisode()
        else:
          training_step = ray.get(shared_storage.getInfo.remote('training_step'))
          eps_history = self.generateEpisode(self.config.getEps(training_step))
        replay_buffer.add.remote(eps_history, shared_storage)
      else:
        eps_history = self.generateEpisode(0.0)

        past_100_rewards = ray.get(shared_storage.getInfo.remote('past_100_rewards'))
        past_100_rewards.append(eps_history.reward_history[-1])

        shared_storage.setInfo.remote(
          {
            'past_100_rewards' : past_100_rewards
          }
        )

      if not test_mode and self.config.data_gen_delay:
        time.sleep(self.config.gen_delay)
      if not test_mode and self.config.train_data_ratio:
        while(
          ray.get(shared_storage.getInfo.remote('training_step'))
          / max(1, ray.get(shared_storoage.getInfo.remote('num_steps')))
          < self.config.train_data_ratio
        ):
          time.sleep(0.5)

  def generateEpisode(self, eps):
    '''
    Generate a single episode.

    Args:
      eps (double): Random action chance

    Returns:
      EpisodeHistory : Episode history
    '''
    eps_history = EpisodeHistory()

    obs = self.env.reset()
    eps_history.logStep(torch.tensor(obs[0]).float(), torch.from_numpy(obs[2]), torch.tensor([0,0,0,0,0]), 0, 0, 0)

    done = False
    while not done:
      if npr.rand() < eps:
        action, value = self.agent.getRandomAction(obs[0], obs[2])
      else:
        action, value = self.agent.getAction(obs[0], obs[2], evaluate=True)

      obs, reward, done = self.env.step(action.cpu().squeeze().numpy(), auto_reset=False)
      eps_history.logStep(torch.tensor(obs[0]).float(), torch.from_numpy(obs[2]), action.squeeze(), value[0], reward, done)

    return eps_history

  def generateExpertEpisode(self):
    '''
    Generate a single episode using a expert planner.

    Returns:
      None : Episode history
    '''
    eps_history = EpisodeHistory()

    obs = self.env.reset()
    eps_history.logStep(torch.tensor(obs[0]).float(), torch.from_numpy(obs[2]), torch.tensor([0,0,0,0,0]), 0, 0, 0)

    done = False
    while not done:
      expert_action = torch.tensor(self.env.getNextAction()).float()
      expert_action_idx, expert_action = self.agent.convertPlanAction(expert_action)
      obs, reward, done = self.env.step(expert_action.cpu().squeeze().numpy(), auto_reset=False)
      eps_history.logStep(torch.tensor(obs[0]).float(), torch.from_numpy(obs[2]), expert_action_idx.squeeze(), 0.0, reward, done)

    return eps_history

class EpisodeHistory(object):
  '''
  Class containing the history of an episode.
  '''
  def __init__(self):
    self.obs_history = list()
    self.action_history = list()
    self.value_history = list()
    self.reward_history = list()
    self.done_history = list()

    self.priorities = None
    self.eps_priority = None

  def logStep(self, state, obs, action, value, reward, done):
    self.obs_history.append(
      torch.cat((obs, state.reshape(1, 1, 1, 1).repeat(1, 1, obs.size(2), obs.size(3))), dim=1)
    )
    self.action_history.append(action)
    self.value_history.append(value)
    self.reward_history.append(reward)
    self.done_history.append(done)
