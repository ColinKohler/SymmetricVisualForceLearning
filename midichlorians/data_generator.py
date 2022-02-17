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
          eps_history = self.generateEpisode(test_mode)
        replay_buffer.add.remote(eps_history, shared_storage)
      else:
        eps_history = self.generateEpisode(test_mode)

        past_100_rewards = ray.get(shared_storage.getInfo.remote('past_100_rewards'))
        past_100_rewards.append(eps_history.reward_history[-1])

        shared_storage.setInfo.remote(
          {
            'eps_len' : len(episode_history.obs_history),
            'total_reward' : sum(episode_history.reward_history),
            'past_100_rewards' : past_100_rewards,
            'mean_value' : np.mean([value for value in eps_history.value_history])
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

  def generateEpisode(self, test):
    '''
    Generate a single episode.

    Args:
      test (bool): Flag indicating if this is a training or evaluation data generation

    Returns:
      EpisodeHistory : Episode history
    '''
    eps_history = EpisodeHistory()

    obs = self.env.reset()
    eps_history.logStep(torch.tensor([obs[0]]).float(), torch.from_numpy(obs[2]), torch.tensor([0,0,0,0,0]), 0, 0, 0)

    done = False
    while not done:
      action_idx, action, value = self.agent.getAction(obs[0], obs[2], evaluate=test)

      obs, reward, done = self.env.step(action.cpu().squeeze().numpy(), auto_reset=False)
      eps_history.logStep(torch.tensor([obs[0]]).float(), torch.from_numpy(obs[2]), action.squeeze(), value[0], reward, done)

    return eps_history

  def generateExpertEpisode(self):
    '''
    Generate a single episode using a expert planner.

    Returns:
      None : Episode history
    '''
    eps_history = EpisodeHistory()

    obs = self.env.reset()
    eps_history.logStep(torch.tensor([obs[0]]).float(), torch.from_numpy(obs[2]), torch.tensor([0,0,0,0,0]), 0, 0, 0)

    done = False
    while not done:
      expert_action = torch.tensor(self.env.getNextAction()).float()
      expert_action_idx, expert_action = self.agent.convertPlanAction(expert_action)
      obs, reward, done = self.env.step(expert_action.cpu().squeeze().numpy(), auto_reset=False)
      eps_history.logStep(torch.tensor([obs[0]]).float(), torch.from_numpy(obs[2]), expert_action_idx.squeeze(), 0.0, reward, done)

    return eps_history

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
    #obs = obs.view(1, 1, 128, 128)
    #state_tile = state.view(1, 1, 1, 1).repeat(1, 1, obs.size(2), obs.size(3))
    #self.obs_history.append(
    #  torch.cat((obs, state_tile), dim=1)
    #)
    self.state_history.append(state)
    self.obs_history.append(obs)
    self.action_history.append(action)
    self.value_history.append(value)
    self.reward_history.append(reward)
    self.done_history.append(done)
