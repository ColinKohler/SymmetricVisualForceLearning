import os
import time
import copy
import collections
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from midichlorians.trainer import Trainer
from midichlorians.replay_buffer import ReplayBuffer
from midichlorians.data_generator import DataGenerator
from midichlorians.shared_storage import SharedStorage
from midichlorians.models.sac import Critic, GaussianPolicy
from midichlorians import torch_utils

class Runner(object):
  '''
  Runner class. Used to train the model and log the results to Tensorboard.

  Args:
    config (dict): Task configuration
    checkpoint (str): Path to checkpoint to load after initialization. Defaults to None.
    replay_buffer (dict): Path to replay buffer to load after initialization. Defaults to None.
  '''
  def __init__(self, config, checkpoint=None, replay_buffer=None):
    self.config = config

    # Set random seeds
    npr.seed(self.config.seed)
    torch.manual_seed(self.config.seed)
    ray.init(num_gpus=self.config.num_gpus, ignore_reinit_error=True)

    # Initialize checkpoint and replay buffer
    self.checkpoint = {
      'weights' : None,
      'optimizer_state' : None,
      'total_reward' : 0,
      'past_100_rewards' : collections.deque([0] * 100, maxlen=100),
      'mean_value' : 0,
      'training_step' : 0,
      'lr' : (0, 0),
      'loss' : (0, 0),
      'num_eps' : 0,
      'num_steps' : 0,
      'eps_len': 0,
      'eps_reward': list(),
      'log_counter' : 0,
      'terminate' : False
    }
    self.replay_buffer = dict()

    # Load checkpoint/replay buffer
    if checkpoint:
      checkpoint = os.path.join(self.config.root_path,
                                checkpoint,
                                'model.checkpoint')
    if replay_buffer:
      replay_buffer = os.path.join(self.config.root_path,
                                   replay_buffer,
                                   'replay_buffer.pkl')
    self.load(checkpoint_path=checkpoint,
              replay_buffer_path=replay_buffer)

    if not self.checkpoint['weights']:
      self.initWeights()

    # Workers
    self.data_gen_workers = None
    self.replay_buffer_worker = None
    self.shared_storage_worker = None
    self.training_worker = None
    self.test_worker = None

  def initWeights(self):
    '''
    Initalize model weights
    '''
    device = torch.device('cpu')

    actor = GaussianPolicy(self.config.obs_channels, self.config.action_dim)
    critic = Critic(self.config.obs_channels, self.config.action_dim)
    self.checkpoint['weights'] = (
      torch_utils.dictToCpu(actor.state_dict()),
      torch_utils.dictToCpu(critic.state_dict())
    )

  def train(self):
    '''
    Initialize the various workers, start the trainers, and run the logging loop.
    '''
    self.training_worker = Trainer.options(num_cpus=0, num_gpus=1.0).remote(self.checkpoint, self.config)
    self.replay_buffer_worker = ReplayBuffer.options(num_cpus=0, num_gpus=0).remote(self.checkpoint, self.replay_buffer, self.config)
    self.data_gen_workers = [
      DataGenerator.options(num_cpus=0, num_gpus=0).remote(self.checkpoint, self.config, self.config.seed + seed)
      for seed in range(self.config.num_data_gen_workers)
    ]

    self.shared_storage_worker = SharedStorage.remote(self.checkpoint, self.config)
    self.shared_storage_worker.setInfo.remote('terminate', False)

    # Start workers
    for data_gen_worker in self.data_gen_workers:
      data_gen_worker.continuousDataGen.remote(self.shared_storage_worker, self.replay_buffer_worker)
    self.training_worker.continuousUpdateWeights.remote(self.replay_buffer_worker, self.shared_storage_worker)

    self.loggingLoop()

  def loggingLoop(self):
    '''
    Initialize the testing model and log the training data
    '''
    self.save(logging=True)

    self.test_worker = DataGenerator.options(num_cpus=0, num_gpus=0).remote(self.checkpoint, self.config, self.config.seed + self.config.num_data_gen_workers)
    self.test_worker.continuousDataGen.remote(self.shared_storage_worker, None, True)

    writer = SummaryWriter(self.config.results_path)

    # Log hyperparameters
    hp_table = [
      f'| {k} | {v} |' for k, v in self.config.__dict__.items()
    ]
    writer.add_text('Hyperparameters', '| Parameter | Value |\n|-------|-------|\n' + '\n'.join(hp_table))

    # Log training loop
    counter = self.checkpoint['log_counter']
    keys = [
      'total_reward',
      'mean_value',
      'eps_len',
      'past_100_rewards',
      'training_step',
      'eps_reward',
      'num_eps',
      'num_steps',
      'lr',
      'loss',

    ]

    info = ray.get(self.shared_storage_worker.getInfo.remote(keys))
    try:
      while info['training_step'] < self.config.training_steps:
        info = ray.get(self.shared_storage_worker.getInfo.remote(keys))

        writer.add_scalar('1.Total_reward/1.Total_reward', info['total_reward'], counter)
        writer.add_scalar('1.Total_reward/2.Mean_value', info['mean_value'], counter)
        writer.add_scalar('1.Total_reward/3.Eps_len', info['eps_len'], counter)
        writer.add_scalar('1.Total_reward/4.Success_rate', np.mean(info['past_100_rewards']), info['training_step'])
        writer.add_scalar('1.Total_reward/5.Learning_curve', np.mean(info['eps_reward'][-100:]), info['num_eps'])

        writer.add_scalar('2.Workers/1.Num_eps', info['num_eps'], counter)
        writer.add_scalar('2.Workers/2.Training_steps', info['training_step'], counter)
        writer.add_scalar('2.Workers/3.Num_steps', info['num_steps'], counter)
        writer.add_scalar('2.Workers/4.Training_steps_per_eps_step_ratio',
                          info['training_step'] / max(1, info['num_steps']),
                          counter)

        writer.add_scalar('3.Loss/1.Actor_learning_rate', info['lr'][0], counter)
        writer.add_scalar('3.Loss/2.Actor_loss', info['loss'][0], counter)
        writer.add_scalar('3.Loss/3.Critic_learning_rate', info['lr'][1], counter)
        writer.add_scalar('3.Loss/4.Critic_loss', info['loss'][1], counter)

        counter += 1
        self.shared_storage_worker.setInfo.remote({'log_counter' : counter})
        time.sleep(0.5)
    except KeyboardInterrupt:
      pass

    if self.config.save_model:
      self.shared_storage_worker.setInfo.remote(copy.copy(self.replay_buffer))
    self.terminateWorkers()

  def save(self, logging=False):
    '''
    Save the model checkpoint and replay buffer.

    Args:
      logging (bool): Print logging string when saving. Defaults to False.
    '''
    if logging:
      print('Checkpointing model at: {}'.format(self.config.results_path))
    self.shared_storage_worker.saveReplayBuffer.remote(copy.copy(self.replay_buffer))
    self.shared_storage_worker.saveCheckpoint.remote()

  def load(self, checkpoint_path=None, replay_buffer_path=None):
    '''
    Load the model checkpoint and replay buffer.

    Args:
      checkpoint_path (str): Path to model checkpoint to load. Defaults to None.
      replay_buffer_path (str): Path to replay buffer to load. Defaults to None.
    '''
    if checkpoint_path:
      if os.path.exists(checkpoint_path):
        self.checkpoint = torch.load(checkpoint_path)
        print('Loading checkpoint from {}'.format(checkpoint_path))
      else:
        print('Checkpoint not found at {}'.format(checkpoint_path))

    if replay_buffer_path:
      if os.path.exists(replay_buffer_path):
        with open(replay_buffer_path, 'rb') as f:
          data = pickle.load(f)

        self.replay_buffer = data['buffer']
        self.checkpoint['num_eps'] = data['num_eps']
        self.checkpoint['num_steps'] = data['num_steps']

        print('Loading replay buffer from {}'.format(replay_buffer_path))
      else:
        print('Replay buffer not found at {}'.format(replay_buffer_path))

  def terminateWorkers(self):
    '''
    Terminate the various workers.
    '''
    if self.shared_storage_worker:
      self.shared_storage_worker.setInfo.remote('terminate', True)
      self.checkpoint = ray.get(self.shared_storage_worker.getCheckpoint.remote())
    if self.replay_buffer_worker:
      self.replay_buffer = ray.get(self.replay_buffer_worker.getBuffer.remote())

    self.data_gen_workers = None
    self.test_worker = None
    self.training_worker = None
    self.replay_buffer_worker = None
    self.shared_storage_worker = None
