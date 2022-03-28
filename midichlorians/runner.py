import os
import shutil
import time
import copy
import collections
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt

from midichlorians.trainer import Trainer
from midichlorians.replay_buffer import ReplayBuffer
from midichlorians.data_generator import DataGenerator, EvalDataGenerator
from midichlorians.shared_storage import SharedStorage
from midichlorians.models.force_equivariant_sac import ForceEquivariantCritic, ForceEquivariantGaussianPolicy
from midichlorians import torch_utils

from helping_hands_rl_baselines.logger.ray_logger import RayLogger

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
    if self.config.seed:
      npr.seed(self.config.seed)
      torch.manual_seed(self.config.seed)
    ray.init(num_gpus=self.config.num_gpus, ignore_reinit_error=True)

    # Create log dir
    if os.path.exists(self.config.results_path):
      shutil.rmtree(self.config.results_path)
    os.makedirs(self.config.results_path)

    # Initialize checkpoint and replay buffer
    self.checkpoint = {
      'weights' : None,
      'optimizer_state' : None,
      'training_step' : 0,
      'lr' : (self.config.actor_lr_init, self.config.critic_lr_init),
      'loss' : (0, 0),
      'num_eps' : 0,
      'num_steps' : 0,
      'train_eps_reward' : list(),
      'num_eval_eps' : 100,
      'eval_mean_value' : list(),
      'eval_eps_len': list(),
      'eval_eps_reward': list(),
      'pause_training' : False,
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
    self.logger_worker = None
    self.data_gen_workers = None
    self.replay_buffer_worker = None
    self.shared_storage_worker = None
    self.training_worker = None
    self.eval_workers = None

  def initWeights(self):
    '''
    Initalize model weights
    '''
    device = torch.device('cpu')

    actor = ForceEquivariantGaussianPolicy(self.config.obs_channels, self.config.action_dim)
    actor.train()
    critic = ForceEquivariantCritic(self.config.obs_channels, self.config.action_dim)
    critic.train()

    self.checkpoint['weights'] = (
      torch_utils.dictToCpu(actor.state_dict()),
      torch_utils.dictToCpu(critic.state_dict())
    )

  def train(self):
    '''
    Initialize the various workers, start the trainers, and run the logging loop.
    '''
    self.logger_worker = RayLogger.options(num_cpus=0, num_gpus=0).remote(self.config.results_path, self.config.num_eval_episodes, self.config.__dict__)
    self.training_worker = Trainer.options(num_cpus=0, num_gpus=0.75).remote(self.checkpoint, self.config)

    self.replay_buffer_worker = ReplayBuffer.options(num_cpus=0, num_gpus=0).remote(self.checkpoint, self.replay_buffer, self.config)
    self.eval_worker = EvalDataGenerator.options(num_cpus=0, num_gpus=0.25).remote(self.config, self.config.seed+self.config.num_data_gen_envs if self.config.seed else None)

    self.shared_storage_worker = SharedStorage.remote(self.checkpoint, self.config)
    self.shared_storage_worker.setInfo.remote('terminate', False)

    # Blocking call to generate expert data
    self.training_worker.generateExpertData.remote(self.replay_buffer_worker, self.shared_storage_worker, self.logger_worker)

    # Start training
    self.training_worker.continuousUpdateWeights.remote(self.replay_buffer_worker, self.shared_storage_worker, self.logger_worker)

    self.loggingLoop()

  def loggingLoop(self):
    '''
    Initialize the testing model and log the training data
    '''
    self.save(logging=True)

    # Log training loop
    keys = [
      'training_step',
      'lr',
    ]

    info = ray.get(self.shared_storage_worker.getInfo.remote(keys))
    try:
      while info['training_step'] < self.config.training_steps:
        info = ray.get(self.shared_storage_worker.getInfo.remote(keys))

        # Eval
        if info['training_step'] > 0 and info['training_step'] % self.config.eval_interval == 0:
          if ray.get(self.shared_storage_worker.getInfo.remote('num_eval_eps')) < self.config.num_eval_episodes:
            self.shared_storage_worker.setInfo.remote('pause_training', True)
          while(ray.get(self.shared_storage_worker.getInfo.remote('num_eval_eps')) < self.config.num_eval_episodes):
            time.sleep(0.5)
          self.shared_storage_worker.setInfo.remote('pause_training', False)
          self.eval_worker.generateEpisodes.remote(self.config.num_eval_episodes, self.shared_storage_worker, self.replay_buffer_worker, self.logger_worker)

        # Logging
        self.logger_worker.updateScalars.remote(
          {
            '3.Loss/3.Actor_lr' : info['lr'][0],
            '3.Loss/4.Critic_lr' : info['lr'][0]
          }
        )
        self.logger_worker.writeLog.remote()

        time.sleep(0.5)
    except KeyboardInterrupt:
      pass

    if self.config.save_model:
      self.shared_storage_worker.setInfo.remote(copy.copy(self.replay_buffer))
      self.logger_worker.exportData.remote()
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

        print('Loaded replay buffer at {}'.format(replay_buffer_path))
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

    self.logger_worker = None
    self.data_gen_workers = None
    self.test_worker = None
    self.training_worker = None
    self.replay_buffer_worker = None
    self.shared_storage_worker = None
