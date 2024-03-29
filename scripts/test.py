import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import tqdm
import argparse
import torch
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt

from svfl.agent import Agent
from svfl import torch_utils
from scripts.train import task_configs
from bulletarm import env_factory

if __name__ == '__main__':
  parser=  argparse.ArgumentParser()
  parser.add_argument('task', type=str,
    help='Task to train on.')
  parser.add_argument('checkpoint', type=str,
    help='Path to the checkpoint to load.')
  parser.add_argument('--num_eps', type=int, default=100,
    help='Number of episodes to test on.')
  parser.add_argument('--vision_size', type=int, default=64,
    help='The size of the RGB-D image used for vision.')
  parser.add_argument('--encoder', type=str, default='vision+force+proprio',
    help='Type of latent encoder to use')
  parser.add_argument('--num_gpus', type=int, default=1,
    help='Number of GPUs to use for training.')
  parser.add_argument('--render', action='store_true', default=False,
    help='Render the simulation while evaluating.')
  parser.add_argument('--plot_obs', action='store_true', default=False,
    help='Render the simulation while evaluating.')
  args = parser.parse_args()

  task_config = task_configs[args.task](
    equivariant=True,
    vision_size=args.vision_size,
    encoder=args.encoder,
    num_gpus=args.num_gpus,
    results_path=args.checkpoint
  )
  checkpoint_path = os.path.join(task_config.results_path,
                                 'model.checkpoint')
  if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    print('Loading checkpoint from {}'.format(checkpoint_path))
  else:
    print('Checkpoint not found at {}'.format(checkpoint_path))
    sys.exit()

  env_config = task_config.getEnvConfig(render=args.render)
  planner_config = task_config.getPlannerConfig()
  env = env_factory.createEnvs(0, task_config.env_type, env_config, planner_config)

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  agent = Agent(task_config, device, initialize_models=False)
  agent.setWeights(checkpoint['weights'])

  num_success = 0
  pbar = tqdm.tqdm(total=args.num_eps)
  pbar.set_description('SR: 0%')
  eps_lens = list()
  for i in range(args.num_eps):
    done = False
    obs = env.reset()
    eps_lens.append(0)
    while not done:
      action_idx, action, value = agent.getAction(
        obs[0].reshape(1, *obs[0].shape),
        obs[1],
        obs[2],
        evaluate=True
      )

      if args.plot_obs:
        norm_force = torch_utils.normalizeForce(obs[1], task_config.max_force)

        fig, ax = plt.subplots(nrows=1, ncols=3)
        ax[0].imshow(obs[0][3].squeeze(), cmap='gray')
        ax[1].imshow(obs[0][:3].transpose(1,2,0))
        ax[2].plot(norm_force[:,0], label='Fx')
        ax[2].plot(norm_force[:,1], label='Fy')
        ax[2].plot(norm_force[:,2], label='Fz')
        ax[2].plot(norm_force[:,3], label='Mx')
        ax[2].plot(norm_force[:,4], label='My')
        ax[2].plot(norm_force[:,5], label='Mz')
        plt.legend()
        plt.show()

      obs, reward, done = env.step(action.cpu().squeeze().numpy(), auto_reset=False)
      eps_lens[-1] += 1

    num_success += int(reward >= 1)
    pbar.set_description('SR: {}%'.format(int((num_success / (i+1)) * 100)))
    pbar.update(1)
  print(eps_lens)
  pbar.close()
