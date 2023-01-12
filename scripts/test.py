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

from midichlorians.agent import Agent
from midichlorians import torch_utils
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
  parser.add_argument('--render', action='store_true', default=False,
    help='Render the simulation while evaluating.')
  args = parser.parse_args()

  task_config = task_configs[args.task](1, results_path=args.checkpoint)
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
        [obs[0]],
        obs[2],
        obs[3],
        obs[4],
        evaluate=True
      )

      #print(value)
      #if np.mean(np.abs(obs[3])) > 2e-2:
      if False:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(obs[2].squeeze(), cmap='gray')
        ax[1].plot(torch_utils.normalizeForce(obs[3][:,0], task_config.max_force), label='Fx')
        ax[1].plot(torch_utils.normalizeForce(obs[3][:,1], task_config.max_force), label='Fy')
        ax[1].plot(torch_utils.normalizeForce(obs[3][:,2], task_config.max_force), label='Fz')
        ax[1].plot(torch_utils.normalizeForce(obs[3][:,3], task_config.max_force), label='Mx')
        ax[1].plot(torch_utils.normalizeForce(obs[3][:,4], task_config.max_force), label='My')
        ax[1].plot(torch_utils.normalizeForce(obs[3][:,5], task_config.max_force), label='Mz')
        plt.legend()
        plt.show()

      obs, reward, done = env.step(action.cpu().squeeze().numpy(), auto_reset=False)
      eps_lens[-1] += 1

    num_success += int(reward >= 1)
    pbar.set_description('SR: {}%'.format(int((num_success / (i+1)) * 100)))
    pbar.update(1)
  print(eps_lens)
  pbar.close()
