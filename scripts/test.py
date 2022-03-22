import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import tqdm
import argparse
import torch
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from midichlorians.sac_agent import SACAgent
from midichlorians import torch_utils
from configs.block_picking import BlockPickingConfig
from scripts.train import task_configs
from helping_hands_rl_envs import env_factory

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
  agent = SACAgent(task_config, device)
  agent.setWeights(checkpoint['weights'])

  num_success = 0
  pbar = tqdm.tqdm(total=args.num_eps)
  pbar.set_description('SR: 0%')
  for i in range(args.num_eps):
    done = False
    obs = env.reset()
    force_stack = np.zeros((4, 6))
    force_stack[-1] = obs[3]
    while not done:
      #print(np.round(force_stack, 2))
      #print(np.round(torch_utils.normalizeForce(force_stack), 2))
      #print()
      #plt.imshow(obs[2].squeeze(), cmap='gray'); plt.show()
      action_idx, action, value = agent.getAction(
        [obs[0]],
        obs[2],
        torch_utils.normalizeForce(force_stack),
        evaluate=True
      )
      #expert_action = torch.tensor(env.getNextAction()).float()
      #action_idx, action = agent.convertPlanAction(expert_action.view(1, -1))

      obs, reward, done = env.step(action.cpu().squeeze().numpy(), auto_reset=False)
      force_stack_ = np.zeros((4, 6))
      force_stack_[:-1] = force_stack[1:]
      force_stack_[-1] = obs[3]
      force_stack = force_stack_

    if reward != 1:
      plt.imshow(obs[2].squeeze(), cmap='gray'); plt.show()

    num_success += reward
    pbar.set_description('SR: {}%'.format(int((num_success / (i+1)) * 100)))
    pbar.update(1)
  pbar.close()
