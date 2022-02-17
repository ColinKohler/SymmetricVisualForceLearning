import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import argparse
import torch
import numpy as np
import numpy.random as npr

from midichlorians.sac_agent import SACAgent
from configs.block_picking import BlockPickingConfig
from scripts.train import task_configs
from helping_hands_rl_envs import env_factory

if __name__ == '__main__':
  parser=  argparse.ArgumentParser()
  parser.add_argument('task', type=str,
    help='Task to train on.')
  parser.add_argument('checkpoint', type=str,
    help='Path to the checkpoint to load.')
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

  env_config = task_config.getEnvConfig(render=True)
  planner_config = task_config.getPlannerConfig()
  env = env_factory.createEnvs(0, task_config.env_type, env_config, planner_config)

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  agent = SACAgent(task_config, device)
  agent.setWeights(checkpoint['weights'])

  done = False
  obs = env.reset()
  while not done:
    #action = torch.tensor(env.getNextAction()).float()
    #action_idx, action = agent.convertPlanAction(action)
    action_idx, action, value = agent.getAction(obs[0], obs[2], evaluate=True)
    a = action.cpu().squeeze().tolist()
    obs, reward, done = env.step(action.cpu().squeeze().numpy(), auto_reset=False)

    print('p: {} | x: {} | y: {} | z: {} | r: {}'.format(a[0], a[1], a[2], a[3], a[4]))
    print('V1: {} | V2: {}'.format(value[0].item(), value[1].item()))
    #input('continue')

  print(reward)
