import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import ray
import argparse

from configs.block_picking import BlockPickingConfig
from configs.block_stacking import BlockStackingConfig
from configs.block_reaching import BlockReachingConfig
from configs.block_pulling import BlockPullingConfig
from configs.block_picking_corner import BlockPickingCornerConfig
from configs.block_pulling_corner import BlockPullingCornerConfig
from configs.peg_insertion import PegInsertionConfig

from midichlorians.runner import Runner

task_configs = {
  'block_reaching' : BlockReachingConfig,
  'block_picking' : BlockPickingConfig,
  'block_stacking' : BlockStackingConfig,
  'block_pulling' : BlockPullingConfig,
  'block_picking_corner' : BlockPickingCornerConfig,
  'block_pulling_corner' : BlockPullingCornerConfig,
  'peg_insertion' : PegInsertionConfig
}

if __name__ == '__main__':
  parser=  argparse.ArgumentParser()
  parser.add_argument('task', type=str,
    help='Task to train on.')
  parser.add_argument('--num_gpus', type=int, default=1,
    help='Number of GPUs to use for training.')
  parser.add_argument('--results_path', type=str, default=None,
    help='Path to save results & logs to while training. Defaults to current timestamp.')
  parser.add_argument('--checkpoint', type=str, default=None,
    help='Path to the checkpoint to load.')
  parser.add_argument('--buffer', type=str, default=None,
    help='Path to the replay buffer to load.')
  args = parser.parse_args()

  task_config = task_configs[args.task](args.num_gpus, results_path=args.results_path)
  runner = Runner(task_config, checkpoint=args.checkpoint, replay_buffer=args.buffer)

  runner.train()
  ray.shutdown()
