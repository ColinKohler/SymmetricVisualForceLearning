import ray
import argparse

import tasks
from midichlorians.runner import Runner

if __name__ == '__main__':
  parser=  argparse.ArgumentParser()
  parser.add_argument('test', type=str,
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

  task_config = tasks.getTaskConfig(args.task, args.num_Gpus, results_path=args.results_path)
  runner = Runner(task_config, checkpoint=args.checkpoint, replay_buffer=args.buffer)

  runner.train()
  runner.shutdown()
