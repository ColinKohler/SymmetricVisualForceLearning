import torch
import e2cnn.nn

def dictToCpu(state_dict):
  cpu_dict = dict()
  for k, v in state_dict.items():
    if isinstance(v, torch.Tensor):
      cpu_dict[k] = v.cpu()
    elif isinstance(v, dict):
      cpu_dict[k] = dictToCpu(v)
    elif isinstance(v, e2cnn.nn.EquivariantModule):
      cpu_dict[k] = v.cpu()
    else:
      cpu_dict[k] = v

  return cpu_dict
