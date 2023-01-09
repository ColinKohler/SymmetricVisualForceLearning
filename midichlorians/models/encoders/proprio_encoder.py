import torch
import torch.nn as nn
import torch.nn.functional as F

class ProprioEncoder(nn.Module):
  '''
  '''
  def __init__(self, z_dim=64):
    super().__init__()

    self.fc = nn.Sequential(
      nn.Linear(4, z_dim // 8),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Linear(z_dim // 8, z_dim // 4),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Linear(z_dim // 4, z_dim // 2),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Linear(z_dim // 2, 2 * z_dim),
      nn.LeakyReLU(0.1, inplace=True),
    )

  def forward(self, proprio):
    return self.fc(proprio)
