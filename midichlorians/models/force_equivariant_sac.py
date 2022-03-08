import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from e2cnn import gspaces
from e2cnn import nn as enn

from midichlorians.models.equivariant_sac import EquivariantBlock, EquivariantResNet, EquivariantCritic, EquivariantActor

class ForceEquivariantResNet(EquivariantResNet):
  '''
  ForceEquivariantResNet trunk.
  '''
  def __init__(self, obs_channels, n_out=64, initialize=True, N=8):
    super().__init__(obs_channels, n_out=n_out, initialize=initialize, N=N)

    self.in_type = enn.FieldType(
      self.c4_act,
      n_out * [self.c4_act.regular_repr] + 8 * [self.c4_act.trivial_repr]
    )
    out_type = enn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
    self.conv_2 = EquivariantBlock(self.in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize)

  def forward(self, obs, force):
    batch_size = obs.size(0)

    feat = super().forward(obs)
    obs_force = torch.cat((feat.tensor, force.view(batch_size, -1, 1, 1)), dim=1)
    obs_force = enn.GeometricTensor(obs_force, self.in_type)

    return self.conv_2(obs_force)

class ForceEquivariantCritic(EquivariantCritic):
  '''
  Force equivariant critic model.
  '''
  def __init__(self, obs_channels, action_dim, n_out=64, initialize=True, N=8):
    super().__init__(obs_channels, action_dim, n_out=n_out, initialize=initialize, N=N)

    self.resnet = ForceEquivariantResNet(obs_channels, n_out=n_out, initialize=initialize, N=N)

  def forward(self, obs, act):
    batch_size = obs.size(0)
    obs, force = obs

    obs_geo = enn.GeometricTensor(obs, enn.FieldType(self.c4_act, self.in_channels * [self.c4_act.trivial_repr]))
    feat = self.resnet(obs_geo, force)

    dxy = act[:, 1:3].reshape(batch_size,  2, 1, 1)

    inv_act = torch.cat((act[:,0:1], act[:,3:]), dim=1)
    inv_act = inv_act.reshape(batch_size, self.action_dim - 2, 1, 1)

    cat = torch.cat((feat.tensor, inv_act, dxy), dim=1)
    cat_geo = enn.GeometricTensor(cat, self.in_type)

    out_1 = self.conv_1(cat_geo).tensor.reshape(batch_size, 1)
    out_2 = self.conv_2(cat_geo).tensor.reshape(batch_size, 1)

    return out_1, out_2

class ForceEquivariantGaussianPolicy(EquivariantGaussianPolicy):
  '''
  Equivariant actor model that uses a Normal distribution to sample actions.
  '''
  def __init__(self, obs_channels, action_dim, n_out=64, initialize=True, N=8):
    super().__init__(obs_channels, action_dim, n_out=n_out, initialize=initialize, N=N)

  def forward(self, obs):
    batch_size = obs.size(0)
    obs, force = obs

    obs_geo = enn.GeometricTensor(obs, enn.FieldType(self.c4_act, self.in_channels * [self.c4_act.trivial_repr]))
    feat = self.resnet(obs_geo, force)
    out = self.conv(feat).tensor.reshape(batch_size, -1)

    dxy = out[:, 0:2]
    inv_act = out[:, 2:self.action_dim]

    mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
    log_std = out[:, self.action_dim:]
    log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)

    return mean, log_std