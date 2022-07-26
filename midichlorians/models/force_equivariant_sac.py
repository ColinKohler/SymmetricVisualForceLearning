import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from escnn import gspaces
from escnn import nn as enn

from midichlorians.models.equivariant_sac import EquivariantBlock, EquivariantLinearBlock, EquivariantDepthEncoder, EquivariantCritic, EquivariantGaussianPolicy

class CausalConv1d(nn.Conv1d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
    self.__padding = (kernel_size - 1) * dilation

    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=self.__padding, dilation=dilation, bias=bias)

  def forward(self, x):
    res = super().forward(x)
    if self.__padding != 0:
      return res[:, :, :-self.__padding]
    return res

class ForceEncoder(nn.Module):
  '''

  '''
  def __init__(self, n_out):
    super().__init__()
    self.conv = nn.Sequential(
      CausalConv1d(6, 8, kernel_size=2, stride=2),
      nn.LeakyReLU(0.1, inplace=True),
      CausalConv1d(8, 16, kernel_size=2, stride=2),
      nn.LeakyReLU(0.1, inplace=True),
      CausalConv1d(16, 32, kernel_size=2, stride=2),
      nn.LeakyReLU(0.1, inplace=True),
      CausalConv1d(32, 64, kernel_size=2, stride=2),
      nn.LeakyReLU(0.1, inplace=True),
      CausalConv1d(64, 128, kernel_size=2, stride=2),
      nn.LeakyReLU(0.1, inplace=True),
      CausalConv1d(128, n_out, kernel_size=2, stride=2),
      nn.LeakyReLU(0.1, inplace=True),
    )

  def forward(self, x):
    return self.conv(x)

class EquivariantForceEncoder(nn.Module):
  '''

  '''
  def __init__(self, xy_channels, z_channels, n_out=64, initialize=True, N=8):
    super().__init__()

    self.c4_act = gspaces.rot2dOnR2(N)
    self.layers = list()

    self.xy_force_type = xy_channels * [self.c4_act.irrep(1)]
    self.z_force_type = z_channels * [self.c4_act.trivial_repr]

    self.in_type = enn.FieldType(
      self.c4_act,
      xy_channels * [self.c4_act.irrep(1)] + z_channels * [self.c4_act.trivial_repr]
    )
    out_type = enn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(self.in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, n_out // 4 * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))

    in_type = out_type
    self.out_type = enn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))

    self.conv = nn.Sequential(*self.layers)

  def forward(self, x):
    return self.conv(x)

class EquivariantEncoder(nn.Module):
  '''
  '''
  def __init__(self, depth_channels, n_out=64, initialize=True, N=8):
    super().__init__()

    #self.force_enc = EquivariantForceEncoder(xy_channels, z_channels, n_out=n_out, initialize=initialize, N=N)
    self.force_enc = ForceEncoder(n_out)
    self.depth_enc = EquivariantDepthEncoder(depth_channels, n_out=n_out, initialize=initialize, N=N)
    self.c4_act = gspaces.rot2dOnR2(N)

    self.layers = list()

    self.force_out_type = enn.FieldType(self.c4_act, n_out * [self.c4_act.trivial_repr])
    in_type = self.force_out_type + self.depth_enc.out_type
    out_type = enn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))

    in_type = out_type
    self.out_type = enn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))

    self.conv = nn.Sequential(*self.layers)

  def forward(self, depth, force):
    batch_size = force.size(0)

    force_feat = self.force_enc(force.view(batch_size, 6, 64))

    depth_geo = enn.GeometricTensor(depth, self.depth_enc.in_type)
    depth_feat = self.depth_enc(depth_geo)

    feat = torch.cat((depth_feat.tensor, force_feat.reshape(batch_size, -1, 1, 1)), dim=1)
    feat = enn.GeometricTensor(feat, self.force_out_type + self.depth_enc.out_type)

    return self.conv(feat)

class ForceEquivariantCritic(EquivariantCritic):
  '''
  Force equivariant critic model.
  '''
  def __init__(self, depth_channels, action_dim, n_out=64, initialize=True, N=8):
    super().__init__(depth_channels, action_dim, n_out=n_out, initialize=initialize, N=N)

    self.enc = EquivariantEncoder(depth_channels, n_out=n_out, initialize=initialize, N=N)
    #self.enc = EquivariantDepthEncoder(depth_channels, n_out=n_out, initialize=initialize, N=N)

  def forward(self, depth, act):
    depth, force = depth
    batch_size = depth.size(0)

    feat = self.enc(depth, force)
    #depth_geo = enn.GeometricTensor(depth, self.enc.in_type)
    #feat = self.enc(depth_geo)

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
  def __init__(self, depth_channels, action_dim, n_out=64, initialize=True, N=8):
    super().__init__(depth_channels, action_dim, n_out=n_out, initialize=initialize, N=N)

    self.enc = EquivariantEncoder(depth_channels, n_out=n_out, initialize=initialize, N=N)
    #self.enc = EquivariantDepthEncoder(depth_channels, n_out=n_out, initialize=initialize, N=N)

  def forward(self, depth):
    depth, force = depth
    batch_size = depth.size(0)

    feat = self.enc(depth, force)
    #depth_geo = enn.GeometricTensor(depth, self.enc.in_type)
    #feat = self.enc(depth_geo)
    out = self.conv(feat).tensor.reshape(batch_size, -1)

    dxy = out[:, 0:2]
    inv_act = out[:, 2:self.action_dim]

    mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
    log_std = out[:, self.action_dim:]
    log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)

    return mean, log_std
