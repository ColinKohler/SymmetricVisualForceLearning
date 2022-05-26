import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from escnn import gspaces
from escnn import nn as enn

from midichlorians.models.equivariant_sac import EquivariantBlock, EquivariantLinearBlock, EquivariantDepthEncoder, EquivariantCritic, EquivariantGaussianPolicy

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
  def __init__(self, xy_channels, z_channels, depth_channels, n_out=64, initialize=True, N=8):
    super().__init__()

    self.force_enc = EquivariantForceEncoder(xy_channels, z_channels, n_out=n_out, initialize=initialize, N=N)
    self.depth_enc = EquivariantDepthEncoder(depth_channels, n_out=n_out, initialize=initialize, N=N)
    self.c4_act = gspaces.rot2dOnR2(N)

    self.layers = list()

    in_type = self.force_enc.out_type + self.depth_enc.out_type
    out_type = enn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))

    in_type = out_type
    self.out_type = enn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))

    self.conv = nn.Sequential(*self.layers)

  def forward(self, depth, force):
    batch_size = force.size(0)

    xy_force = torch.cat((force[:,:,:2], force[:,:,3:5])).view(batch_size, -1, 1, 1)
    z_force = torch.cat((force[:,:,2], force[:,:,5])).view(batch_size, -1, 1, 1)
    force_geo = enn.GeometricTensor(torch.cat((xy_force, z_force), dim=1), self.force_enc.in_type)
    force_feat = self.force_enc(force_geo)

    depth_geo = enn.GeometricTensor(depth, self.depth_enc.in_type)
    depth_feat = self.depth_enc(depth_geo)

    feat = torch.cat((depth_feat.tensor, force_feat.tensor), dim=1)
    feat = enn.GeometricTensor(feat, self.force_enc.out_type + self.depth_enc.out_type)

    return self.conv(feat)

class ForceEquivariantCritic(EquivariantCritic):
  '''
  Force equivariant critic model.
  '''
  def __init__(self, depth_channels, action_dim, n_out=64, initialize=True, N=8):
    super().__init__(depth_channels, action_dim, n_out=n_out, initialize=initialize, N=N)

    xy_channels = 2 * 100
    z_channels = 2 * 100
    self.enc = EquivariantEncoder(xy_channels, z_channels, depth_channels, n_out=n_out, initialize=initialize, N=N)

  def forward(self, depth, act):
    depth, force = depth
    batch_size = depth.size(0)

    depth = torch.cat((depth[:,0].unsqueeze(1), depth[:,2].unsqueeze(1)), dim=1)
    feat = self.enc(depth, force)

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

    xy_channels = 2 * 100
    z_channels = 2 * 100
    self.enc = EquivariantEncoder(xy_channels, z_channels, depth_channels, n_out=n_out, initialize=initialize, N=N)

  def forward(self, depth):
    depth, force = depth
    batch_size = depth.size(0)

    depth = torch.cat((depth[:,0].unsqueeze(1), depth[:,2].unsqueeze(1)), dim=1)
    feat = self.enc(depth, force)
    out = self.conv(feat).tensor.reshape(batch_size, -1)

    dxy = out[:, 0:2]
    inv_act = out[:, 2:self.action_dim]

    mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
    log_std = out[:, self.action_dim:]
    log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)

    return mean, log_std
