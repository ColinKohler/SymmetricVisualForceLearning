import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from escnn import gspaces
from escnn import nn as enn

class EquivariantLinearBlock(nn.Module):
  '''
  A equivariant ResNet block.
  '''
  def __init__(self, in_type, out_type, initialize=True, act=True, norm=False):
    super().__init__()
    self.norm = norm
    self.act = act

    self.fc = enn.Linear(
      in_type,
      out_type,
      initialize=initialize
    )
    if self.norm:
      self.bn = enn.InnerBatchNorm(out_type)
    if self.act:
      self.relu = enn.ReLU(out_type, inplace=True)

  def forward(self, x):
    out = self.fc(x)
    if self.norm:
      out = self.bn(out)
    if self.act:
      out = self.relu(out)

    return out

class EquivariantBlock(nn.Module):
  '''
  A equivariant ResNet block.
  '''
  def __init__(self, in_type, out_type, kernel_size=3, stride=1, padding=1, initialize=True, act=True, norm=False):
    super().__init__()
    self.norm = norm
    self.act = act

    self.conv = enn.R2Conv(
      in_type,
      out_type,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      initialize=initialize
    )
    if self.norm:
      self.bn = enn.InnerBatchNorm(out_type)
    if self.act:
      self.relu = enn.ReLU(out_type, inplace=True)

  def forward(self, x):
    out = self.conv(x)
    if self.norm:
      out = self.bn(out)
    if self.act:
      out = self.relu(out)

    return out

class EquivariantDepthEncoder(nn.Module):
  '''
  '''
  def __init__(self, obs_channels, n_out=64, initialize=True, N=8):
    super().__init__()
    self.obs_channels = obs_channels
    self.c4_act = gspaces.rot2dOnR2(N)
    self.layers = list()

    in_type = enn.FieldType(self.c4_act, obs_channels * [self.c4_act.trivial_repr])
    self.in_type = in_type
    out_type = enn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr])
    self.layers.append(
      EquivariantBlock(in_type, out_type, kernel_size=3, stride=1, padding=1, initialize=initialize)
    )
    self.layers.append(enn.PointwiseMaxPool(out_type, 2))

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, n_out // 4 * [self.c4_act.regular_repr])
    self.layers.append(
      EquivariantBlock(in_type, out_type, kernel_size=3, stride=1, padding=1, initialize=initialize)
    )
    self.layers.append(enn.PointwiseMaxPool(out_type, 2))

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, n_out // 2 * [self.c4_act.regular_repr])
    self.layers.append(
      EquivariantBlock(in_type, out_type, kernel_size=3, stride=1, padding=1, initialize=initialize)
    )
    self.layers.append(enn.PointwiseMaxPool(out_type, 2))

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr])
    self.layers.append(
      EquivariantBlock(in_type, out_type, kernel_size=3, stride=1, padding=1, initialize=initialize)
    )
    self.layers.append(enn.PointwiseMaxPool(out_type, 2))

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, 2 * n_out * [self.c4_act.regular_repr])
    self.layers.append(
      EquivariantBlock(in_type, out_type, kernel_size=3, stride=1, padding=1, initialize=initialize)
    )

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr])
    self.layers.append(
      EquivariantBlock(in_type, out_type, kernel_size=3, stride=1, padding=0, initialize=initialize)
    )
    self.layers.append(enn.PointwiseMaxPool(out_type, 2))

    in_type = out_type
    self.out_type = enn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr])
    self.layers.append(
      EquivariantBlock(in_type, self.out_type, kernel_size=3, stride=1, padding=0, initialize=initialize)
    )

    self.conv_1 = nn.Sequential(*self.layers)

  def forward(self, obs):
    return self.conv_1(obs)

class EquivariantCritic(nn.Module):
  '''
  Equivariant critic model.
  '''
  def __init__(self, obs_channels, action_dim, n_out=64, initialize=True, N=8):
    super().__init__()
    self.obs_channels = obs_channels
    self.n_out = n_out
    self.action_dim = action_dim
    self.N = N

    self.c4_act = gspaces.rot2dOnR2(self.N)
    self.n_rho1 = 1
    self.feat_repr = self.n_out * [self.c4_act.regular_repr]
    self.invariant_action_repr = (self.action_dim - 2) * [self.c4_act.trivial_repr]
    self.equivariant_action_repr = self.n_rho1 * [self.c4_act.irrep(1)]

    self.in_type = enn.FieldType(self.c4_act, self.feat_repr + self.invariant_action_repr + self.equivariant_action_repr)
    self.inner_type = enn.FieldType(self.c4_act, self.n_out * [self.c4_act.regular_repr])
    self.inner_type_2 = enn.FieldType(self.c4_act, self.n_out * [self.c4_act.trivial_repr])
    self.out_type = enn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr])

    self.enc = EquivariantDepthEncoder(obs_channels, n_out=n_out, initialize=initialize, N=N)

    self.conv_1 = nn.Sequential(
      EquivariantBlock(self.in_type, self.inner_type, kernel_size=1, stride=1, padding=0, initialize=initialize),
      enn.GroupPooling(self.inner_type),
      EquivariantBlock(self.inner_type_2, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize, act=False)
    )

    self.conv_2 = nn.Sequential(
      EquivariantBlock(self.in_type, self.inner_type, kernel_size=1, stride=1, padding=0, initialize=initialize),
      enn.GroupPooling(self.inner_type),
      EquivariantBlock(self.inner_type_2, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize, act=False)
    )

  def forward(self, obs, act):
    batch_size = obs.size(0)

    obs_geo = enn.GeometricTensor(obs, enn.FieldType(self.c4_act, self.obs_channels * [self.c4_act.trivial_repr]))
    feat = self.enc(obs_geo)

    dxy = act[:, 1:3].reshape(batch_size,  2, 1, 1)

    inv_act = torch.cat((act[:,0:1], act[:,3:]), dim=1)
    n_inv = inv_act.shape[1]
    inv_act = inv_act.reshape(batch_size, n_inv, 1, 1)

    cat = torch.cat((feat.tensor, inv_act, dxy), dim=1)
    cat_geo = enn.GeometricTensor(cat, self.in_type)

    out_1 = self.conv_1(cat_geo).tensor.reshape(batch_size, 1)
    out_2 = self.conv_2(cat_geo).tensor.reshape(batch_size, 1)

    return out_1, out_2

class EquivariantGaussianPolicy(nn.Module):
  '''
  Equivariant actor model that uses a Normal distribution to sample actions.
  '''
  def __init__(self, obs_channels, action_dim, n_out=64, initialize=True, N=8):
    super().__init__()
    self.log_sig_min = -20
    self.log_sig_max = 2
    self.eps = 1e-6

    self.obs_channels = obs_channels
    self.action_dim = action_dim
    self.n_out = n_out
    self.initialize = initialize

    self.c4_act = gspaces.rot2dOnR2(N)
    self.n_rho1 = 1
    self.feat_repr = self.n_out * [self.c4_act.regular_repr]
    self.invariant_action_repr = (self.action_dim * 2 - 2) * [self.c4_act.trivial_repr]
    self.equivariant_action_repr = self.n_rho1 * [self.c4_act.irrep(1)]

    self.in_type = enn.FieldType(self.c4_act, self.feat_repr)
    self.out_type = enn.FieldType(self.c4_act, self.equivariant_action_repr + self.invariant_action_repr)

    self.enc = EquivariantDepthEncoder(obs_channels, n_out=n_out, initialize=initialize, N=N)
    self.conv = EquivariantBlock(self.in_type, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize, act=False)

  def forward(self, obs):
    batch_size = obs.size(0)

    obs_geo = enn.GeometricTensor(obs, enn.FieldType(self.c4_act, self.obs_channels * [self.c4_act.trivial_repr]))
    feat = self.enc(obs_geo)
    out = self.conv(feat).tensor.reshape(batch_size, -1)

    dxy = out[:, 0:2]
    inv_act = out[:, 2:self.action_dim]

    mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
    log_std = out[:, self.action_dim:]
    log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)

    return mean, log_std

  def sample(self, x):
    '''
    Sample an action from a Normal distribution generated by the model.
    '''
    mean, log_std = self.forward(x)
    std = log_std.exp()

    normal = Normal(mean, std)
    x_t = normal.rsample()
    y_t = torch.tanh(x_t)
    action = y_t

    log_prob = normal.log_prob(x_t)
    log_prob -= torch.log((1 - y_t.pow(2)) + self.eps)
    log_prob = log_prob.sum(1, keepdim=True)
    mean = torch.tanh(mean)

    return action, log_prob, mean
