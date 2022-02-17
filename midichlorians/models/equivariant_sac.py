import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from e2cnn import gspaces
from e2cnn import nn as enn

def conv1x1(in_type, out_type, stride=1, padding=0):
  '''
  Create a 1x1 kernel convolution layer

  Args:
    in_channels (int): Number of channels in the input
    out_channels (int): Number of channels produced by the convolution
    stride (int): Stride of the convolution. Default: 1

  Returns:
    torch.nn.Conv2d : The 1x1 convolution layer
  '''
  return enn.R2Conv(in_type, out_type, kernel_size=1, stride=stride, padding=padding)

def conv3x3(in_type, out_type, stride=1, padding=1):
  '''
  Create a 3x3 kernel convolution layer

  Args:ww
    in_channels (int): Number of channels in the input
    out_channels (int): Number of channels produced by the convolution
    stride (int): Stride of the convolution. Default: 1

  Returns:
    torch.nn.Conv2d : The 3x3 convolution layer

  '''
  return enn.R2Conv(in_type, out_type, kernel_size=3, stride=stride, padding=padding)

def makeLayer(block, in_type, out_type, blocks, stride=1):
  '''

  '''
  layers = list()
  layers.append(block(in_type, out_type, stride))
  for i in range(1, blocks):
    layers.append(block(out_channels, out_channels))

  return nn.Sequential(*layers)

class EquivariantBlock(nn.Module):
  '''
  A equivariant ResNet block. Consists of two 3x3 convolutions.
  '''
  def __init__(self, in_type, out_type, stride=1):
    super().__init__()

    self.conv_1 = conv3x3(in_type, out_type, stride)
    #self.bn_1 = enn.InnerBatchNorm(out_type)
    self.relu_1 = enn.ReLU(out_type, inplace=True)
    #self.conv_2 = conv3x3(out_type, out_type)
    #self.bn_2 = enn.InnerBatchNorm(out_type)
    #self.relu_2 = enn.ReLU(out_type, inplace=True)

  def forward(self, x):
    #identity = x

    out = self.conv_1(x)
    #out = self.bn_1(out)
    out = self.relu_1(out)

    #out = self.conv_2(out)
    #out = self.bn_2(out)

    #if self.downsample is not None:
    #  identity = self.downsample(x)

    #out += identity
    #out = self.relu_2(out)

    return out

class EquivariantResNet(nn.Module):
  '''
  EquivariantResNet trunk.
  '''
  def __init__(self, in_channels):
    super().__init__()

    self.c4_act = gspaces.Rot2dOnR2(8)

    in_type = enn.FieldType(self.c4_act, in_channels * [self.c4_act.trivial_repr])
    out_type = enn.FieldType(self.c4_act, 8 * [self.c4_act.regular_repr])
    self.conv_1 = makeLayer(EquivariantBlock, in_type, out_type, 1, stride=1)
    self.pool_1 = enn.PointwiseMaxPool(out_type, 2)

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, 16 * [self.c4_act.regular_repr])
    self.conv_2 = makeLayer(EquivariantBlock, in_type, out_type, 1, stride=1)
    self.pool_2 = enn.PointwiseMaxPool(out_type, 2)

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, 32 * [self.c4_act.regular_repr])
    self.conv_3 = makeLayer(EquivariantBlock, in_type, out_type, 1, stride=1)
    self.pool_3 = enn.PointwiseMaxPool(out_type, 2)

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, 64 * [self.c4_act.regular_repr])
    self.conv_4 = makeLayer(EquivariantBlock, in_type, out_type, 1, stride=1)
    self.pool_4 = enn.PointwiseMaxPool(out_type, 2)

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, 128 * [self.c4_act.regular_repr])
    self.conv_5 = makeLayer(EquivariantBlock, in_type, out_type, 1, stride=1)

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, 64 * [self.c4_act.regular_repr])
    self.conv_6 = conv3x3(in_type, out_type, stride=1, padding=0)
    self.relu_6 = enn.ReLU(out_type, inplace=True)
    self.pool_6 = enn.PointwiseMaxPool(out_type, 2)

    # Output conv
    in_type = out_type
    out_type = enn.FieldType(self.c4_act, 64 * [self.c4_act.regular_repr])
    self.conv_7 = nn.Sequential(
      enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=0),
      enn.ReLU(out_type, inplace=True)
    )

  def forward(self, x):
    out = self.conv_1(x)
    out = self.pool_1(out)

    out = self.conv_2(out)
    out = self.pool_2(out)

    out = self.conv_3(out)
    out = self.pool_3(out)

    out = self.conv_4(out)
    out = self.pool_4(out)

    out = self.conv_5(out)

    out = self.conv_6(out)
    out = self.relu_6(out)
    out = self.pool_6(out)

    out = self.conv_7(out)

    return out

class EquivariantCritic(nn.Module):
  '''
  Equivariant critic model.
  '''
  def __init__(self, in_channels, action_dim):
    super().__init__()
    self.in_channels = in_channels
    self.action_dim = action_dim

    self.c4_act = gspaces.Rot2dOnR2(8)
    self.n_rho1 = 1
    self.feat_repr = 64 * [self.c4_act.regular_repr]
    self.invariant_action_repr = (self.action_dim - 2) * [self.c4_act.trivial_repr]
    self.equivariant_action_repr = self.n_rho1 * [self.c4_act.irrep(1)]

    self.in_type = enn.FieldType(self.c4_act, self.feat_repr + self.invariant_action_repr + self.equivariant_action_repr)
    self.inner_type = enn.FieldType(self.c4_act, 64 * [self.c4_act.regular_repr])
    self.out_type = enn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr])

    self.resnet = EquivariantResNet(in_channels)

    self.conv_1 = nn.Sequential(
      conv1x1(self.in_type, self.inner_type),
      enn.ReLU(self.inner_type, inplace=True),
      enn.GroupPooling(self.inner_type),
      conv1x1(enn.FieldType(self.c4_act, 64 * [self.c4_act.trivial_repr]), self.out_type)
    )

    self.conv_2 = nn.Sequential(
      conv1x1(self.in_type, self.inner_type),
      enn.ReLU(self.inner_type, inplace=True),
      enn.GroupPooling(self.inner_type),
      conv1x1(enn.FieldType(self.c4_act, 64 * [self.c4_act.trivial_repr]), self.out_type)
    )

  def forward(self, obs, act):
    batch_size = obs.size(0)

    obs_geo = enn.GeometricTensor(obs, enn.FieldType(self.c4_act, self.in_channels * [self.c4_act.trivial_repr]))
    feat = self.resnet(obs_geo)

    dxy = act[:, 1:3].reshape(batch_size,  2, 1, 1)

    inv_act = torch.cat((act[:,0:1], act[:,3:]), dim=1)
    inv_act = inv_act.reshape(batch_size, self.action_dim - 2, 1, 1)

    cat = torch.cat((feat.tensor, inv_act, dxy), dim=1)
    cat_geo = enn.GeometricTensor(cat, self.in_type)

    out_1 = self.conv_1(cat_geo).tensor.reshape(batch_size, 1)
    out_2 = self.conv_2(cat_geo).tensor.reshape(batch_size, 1)

    return out_1, out_2

class EquivariantGaussianPolicy(nn.Module):
  '''
  Equivariant actor model that uses a Normal distribution to sample actions.
  '''
  def __init__(self, in_channels, action_dim):
    super().__init__()
    self.log_sig_min = -20
    self.log_sig_max = 2
    self.eps = 1e-6

    self.in_channels = in_channels
    self.action_dim = action_dim

    self.c4_act = gspaces.Rot2dOnR2(8)
    self.n_rho1 = 1
    self.feat_repr = 64 * [self.c4_act.regular_repr]
    self.invariant_action_repr = (self.action_dim * 2 - 2) * [self.c4_act.trivial_repr]
    self.equivariant_action_repr = self.n_rho1 * [self.c4_act.irrep(1)]

    self.in_type = enn.FieldType(self.c4_act, self.feat_repr)
    self.out_type = enn.FieldType(self.c4_act, self.invariant_action_repr + self.equivariant_action_repr)

    self.resnet = EquivariantResNet(in_channels)
    self.conv = conv1x1(self.in_type, self.out_type)

  def forward(self, obs):
    batch_size = obs.size(0)

    obs_geo = enn.GeometricTensor(obs, enn.FieldType(self.c4_act, self.in_channels * [self.c4_act.trivial_repr]))
    feat = self.resnet(obs_geo)
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
