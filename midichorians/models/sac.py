import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def initWeights(modules):
  '''
  Init the weigths for each module in the model

  Args:
    modules (list[torch.nn.Module]): Torch layers to initialize
  '''
  for m in modules:
    if isinstance(m, nn.Conv2d):
      nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight, gain=1)
      nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight, 1)
      nn.init.constant_(m.bias, 0)

def conv1x1(in_channels, out_channels, stride=1):
  '''
  Create a 1x1 kernel convolution layer

  Args:
    in_channels (int): Number of channels in the input
    out_channels (int): Number of channels produced by the convolution
    stride (int): Stride of the convolution. Default: 1

  Returns:
    torch.nn.Conv2d : The 1x1 convolution layer
  '''
  return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
  '''
  Create a 3x3 kernel convolution layer

  Args:
    in_channels (int): Number of channels in the input
    out_channels (int): Number of channels produced by the convolution
    stride (int): Stride of the convolution. Default: 1

  Returns:
    torch.nn.Conv2d : The 3x3 convolution layer

  '''
  return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False)

def makeLayer(block, in_channels, channels, blocks, stride=1):
  '''

  '''
  downsample = None
  if stride != 1 or in_channels != channels * block.expansion:
    downsample = nn.Sequential(
      nn.Conv2d(in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
      nn.BatchNorm2d(kernels * block.expansion)
    )

  layers = list()
  layers.append(block(in_channels, channels, stride, downsample))
  in_channels = channels * block.expansion
  for i in range(blocks):
    layers.append(block(in_channels, channels))

  return nn.Sequential(*layers)

class ResNetBlock(nn.Module):
  '''
  A ResNet block. Consists of two 3x3 convolutions.
  '''
  expansion = 1

  def __init__(self, in_channels, channels, stride=1, downsample=None):
    super().__init__()

    self.conv_1 = conv3x3(in_channels, channels, stride)
    self.bn_1 = nn.BatchNorm2d(channels)
    self.conv_2 = conv3x3(channels, channels, stride)
    self.bn_2 = nn.BatchNorm2d(channels)

    self.downsample = downsample
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    identity = x

    out = self.conv_1(x)
    out = self.bn_1(out)
    out = self.relu(out)

    out = self.conv_2(out)
    out = self.bn_2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out

class ResNet(nn.Module):
  '''
  ResNet trunk.
  '''
  def __init__(self, in_channels):
    self.in_channels = in_channels

    self.conv_1 = nn.Sequential(
      nn.Conv2d(self.in_channels, 16, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True)
    )

    self.layer_1 = makeLayer(ResNetBlock, 16, 32, 1, stride=2)
    self.layer_2 = makeLayer(ResNetBlock, 32, 64, 1, stride=2)
    self.layer_3 = makeLayer(ResNetBlock, 64, 128, 1, stride=2)
    self.layer_4 = makeLayer(ResNetBlock, 128, 256, 1, stride=2)
    self.layer_5 = makeLayer(ResNetBlock, 256, 512, 1, stride=2)

    self.flatten = nn.Flatten()
    self.fc_1 = nn.Linear(512 * 4 * 4, 1024)
    self.relu = nn.ReLU(inplace=True)

    initWeights(self.modules())

  def forward(self, x):
    out = self.conv_1(x)

    out = self.layer_1(out)
    out = self.layer_2(out)
    out = self.layer_3(out)
    out = self.layer_4(out)
    out = self.layer_5(out)

    out = seelf.flatten(out)
    out = self.fc_1(out)
    out = self.relu(out)

    return out

class Critic(nn.Module):
  '''
  Critic model.
  '''
  def __init__(self, in_channels, action_dim):
    super().__init__()

    self.resnet = ResNet(in_channels)

    self.fc_1 = nn.Sequential(
      nn.Linear(1024 + action_dim, 512),
      nn.ReLU(inplace=True),
      nn.Linear(512, 1)
    )

    self.fc_2 = nn.Sequential(
      nn.Linear(1024 + action_dim, 512),
      nn.ReLU(inplace=True),
      nn.Linear(512, 1)
    )

    initWeights(self.modules())

  def forward(self, x):
    feat = self.resnet(x)
    out_1 = self.fc_1(feat)
    out_2 = self.fc_2(feat)

    return out_1, out_2

class GaussianPolicy(nn.Module):
  '''
  Actor model that uses a Normal distribution to sample actions.
  '''
  def __init__(self, action_dim):
    super().__init__()
    self.log_sig_min = -20
    self.log_sig_max = 2
    self.eps = 1e-6

    self.resnet = ResNet(in_channels)
    self.fc_1 = nn.Linear(1024, action_dim)
    self.fc_2 = nn.Linear(1024, action_dim)

    initWeights(self.modules())

  def forward(self, x):
    feat = self.resnet(x)

    mean = self.fc_1(feat)
    log_std = self.fc_2(feat)
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
