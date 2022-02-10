import torch
from e2cnn import gspaces
from e2cnn import nn

import numpy as np
import matplotlib.pyplot as plt

def makeLayer(block, in_filters, filters, blocks, stride=1, bnorm=True):
 downsample = None
 if stride != 1 or in_filters * block.expansion:
   r2_act = gspaces.Rot2dOnR2(N=4)
   in_type = nn.FieldType(r2_act, in_filters * [r2_act.trivial_repr])
   out_type = nn.FieldType(r2_act, filters * [r2_act.regular_repr])
   downsample = nn.SequentialModule(
     nn.R2Conv(in_filters, filters * block.expansion, kernel_size=1, stride=stride, bias=False),
     nn.InnerBatchNorm(out_type),
   )

 layers = list()
 layers.append(block(in_filters, filters, stride, downsample))
 in_filters = filters * block.expansion
 for i in range(1, blocks):
   layers.append(block(in_kernels, kernels))

 return nn.Sequential(*layers)

class C4SteerableBasicBlock(torch.nn.Module):
  expansion = 1

  def __init__(self, in_filters, filters, stride=1, downsample=None):
    super(C4SteerableBasicBlock, self).__init__()
    self.r2_act = gspaces.Rot2dOnR2(N=4)
    self.input_type = nn.FieldType(self.r2_act, in_filters * [self.r2_act.trivial_repr])

    # Conv 1
    in_type = self.input_type
    out_type = nn.FieldType(self.r2_act, filters * [self.r2_act.regular_repr])
    self.conv1 = nn.SequentialModule(
      nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
      nn.InnerBatchNorm(out_type),
      nn.ReLU(out_type, inplace=True)
    )

    # Conv 2
    in_type = out_type
    out_type = nn.FieldType(self.r2_act, filters * [self.r2_act.regular_repr])
    self.conv2 = nn.SequentialModule(
      nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
      nn.InnerBatchNorm(out_type),
    )

    self.downsample = downsample
    self.stride = stride
    self.relu = nn.ReLU(out_type, inplace=True)

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.conv2(x)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out

class C4SteerableCNN(torch.nn.Module):
  def __init__(self):
    super(C8SteerableCNN, self).__init__()

    self.feat = torch.nn.Sequential(
      makeLayer(C4SteerableBasicBlock, 1, 32, 1, stride=2),
      makeLayer(C4SteerableBasicBlock, 32, 64, 1, stride=2),
      makeLayer(C4SteerableBasicBlock, 64, 128, 1, stride=2),
      makeLayer(C4SteerableBasicBlock, 128, 256, 1, stride=2),
    )

    self.q_value_head =

    for m in self.modules():
      if isinstance(m, (nn.R2Conv)):
        torch.init.kaiming_normal_(m.weight, mode='fan_out', out=0.01, nonlinearity='leaky_relu')
      elif isinstance(m. nn.InnerBatchNorm2d):
        torch.init.constant_(m.weight, 1)
        torch.init.constant_(m.bias, 0)

  def forward(self, x):
    pass
