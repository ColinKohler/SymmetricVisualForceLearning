import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import group
from escnn import gspaces
from escnn import nn as enn

from svfl.models.layers import MLP

class PoseEncoder(nn.Module):
  def __init__(self, equivariant=True, z_dim=8, initialize=True, N=8):
    super().__init__()

    if equivariant:
      self.encoder = SO2PoseEncoder(z_dim=z_dim, initialize=initialize, N=N)
      self.in_type = self.encoder.in_type
      self.out_type = self.encoder.out_type
    else:
      self.encoder = MLP([5, z_dim])

  def forward(self, x):
    return x
    #return self.encoder(x)

class SO2PoseEncoder(nn.Module):
  def __init__(self, z_dim=8, initialize=True, N=8):
    super().__init__()

    self.z_dim = z_dim
    self.G = group.so2_group()
    self.gspace = gspaces.no_base_space(self.G)

    self.in_type = self.gspace.type(
      self.G.standard_representation() + self.G.standard_representation() + self.G.irrep(0)
    )

    # 3 signals, bandlimited up to freq 1
    act_1 = enn.FourierELU(
      self.gspace,
      channels=128,
      irreps=self.G.bl_regular_representation(L=1).irreps,
      inplace=True,
      type='regular',
      N=8
    )
    self.block_1 = enn.SequentialModule(
      enn.Linear(self.in_type, act_1.in_type),
      act_1
    )

    # 8 signals, bandlimited up to freq 3
    act_2 = enn.FourierELU(
      self.gspace,
      channels=256,
      irreps=self.G.bl_regular_representation(L=3).irreps,
      inplace=True,
      type='regular',
      N=16
    )
    self.block_2 = enn.SequentialModule(
      enn.Linear(self.block_1.out_type, act_2.in_type),
      act_2
    )

    # 8 signals, bandlimited up to freq 3
    act_3 = enn.FourierELU(
      self.gspace,
      channels=self.z_dim,
      irreps=self.G.bl_regular_representation(L=3).irreps,
      inplace=True,
      type='regular',
      N=16
    )
    self.block_3 = enn.SequentialModule(
      enn.Linear(self.block_2.out_type, act_3.in_type),
      act_3
    )
    self.out_type = self.block_3.out_type

  def forward(self, x):
    x = self.block_1(x)
    x = self.block_2(x)
    x = self.block_3(x)

    return x
