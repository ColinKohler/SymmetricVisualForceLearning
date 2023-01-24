import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn

from midichlorians.models.layers import CausalConv1d, EquivariantBlock

class ForceEncoder(nn.Module):
  '''
  '''
  def __init__(self, z_dim=64, N=8, initialize=True):
    super().__init__()

    self.c4_act = gspaces.rot2dOnR2(N)
    self.layers = list()

    # 64x64
    self.in_type = enn.FieldType(
      self.c4_act,
      [self.c4_act.irrep(1)] + [self.c4_act.trivial_repr] + [self.c4_act.irrep(1)] + [self.c4_act.trivial_repr]
    )
    out_type = enn.FieldType(self.c4_act, z_dim // 16 * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(self.in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))
    self.layers.append(enn.PointwiseMaxPool(out_type, 2))

    # 32x32
    in_type = out_type
    out_type = enn.FieldType(self.c4_act, z_dim // 8 * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))
    self.layers.append(enn.PointwiseMaxPool(out_type, 2))

    # 16x16
    in_type = out_type
    out_type = enn.FieldType(self.c4_act, z_dim // 4 * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))
    self.layers.append(enn.PointwiseMaxPool(out_type, 2))

    # 8x8
    in_type = out_type
    out_type = enn.FieldType(self.c4_act, z_dim // 2 * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))
    self.layers.append(enn.PointwiseMaxPool(out_type, 2))

    # 4x4
    in_type = out_type
    out_type = enn.FieldType(self.c4_act, z_dim * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))
    self.layers.append(enn.PointwiseMaxPool(out_type, 2))

    # 2x2
    in_type = out_type
    out_type = enn.FieldType(self.c4_act, z_dim * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))
    self.layers.append(enn.PointwiseMaxPool(out_type, 2))

    # 1x1
    in_type = out_type
    self.out_type = enn.FieldType(self.c4_act, z_dim * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))

    self.conv = nn.Sequential(*self.layers)

  def forward(self, x):
    batch_size = x.size(0)

    x_tile = x.permute(0,2,1).view(batch_size, 6, 64)
    x_tile = torch.tile(x_tile, (1, 1, 64)).view(batch_size, 6, 64, 64)

    x_geo = enn.GeometricTensor(x_tile, self.in_type)
    return self.conv(x_geo)
