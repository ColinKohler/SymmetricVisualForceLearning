import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn

from midichlorians.models.layers import EquivariantBlock

class VisionEncoder(nn.Module):
  '''
  '''
  def __init__(self, vision_size=64, z_dim=64, initialize=True, N=8):
    super().__init__()

    self.z_dim = z_dim
    self.c4_act = gspaces.rot2dOnR2(N)
    self.layers = list()

    # 64x64
    self.in_type = enn.FieldType(self.c4_act, 4 * [self.c4_act.trivial_repr])
    out_type = enn.FieldType(self.c4_act, z_dim // 8 * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(self.in_type, out_type, kernel_size=3, stride=1, padding=1, initialize=initialize))
    if vision_size >= 64:
      self.layers.append(enn.PointwiseMaxPool(out_type, 2))

    # 32x32
    in_type = out_type
    out_type = enn.FieldType(self.c4_act, z_dim // 4 * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=3, stride=1, padding=1, initialize=initialize))
    if vision_size >= 32:
      self.layers.append(enn.PointwiseMaxPool(out_type, 2))

    # 16x16
    in_type = out_type
    out_type = enn.FieldType(self.c4_act, z_dim // 2 * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=3, stride=1, padding=1, initialize=initialize))
    if vision_size >= 16:
      self.layers.append(enn.PointwiseMaxPool(out_type, 2))

    # 8x8
    in_type = out_type
    out_type = enn.FieldType(self.c4_act, z_dim * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=3, stride=1, padding=1, initialize=initialize))

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, 2 * z_dim * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=3, stride=1, padding=1, initialize=initialize))

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, z_dim * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=3, stride=1, padding=0, initialize=initialize))
    self.layers.append(enn.PointwiseMaxPool(out_type, 2))

    # 3x3
    in_type = out_type
    self.out_type = enn.FieldType(self.c4_act, z_dim * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, self.out_type, kernel_size=3, stride=1, padding=0, initialize=initialize))

    self.conv = nn.Sequential(*self.layers)

  def forward(self, vision):
    vision_geo = enn.GeometricTensor(vision, self.in_type)
    return self.conv(vision_geo)