import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn

from midichlorians.models.layers import EquivariantBlock, ConvBlock

class ProprioEncoder(nn.Module):
  def __init__(self, equivariant=False, z_dim=64, initialize=True, N=8):
    super().__init__()
    if equivariant:
      self.encoder = EquivProprioEncoder(z_dim=z_dim, initialize=initialize, N=N)
    else:
      self.encoder = CnnProprioEncoder(z_dim=z_dim)

  def forward(self, x):
    return self.encoder(x)

class EquivProprioEncoder(nn.Module):
  '''
  '''
  def __init__(self, z_dim=64, initialize=True, N=8):
    super().__init__()

    self.z_dim = z_dim
    self.N = N

    self.c4_act = gspaces.rot2dOnR2(self.N)
    self.proprio_repr = 1 * [self.c4_act.trivial_repr] + [self.c4_act.irrep(1)] + 2 * [self.c4_act.trivial_repr]

    self.layers = list()
    self.in_type = enn.FieldType(self.c4_act, self.proprio_repr)
    out_type = enn.FieldType(self.c4_act, z_dim // 4 * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(self.in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, z_dim // 2 * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))

    in_type = out_type
    out_type = enn.FieldType(self.c4_act, z_dim * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))

    in_type = out_type
    self.out_type = enn.FieldType(self.c4_act, z_dim * [self.c4_act.regular_repr])
    self.layers.append(EquivariantBlock(in_type, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))

    self.conv = nn.Sequential(*self.layers)

  def forward(self, proprio):
    batch_size = proprio.size(0)

    proprio = proprio.reshape(batch_size, -1, 1, 1)
    proprio_geo = enn.GeometricTensor(proprio, self.in_type)
    return self.conv(proprio_geo)

class CnnProprioEncoder(nn.Module):
  '''
  '''
  def __init__(self, z_dim=64):
    super().__init__()

    self.z_dim = z_dim

    self.layers = list()
    self.layers.append(ConvBlock(5, z_dim // 4 * 2, kernel_size=1, stride=1, padding=0))
    self.layers.append(ConvBlock(z_dim // 4 * 2, z_dim // 2 * 2, kernel_size=1, stride=1, padding=0))
    self.layers.append(ConvBlock(z_dim // 2 * 2, z_dim * 2, kernel_size=1, stride=1, padding=0))
    self.layers.append(ConvBlock(z_dim * 2, z_dim, kernel_size=1, stride=1, padding=0))

    self.conv = nn.Sequential(*self.layers)

  def forward(self, proprio):
    batch_size = proprio.size(0)

    proprio = proprio.reshape(batch_size, -1, 1, 1)
    return self.conv(proprio)
