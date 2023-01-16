import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn

from midichlorians.models.layers import EquivariantBlock

class ProprioEncoder(nn.Module):
  '''
  '''
  def __init__(self, z_dim=64, initialize=True, N=8):
    super().__init__()

    self.z_dim = z_dim
    self.N = N

    self.c4_act = gspaces.rot2dOnR2(self.N)
    self.n_rho1 = 1
    self.invariant_proprio_repr = 3 * [self.c4_act.trivial_repr]
    self.equivariant_proprio_repr = self.n_rho1 * [self.c4_act.irrep(1)]

    self.layers = list()
    self.in_type = enn.FieldType(self.c4_act, self.equivariant_proprio_repr + self.invariant_proprio_repr)
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
    proprio_geo = enn.GeometricTensor(proprio.view(batch_size, -1, 1, 1), self.in_type)
    return self.conv(proprio_geo)
