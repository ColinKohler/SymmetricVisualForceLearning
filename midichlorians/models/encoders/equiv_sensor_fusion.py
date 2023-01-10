import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn

from midichlorians.models.layers import EquivariantBlock
from midichlorians.models.encoders.proprio_encoder import ProprioEncoder
from midichlorians.models.encoders.equiv_depth_encoder import EquivariantDepthEncoder
from midichlorians.models.encoders.equiv_force_encoder import EquivariantForceEncoder

class EquivariantSensorFusion(nn.Module):
  def __init__(self, z_dim=64, N=8, initialize=True):
    super().__init__()

    self.z_dim = z_dim
    self.N = N

    self.proprio_encoder = ProprioEncoder(z_dim=self.z_dim)
    self.depth_encoder = EquivariantDepthEncoder(z_dim=self.z_dim, N=self.N, initialize=initialize)
    self.force_encoder = EquivariantForceEncoder(z_dim=self.z_dim, N=self.N, initialize=initialize)

    self.c4_act = gspaces.rot2dOnR2(self.N)
    self.proprio_repr = self.z_dim * [self.c4_act.trivial_repr]
    self.depth_repr = self.z_dim * [self.c4_act.regular_repr]
    self.force_repr = self.z_dim * [self.c4_act.regular_repr]

    self.in_type = enn.FieldType(self.c4_act, self.proprio_repr + self.depth_repr + self.force_repr)
    #self.in_type = enn.FieldType(self.c4_act, self.depth_repr + self.force_repr)
    self.out_type = enn.FieldType(self.c4_act, self.z_dim * [self.c4_act.regular_repr])
    self.conv = EquivariantBlock(
      self.in_type,
      self.out_type,
      kernel_size=1,
      stride=1,
      padding=0,
      initialize=initialize
    )

  def forward(self, depth, force, proprio):
    batch_size = depth.size(0)

    depth_feat = self.depth_encoder(depth)
    force_feat = self.force_encoder(force)
    proprio_feat = self.proprio_encoder(proprio).view(batch_size, self.z_dim, 1, 1)

    gate = (torch.mean(torch.abs(force.view(batch_size, -1)), dim=1) > 2e-2).float().cuda()
    gated_force_feat = force_feat.tensor.squeeze() * gate.view(batch_size, 1)
    force_feat = enn.GeometricTensor(gated_force_feat.view(batch_size, 512, 1, 1), enn.FieldType(self.c4_act, self.force_repr))

    feat = torch.cat((proprio_feat, depth_feat.tensor, force_feat.tensor), dim=1)
    #feat = torch.cat((depth_feat.tensor, force_feat.tensor), dim=1)
    feat = enn.GeometricTensor(feat, self.in_type)

    return self.conv(feat)
