import torch
import torch.nn as nn

from escnn import group
from escnn import gspaces
from escnn import nn as enn

from svfl.models.layers import EquivariantBlock, ConvBlock
from svfl.models.encoders.pose_encoder import PoseEncoder
from svfl.models.encoders.proprio_encoder import ProprioEncoder
from svfl.models.encoders.vision_encoder import VisionEncoder
from svfl.models.encoders.force_encoder import ForceEncoder
from svfl import torch_utils

class Latent(nn.Module):
  def __init__(self, vision_size=64, z_dim=8, N=8, encoder='fusion', equivariant=True, initialize=True):
    super().__init__()

    self.equivariant = equivariant
    self.encoder_dim = z_dim
    self.z_dim = z_dim
    self.N = N

    self.G = group.so2_group()
    self.gspace = gspaces.no_base_space(self.G)

    #self.encoders[e] = VisionEncoder(equivariant=equivariant, vision_size=vision_size, z_dim=self.encoder_dim, N=self.N, initialize=initialize)
    self.pose_encoder = PoseEncoder(z_dim=self.encoder_dim, N=self.N, initialize=initialize)
    #self.encoders[e] = ForceEncoder(equivariant=equivariant, z_dim=self.encoder_dim, N=self.N, initialize=initialize)
    self.proprio_encoder = ProprioEncoder(z_dim=self.encoder_dim, N=self.N, initialize=initialize)

    self.in_type = self.pose_encoder.out_type + self.proprio_encoder.out_type

    act_1 = enn.FourierELU(
      self.gspace,
      channels=self.z_dim,
      irreps=self.G.bl_regular_representation(L=3).irreps,
      inplace=True,
      type='regular',
      N=16
    )
    self.block_1 = enn.SequentialModule(
      enn.Linear(self.in_type, act_1.in_type),
      act_1
    )
    self.out_type = self.block_1.out_type

  def forward(self, obs):
    pose, force, proprio = obs

    pose_feat = self.pose_encoder(self.pose_encoder.in_type(pose))
    proprio_feat = self.proprio_encoder(self.proprio_encoder.in_type(proprio))

    z = self.in_type(torch.cat([pose_feat.tensor, proprio_feat.tensor], dim=1))
    z = self.block_1(z)

    return z
