import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn

from svfl.models.layers import EquivariantBlock, ConvBlock
from svfl.models.encoders.proprio_encoder import ProprioEncoder
from svfl.models.encoders.vision_encoder import VisionEncoder
from svfl.models.encoders.force_encoder import ForceEncoder
from svfl import torch_utils

class Latent(nn.Module):
  def __init__(self, vision_size=64, z_dim=64, N=8, encoder='fusion', equivariant=True, initialize=True):
    super().__init__()

    self.equivariant = equivariant
    self.encoder_dim = z_dim
    self.z_dim = z_dim
    self.N = N
    self.group = gspaces.rot2dOnR2(self.N)

    self.encoders = nn.ModuleDict()
    for e in encoder:
      if e == 'vision':
        self.encoders[e] = VisionEncoder(equivariant=equivariant, vision_size=vision_size, z_dim=self.encoder_dim, N=self.N, initialize=initialize)
      elif e == 'force':
        self.encoders[e] = ForceEncoder(equivariant=equivariant, z_dim=self.encoder_dim, N=self.N, initialize=initialize)
      elif e == 'proprio':
        self.encoders[e] = ProprioEncoder(equivariant=equivariant, z_dim=self.encoder_dim, N=self.N, initialize=initialize)
      else:
        raise ValueError('Invalid latent encoder specified: {}'.format(e))
    self.in_type = enn.FieldType(self.group, len(self.encoders) * self.encoder_dim * [self.group.regular_repr])
    self.out_type = enn.FieldType(self.group, self.z_dim * [self.group.regular_repr])

    self.layers = list()
    if self.equivariant:
      out_type = enn.FieldType(self.group, self.z_dim * [self.group.regular_repr])
      self.layers.append(EquivariantBlock(
        self.in_type,
        out_type,
        kernel_size=1,
        stride=1,
        padding=0,
        initialize=initialize
      ))

      in_type = out_type
      self.layers.append(EquivariantBlock(
        in_type,
        self.out_type,
        kernel_size=1,
        stride=1,
        padding=0,
        initialize=initialize
      ))
    else:
      self.layers.append(ConvBlock(
        len(self.encoders) * self.encoder_dim,
        self.z_dim,
        kernel_size=1,
        stride=1,
        padding=0
      ))

      self.layers.append(ConvBlock(
        self.z_dim,
        self.z_dim,
        kernel_size=1,
        stride=1,
        padding=0
      ))

    self.conv = nn.Sequential(*self.layers)

  def forward(self, obs):
    vision, force, proprio = obs
    batch_size = vision.size(0)

    feat = list()
    for et, encoder in self.encoders.items():
      if et == 'vision':
        feat.append(encoder(vision))
      elif et == 'force':
        feat.append(encoder(force))
      elif et == 'proprio':
        feat.append(encoder(proprio))

    if self.equivariant:
      feat = torch.cat([f.tensor for f in feat], dim=1)
      feat = enn.GeometricTensor(feat, self.in_type)
    else:
      feat = torch.cat(feat, dim=1)

    z = self.conv(feat)

    return z
