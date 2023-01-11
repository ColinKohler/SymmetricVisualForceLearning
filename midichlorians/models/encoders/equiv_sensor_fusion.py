import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn

from midichlorians.models.layers import EquivariantBlock
from midichlorians.models.encoders.proprio_encoder import ProprioEncoder
from midichlorians.models.encoders.equiv_depth_encoder import EquivariantDepthEncoder
from midichlorians.models.encoders.equiv_force_encoder import EquivariantForceEncoder
from midichlorians import torch_utils

class EquivariantSensorFusion(nn.Module):
  def __init__(self, z_dim=64, N=8, deterministic=True, initialize=True):
    super().__init__()

    self.z_dim = z_dim
    self.N = N
    self.deterministic = deterministic

    self.proprio_encoder = ProprioEncoder(z_dim=self.z_dim)
    self.depth_encoder = EquivariantDepthEncoder(z_dim=self.z_dim, N=self.N, initialize=initialize)
    self.force_encoder = EquivariantForceEncoder(z_dim=self.z_dim, N=self.N, initialize=initialize)

    self.c4_act = gspaces.rot2dOnR2(self.N)
    self.proprio_repr = 2 * self.z_dim * [self.c4_act.regular_repr]
    self.depth_repr = 2 * self.z_dim * [self.c4_act.regular_repr]
    self.force_repr = 2 * self.z_dim * [self.c4_act.regular_repr]

    self.z_prior_mu = nn.Parameter(
      torch.zeros(1, self.z_dim * N), requires_grad=False
    )
    self.z_prior_var = nn.Parameter(
      torch.ones(1, self.z_dim * N), requires_grad=False
    )
    self.z_prior = [self.z_prior_mu, self.z_prior_var]

    if self.deterministic:
      self.in_type = enn.FieldType(self.c4_act, self.proprio_repr + self.depth_repr + self.force_repr)
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

    proprio_feat = self.proprio_encoder(proprio)
    depth_feat = self.depth_encoder(depth)
    force_feat = self.force_encoder(force)

    gate = (torch.mean(torch.abs(force.view(batch_size, -1)), dim=1) > 2e-2).float().cuda()
    gated_force_feat = force_feat.tensor.squeeze() * gate.view(batch_size, 1)
    force_feat = enn.GeometricTensor(gated_force_feat.view(batch_size, 2 * self.N * self.z_dim, 1, 1), enn.FieldType(self.c4_act, self.force_repr))

    if self.deterministic:
      feat = torch.cat((proprio_feat, depth_feat.tensor, force_feat.tensor), dim=1)
      feat = enn.GeometricTensor(feat, self.in_type)
      z = self.conv(feat)

      return z
    else:
      # Encoder priors
      mu_prior, var_prior = self.z_prior

      # Duplicate priors for each sample
      mu_prior_resized = torch_utils.duplicate(mu_prior, batch_size).unsqueeze(2)
      var_prior_resized = torch_utils.duplicate(var_prior, batch_size).unsqueeze(2)

      #mu_z_proprio, var_z_proprio = torch_utils.gaussianParameters(proprio_feat.squeeze(-1), dim=1)
      mu_z_depth, var_z_depth = torch_utils.gaussianParameters(depth_feat.tensor.squeeze(-1), dim=1)
      mu_z_force, var_z_force = torch_utils.gaussianParameters(force_feat.tensor.squeeze(-1), dim=1)

      # Tile distribution parameters
      mu_vect = torch.cat([mu_z_depth, mu_z_force, mu_prior_resized], dim=2)
      var_vect = torch.cat([var_z_depth, var_z_force, var_prior_resized], dim=2)

      # Sample gaussian to get latent encoding
      mu_z, var_z = torch_utils.productOfExperts(mu_vect, var_vect)
      z = torch_utils.sampleGaussian(mu_z, var_z)

      return z, mu_z, var_z, mu_prior, var_prior
