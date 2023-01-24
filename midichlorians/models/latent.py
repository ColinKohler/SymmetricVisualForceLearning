import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn

from midichlorians.models.layers import EquivariantBlock
from midichlorians.models.encoders.proprio_encoder import ProprioEncoder
from midichlorians.models.encoders.depth_encoder import DepthEncoder
from midichlorians.models.encoders.force_encoder_2 import ForceEncoder
from midichlorians import torch_utils

class Latent(nn.Module):
  def __init__(self, z_dim=64, N=8, encoder='fusion', deterministic=True, initialize=True):
    super().__init__()

    # Double parameters for stochastic model (mean and variance)
    if deterministic:
      self.encoder_dim = z_dim
      self.z_dim = z_dim
    else:
      self.encoder_dim = 2 * z_dim
      self.z_dim = z_dim

      self.z_prior_mu = nn.Parameter(
        torch.zeros(1, self.z_dim * N), requires_grad=False
      )
      self.z_prior_var = nn.Parameter(
        torch.ones(1, self.z_dim * N), requires_grad=False
      )
      self.z_prior = [self.z_prior_mu, self.z_prior_var]

    self.N = N
    self.deterministic = deterministic
    self.c4_act = gspaces.rot2dOnR2(self.N)

    self.encoders = nn.ModuleDict()
    for e in encoder:
      if e == 'depth':
        self.encoders[e] = DepthEncoder(z_dim=self.encoder_dim, N=self.N, initialize=initialize)
      elif e == 'force':
        self.encoders[e] = ForceEncoder(z_dim=self.encoder_dim, N=self.N, initialize=initialize)
      elif e == 'proprio':
        self.encoders[e] = ProprioEncoder(z_dim=self.encoder_dim, N=self.N, initialize=initialize)
      else:
        raise ValueError('Invalid latent encoder specified.')
    self.in_type = enn.FieldType(self.c4_act, len(self.encoders) * self.encoder_dim * [self.c4_act.regular_repr])
    self.out_type = enn.FieldType(self.c4_act, self.z_dim * [self.c4_act.regular_repr])

    if self.deterministic:
      self.layers = list()

      out_type = enn.FieldType(self.c4_act, self.z_dim * [self.c4_act.regular_repr])
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
      self.conv = nn.Sequential(*self.layers)

  def forward(self, obs):
    depth, force, proprio = obs
    batch_size = depth.size(0)

    feat = list()
    for et, encoder in self.encoders.items():
      if et == 'depth':
        feat.append(encoder(depth))
      elif et == 'force':
        feat.append(encoder(force))
      elif et == 'proprio':
        feat.append(encoder(proprio))

    if self.deterministic:
      feat = torch.cat([f.tensor for f in feat], dim=1)
      feat = enn.GeometricTensor(feat, self.in_type)
      z = self.conv(feat)

      return z
    else:
      # Encoder priors
      mu_prior, var_prior = self.z_prior

      # Duplicate priors for each sample
      mu_prior_resized = torch_utils.duplicate(mu_prior, batch_size).unsqueeze(2)
      var_prior_resized = torch_utils.duplicate(var_prior, batch_size).unsqueeze(2)

      mu_z_proprio, var_z_proprio = torch_utils.gaussianParameters(proprio_feat.tensor.squeeze(-1), dim=1)
      mu_z_depth, var_z_depth = torch_utils.gaussianParameters(depth_feat.tensor.squeeze(-1), dim=1)
      mu_z_force, var_z_force = torch_utils.gaussianParameters(force_feat.tensor.squeeze(-1), dim=1)

      # Tile distribution parameters
      mu_vect = torch.cat([mu_z_proprio, mu_z_depth, mu_z_force, mu_prior_resized], dim=2)
      var_vect = torch.cat([var_z_proprio, var_z_depth, var_z_force, var_prior_resized], dim=2)

      # Sample gaussian to get latent encoding
      mu_z, var_z = torch_utils.productOfExperts(mu_vect, var_vect)
      z = torch_utils.sampleGaussian(mu_z, var_z)
      z = enn.GeometricTensor(z.view(batch_size, -1, 1, 1), self.out_type)

      return z, mu_z, var_z, mu_prior, var_prior
