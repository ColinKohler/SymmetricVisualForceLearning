import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from escnn import gspaces
from escnn import nn as enn

from midichlorians.models.equivariant_sac import EquivariantCritic, EquivariantGaussianPolicy
from midichlorians.models.encoders.equiv_sensor_fusion import EquivariantSensorFusion

class EquivariantFusionCritic(EquivariantCritic):
  '''
  Force equivariant critic model.
  '''
  def __init__(self, action_dim, z_dim=64, deterministic=True, initialize=True, N=8):
    super().__init__(action_dim, z_dim=z_dim, deterministic=deterministic, initialize=initialize, N=N)

    self.fusion_enc = EquivariantSensorFusion(z_dim=z_dim, deterministic=deterministic, initialize=initialize, N=N)

  def forward(self, obs, act):
    depth, force, proprio = obs
    batch_size = depth.size(0)

    if self.deterministic:
      z = self.fusion_enc(depth, force, proprio)
    else:
      z, mu_z, var_z, mu_prior, var_prior = self.fusion_enc(depth, force, proprio)
    z = z.view(batch_size, self.N * self.z_dim, 1, 1)

    dxy = act[:, 1:3].reshape(batch_size,  2, 1, 1)

    inv_act = torch.cat((act[:,0:1], act[:,3:]), dim=1)
    n_inv = inv_act.shape[1]
    inv_act = inv_act.reshape(batch_size, n_inv, 1, 1)

    cat = torch.cat((z, inv_act, dxy), dim=1)
    cat_geo = enn.GeometricTensor(cat, self.in_type)

    out_1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
    out_2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)

    if self.deterministic:
      return out_1, out_2, z
    else:
      return out_1, out_2, z, mu_z, var_z, mu_prior, var_prior

class EquivariantFusionGaussianPolicy(EquivariantGaussianPolicy):
  '''
  Equivariant actor model that uses a Normal distribution to sample actions.
  '''
  def __init__(self, action_dim, z_dim=64, deterministic=True, initialize=True, N=8):
    super().__init__(action_dim, z_dim=z_dim, deterministic=deterministic, initialize=initialize, N=N)

    self.fusion_enc = EquivariantSensorFusion(z_dim=z_dim, deterministic=deterministic, initialize=initialize, N=N)

  def forward(self, obs):
    depth, force, proprio = obs
    batch_size = depth.size(0)

    if self.deterministic:
      z = self.fusion_enc(depth, force, proprio)
    else:
      z, mu_z, var_z, mu_prior, var_prior = self.fusion_enc(depth, force, proprio)
    z = z.view(batch_size, self.N * self.z_dim, 1, 1)
    z_geo = enn.GeometricTensor(z, self.in_type)
    out = self.conv(z_geo).tensor.reshape(batch_size, -1)

    dxy = out[:, 0:2]
    inv_act = out[:, 2:self.action_dim]

    mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
    log_std = out[:, self.action_dim:]
    log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)

    if self.deterministic:
      return mean, log_std, z
    else:
      return mean, log_std, z, mu_z, var_z, mu_prior, var_prior
