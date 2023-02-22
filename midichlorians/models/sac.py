import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from escnn import gspaces
from escnn import nn as enn

from midichlorians.models.latent import Latent
from midichlorians.models.layers import EquivariantBlock, ResnetBlock

class Critic(nn.Module):
  '''
  Equivariant critic model.
  '''
  def __init__(self, vision_size, action_dim, equivariant=True, z_dim=64, encoder='fusion', initialize=True, N=8):
    super().__init__()

    self.equivariant = equivariant
    self.z_dim = z_dim
    self.action_dim = action_dim
    self.N = N

    self.c4_act = gspaces.rot2dOnR2(self.N)
    self.n_rho1 = 1
    self.z_repr = self.z_dim * [self.c4_act.regular_repr]
    self.invariant_action_repr = (self.action_dim - 2) * [self.c4_act.trivial_repr]
    self.equivariant_action_repr = self.n_rho1 * [self.c4_act.irrep(1)]

    self.in_type = enn.FieldType(self.c4_act, self.z_repr + self.invariant_action_repr + self.equivariant_action_repr)
    self.inner_type = enn.FieldType(self.c4_act, self.z_dim * [self.c4_act.regular_repr])
    self.inner_type_2 = enn.FieldType(self.c4_act, self.z_dim * [self.c4_act.trivial_repr])
    self.out_type = enn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr])

    self.encoder = Latent(equivariant=equivariant, vision_size=vision_size, encoder=encoder, initialize=initialize)

    if self.equivariant:
      self.critic_1 = nn.Sequential(
        EquivariantBlock(self.in_type, self.inner_type, kernel_size=1, stride=1, padding=0, initialize=initialize),
        enn.GroupPooling(self.inner_type),
        EquivariantBlock(self.inner_type_2, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize, act=False)
      )

      self.critic_2 = nn.Sequential(
        EquivariantBlock(self.in_type, self.inner_type, kernel_size=1, stride=1, padding=0, initialize=initialize),
        enn.GroupPooling(self.inner_type),
        EquivariantBlock(self.inner_type_2, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize, act=False)
      )
    else:
      self.critic_1 = nn.Sequential(
        ResnetBlock(self.z_dim + self.action_dim, 1, kernel_size=1, stride=1, padding=0, act=False)
      )

      self.critic_2 = nn.Sequential(
        ResnetBlock(self.z_dim + self.action_dim, 1, kernel_size=1, stride=1, padding=0, act=False)
      )


  def forward(self, obs, act):
    batch_size = obs[0].size(0)
    z = self.encoder(obs)

    dxy = act[:, 1:3].reshape(batch_size,  2, 1, 1)

    inv_act = torch.cat((act[:,0:1], act[:,3:]), dim=1)
    n_inv = inv_act.shape[1]
    inv_act = inv_act.reshape(batch_size, n_inv, 1, 1)

    if self.equivariant:
      cat = torch.cat((z.tensor, inv_act, dxy), dim=1)
      cat_geo = enn.GeometricTensor(cat, self.in_type)

      out_1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
      out_2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
    else:
      cat = torch.cat((z, inv_act, dxy), dim=1)
      out_1 = self.critic_1(cat).reshape(batch_size, 1)
      out_2 = self.critic_2(cat).reshape(batch_size, 1)

    return out_1, out_2

class GaussianPolicy(nn.Module):
  '''
  Equivariant actor model that uses a Normal distribution to sample actions.
  '''
  def __init__(self, vision_size, action_dim, equivariant=True, z_dim=64, encoder='fusion', initialize=True, N=8):
    super().__init__()
    self.log_sig_min = -20
    self.log_sig_max = 2
    self.eps = 1e-6

    self.equivariant = equivariant
    self.action_dim = action_dim
    self.z_dim = z_dim
    self.initialize = initialize
    self.N = N

    self.c4_act = gspaces.rot2dOnR2(N)
    self.n_rho1 = 1
    self.z_repr = self.z_dim * [self.c4_act.regular_repr]
    self.invariant_action_repr = (self.action_dim * 2 - 2) * [self.c4_act.trivial_repr]
    self.equivariant_action_repr = self.n_rho1 * [self.c4_act.irrep(1)]

    self.encoder = Latent(equivariant=equivariant, vision_size=vision_size, encoder=encoder, initialize=initialize)

    self.layers = list()

    if self.equivariant:
      self.in_type = enn.FieldType(self.c4_act, self.z_repr)
      out_type = enn.FieldType(self.c4_act, self.z_dim // 2 * [self.c4_act.regular_repr])
      self.layers.append(EquivariantBlock(self.in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))

      in_type = out_type
      out_type = enn.FieldType(self.c4_act, self.z_dim // 4 * [self.c4_act.regular_repr])
      self.layers.append(EquivariantBlock(in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize))

      in_type = out_type
      self.out_type = enn.FieldType(self.c4_act, self.equivariant_action_repr + self.invariant_action_repr)
      self.layers.append(EquivariantBlock(in_type, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize, act=False))
    else:
      self.layers.append(ResnetBlock(self.z_dim, self.z_dim // 2, kernel_size=1, stride=1, padding=0))
      self.layers.append(ResnetBlock(self.z_dim // 2, self.z_dim // 4, kernel_size=1, stride=1, padding=0))
      self.layers.append(ResnetBlock(self.z_dim // 4, self.action_dim*2, kernel_size=1, stride=1, padding=0, act=False))

    self.conv = nn.Sequential(*self.layers)

  def forward(self, obs):
    batch_size = obs[0].size(0)
    z  = self.encoder(obs)

    if self.equivariant:
      out = self.conv(z).tensor.reshape(batch_size, -1)
    else:
      out = self.conv(z).reshape(batch_size, -1)

    dxy = out[:, 0:2]
    inv_act = out[:, 2:self.action_dim]

    mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
    log_std = out[:, self.action_dim:]
    log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)

    return mean, log_std

  def sample(self, obs):
    '''
    Sample an action from a Normal distribution generated by the model.
    '''
    mean, log_std = self.forward(obs)
    std = log_std.exp()

    normal = Normal(mean, std)
    x_t = normal.rsample()
    y_t = torch.tanh(x_t)
    action = y_t

    log_prob = normal.log_prob(x_t)
    log_prob -= torch.log((1 - y_t.pow(2)) + self.eps)
    log_prob = log_prob.sum(1, keepdim=True)
    mean = torch.tanh(mean)

    return action, log_prob, mean
