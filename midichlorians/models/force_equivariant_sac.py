import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from escnn import gspaces
from escnn import nn as enn

from midichlorians.models.equivariant_sac import EquivariantBlock, EquivariantDepthEncoder, EquivariantCritic, EquivariantGaussianPolicy

class CausalConv1d(nn.Conv1d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
    self.__padding = (kernel_size - 1) * dilation

    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=self.__padding, dilation=dilation, bias=bias)

  def forward(self, x):
    res = super().forward(x)
    if self.__padding != 0:
      return res[:, :, :-self.__padding]
    return res

class CausalConvBlock(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Sequential(
      CausalConv1d(1, 4, kernel_size=2, stride=2),
      nn.LeakyReLU(0.1, inplace=True),
      CausalConv1d(4, 8, kernel_size=2, stride=2),
      nn.LeakyReLU(0.1, inplace=True),
      CausalConv1d(8, 16, kernel_size=2, stride=2),
      nn.LeakyReLU(0.1, inplace=True),
      CausalConv1d(16, 32, kernel_size=2, stride=2),
      nn.LeakyReLU(0.1, inplace=True),
      CausalConv1d(32, 32, kernel_size=2, stride=2),
      nn.LeakyReLU(0.1, inplace=True),
      CausalConv1d(32, 16, kernel_size=2, stride=2),
      nn.LeakyReLU(0.1, inplace=True),
    )

  def forward(self, x):
    return self.conv(x)

class ForceEncoder(nn.Module):
  '''

  '''
  def __init__(self, n_out):
    super().__init__()
    self.fx_conv = CausalConvBlock()
    self.fy_conv = CausalConvBlock()
    self.fz_conv = CausalConvBlock()
    self.mx_conv = CausalConvBlock()
    self.my_conv = CausalConvBlock()
    self.mz_conv = CausalConvBlock()

  def forward(self, x):
    batch_size =  x.size(0)

    fx_feat = self.fx_conv(x[:,0].view(batch_size, 1, -1))
    fy_feat = self.fy_conv(x[:,1].view(batch_size, 1, -1))
    fz_feat = self.fz_conv(x[:,2].view(batch_size, 1, -1))
    mx_feat = self.mx_conv(x[:,3].view(batch_size, 1, -1))
    my_feat = self.my_conv(x[:,4].view(batch_size, 1, -1))
    mz_feat = self.mz_conv(x[:,5].view(batch_size, 1, -1))

    # Gate output depending on the force signal
    gate = torch.ones(batch_size).cuda()
    gate *= torch.mean(torch.abs(x).reshape(batch_size, -1), dim=1) > 1e-2
    gate = gate.view(batch_size, 1, 1)

    gated_fx_feat = gate * fx_feat
    gated_fy_feat = gate * fy_feat
    gated_fz_feat = gate * fz_feat
    gated_mx_feat = gate * mx_feat
    gated_my_feat = gate * my_feat
    gated_mz_feat = gate * mz_feat

    return gated_fx_feat, gated_fy_feat, gated_fz_feat, gated_mx_feat, gated_my_feat, gated_mz_feat

class EquivariantEncoder(nn.Module):
  '''
  '''
  def __init__(self, depth_channels, n_out=64, initialize=True, N=8):
    super().__init__()

    self.wrist_force_enc = ForceEncoder(n_out)
    if False:
      self.finger_1_force_enc = ForceEncoder(n_out)
      self.finger_2_force_enc = ForceEncoder(n_out)
    self.depth_enc = EquivariantDepthEncoder(depth_channels, n_out=n_out, initialize=initialize, N=N)
    self.c4_act = gspaces.rot2dOnR2(N)

    self.depth_repr = n_out * [self.c4_act.regular_repr]
    if False:
      self.equivariant_force_repr = 2 * 3 * 16 * [self.c4_act.irrep(1)]
      self.invariant_force_repr = 2 * 3 * 16 * [self.c4_act.trivial_repr]
    else:
      self.equivariant_force_repr = 2 * 16 * [self.c4_act.irrep(1)]
      self.invariant_force_repr = 2 * 16 * [self.c4_act.trivial_repr]

    self.in_type = enn.FieldType(self.c4_act, self.depth_repr + self.equivariant_force_repr + self.invariant_force_repr)
    self.out_type = enn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr])
    self.conv = EquivariantBlock(self.in_type, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize)

  def forward(self, depth, force):
    batch_size = force.size(0)
    force = torch.permute(force, (0,2,1))

    wrist_force_feat = self.wrist_force_enc(force[:,:6])
    wfx, wfy, wmx, wmy = wrist_force_feat[0], wrist_force_feat[1], wrist_force_feat[3], wrist_force_feat[4]
    wfz, wmz = wrist_force_feat[2], wrist_force_feat[5]
    if False:
      finger_1_force_feat = self.finger_1_force_enc(force[:,6:12])
      finger_2_force_feat = self.finger_2_force_enc(force[:,12:])
      f1fx, f1fy, f1mx, f1my = finger_1_force_feat[0], finger_1_force_feat[1], finger_1_force_feat[3], finger_1_force_feat[4]
      f2fx, f2fy, f2mx, f2my = finger_2_force_feat[0], finger_2_force_feat[1], finger_2_force_feat[3], finger_2_force_feat[4]
      equiv_force = torch.cat((wfx, wfy, wmx, wmy, f1fx, f1fy, f1mx, f1my, f2fx, f2fy, f2mx, f2my,), dim=1).reshape(batch_size, -1, 1, 1)

      f1fz, f1mz = finger_1_force_feat[2], finger_1_force_feat[5]
      f2fz, f2mz = finger_2_force_feat[2], finger_2_force_feat[5]
      inv_force = torch.cat((wfz, wmz, f1fz, f1mz, f2fz, f2mz), dim=1).reshape(batch_size, -1, 1, 1)
    else:
      equiv_force = torch.cat((wfx, wfy, wmx, wmy), dim=1).reshape(batch_size, -1, 1, 1)
      inv_force = torch.cat((wfz, wmz), dim=1).reshape(batch_size, -1, 1, 1)

    depth_geo = enn.GeometricTensor(depth, self.depth_enc.in_type)
    depth_feat = self.depth_enc(depth_geo)

    feat = torch.cat((depth_feat.tensor, equiv_force, inv_force), dim=1)
    feat = enn.GeometricTensor(feat, self.in_type)

    return self.conv(feat)

class ForceEquivariantCritic(EquivariantCritic):
  '''
  Force equivariant critic model.
  '''
  def __init__(self, depth_channels, action_dim, n_out=64, initialize=True, N=8):
    super().__init__(depth_channels, action_dim, n_out=n_out, initialize=initialize, N=N)

    self.enc = EquivariantEncoder(depth_channels, n_out=n_out, initialize=initialize, N=N)

  def forward(self, obs, act):
    depth, force = obs
    batch_size = depth.size(0)

    feat = self.enc(depth, force)

    dxy = act[:, 1:3].reshape(batch_size,  2, 1, 1)

    inv_act = torch.cat((act[:,0:1], act[:,3:]), dim=1)
    n_inv = inv_act.shape[1]
    inv_act = inv_act.reshape(batch_size, n_inv, 1, 1)

    cat = torch.cat((feat.tensor, inv_act, dxy), dim=1)
    cat_geo = enn.GeometricTensor(cat, self.in_type)

    out_1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
    out_2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)

    return out_1, out_2

class ForceEquivariantGaussianPolicy(EquivariantGaussianPolicy):
  '''
  Equivariant actor model that uses a Normal distribution to sample actions.
  '''
  def __init__(self, depth_channels, action_dim, n_out=64, initialize=True, N=8):
    super().__init__(depth_channels, action_dim, n_out=n_out, initialize=initialize, N=N)

    self.enc = EquivariantEncoder(depth_channels, n_out=n_out, initialize=initialize, N=N)

  def forward(self, obs):
    depth, force = obs
    batch_size = depth.size(0)

    feat = self.enc(depth, force)
    out = self.conv(feat).tensor.reshape(batch_size, -1)

    dxy = out[:, 0:2]
    inv_act = out[:, 2:self.action_dim]

    mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
    log_std = out[:, self.action_dim:]
    log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)

    return mean, log_std
