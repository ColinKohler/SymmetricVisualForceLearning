import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn

from midichlorians.models.layers import CausalConvBlock1, EquivariantBlock, SelfAttention

class CausalConvBlock(nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.conv = nn.Sequential(
      CausalConv1d(in_dim, 16, kernel_size=2, stride=2, bias=True),
      nn.LeakyReLU(0.1, inplace=True),
      CausalConv1d(16, 32, kernel_size=2, stride=2, bias=True),
      nn.LeakyReLU(0.1, inplace=True),
      CausalConv1d(32, 64, kernel_size=2, stride=2, bias=True),
      nn.LeakyReLU(0.1, inplace=True),
      CausalConv1d(64, 128, kernel_size=2, stride=2, bias=True),
      nn.LeakyReLU(0.1, inplace=True),
      CausalConv1d(128, 128, kernel_size=2, stride=2, bias=True),
      nn.LeakyReLU(0.1, inplace=True),
      CausalConv1d(128, out_dim, kernel_size=2, stride=2, bias=True),
      nn.LeakyReLU(0.1, inplace=True),
    )

  def forward(self, x):
    return self.conv(x)


class EquivariantForceEncoder(nn.Module):
  '''
  '''
  def __init__(self, z_dim=64, N=8, initialize=True):
    super().__init__()

    n_head = 2
    d_model = 6
    d_k = 64
    d_v = 64
    dropout = 0.1
    self.self_attn = SelfAttention(n_head, d_model, d_k, d_v, dropout=dropout)

    self.fx_conv = CausalConvBlock(1, 16)
    self.fy_conv = CausalConvBlock(1, 16)
    self.fz_conv = CausalConvBlock(1, 16)
    self.mx_conv = CausalConvBlock(1, 16)
    self.my_conv = CausalConvBlock(1, 16)
    self.mz_conv = CausalConvBlock(1, 16)

    self.c4_act = gspaces.rot2dOnR2(N)

    self.equivariant_force_repr = 2 * 16 * [self.c4_act.irrep(1)]
    self.invariant_force_repr = 2 * 16 * [self.c4_act.trivial_repr]

    self.in_type = enn.FieldType(self.c4_act, self.equivariant_force_repr + self.invariant_force_repr)
    self.out_type = enn.FieldType(self.c4_act, z_dim  * [self.c4_act.regular_repr])
    self.conv = EquivariantBlock(self.in_type, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize, norm=True)

  def forward(self, x):
    batch_size =  x.size(0)

    x, attn = self.self_attn(
      x,
      x,
      x,
      mask=None,
    )
    x = torch.permute(x, (0,2,1))

    wfx = self.fx_conv(x[:,0].view(batch_size, 1, -1))
    wfy = self.fy_conv(x[:,1].view(batch_size, 1, -1))
    wfz = self.fz_conv(x[:,2].view(batch_size, 1, -1))
    wmx = self.mx_conv(x[:,3].view(batch_size, 1, -1))
    wmy = self.my_conv(x[:,4].view(batch_size, 1, -1))
    wmz = self.mz_conv(x[:,5].view(batch_size, 1, -1))

    equiv_force = torch.cat((wfx, wfy, wmx, wmy), dim=1).reshape(batch_size, -1, 1, 1)
    inv_force = torch.cat((wfz, wmz), dim=1).reshape(batch_size, -1, 1, 1)

    feat = torch.cat((equiv_force, inv_force), dim=1)
    feat = enn.GeometricTensor(feat, self.in_type)

    return self.conv(feat)
