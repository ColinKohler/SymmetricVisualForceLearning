import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn

from midichlorians.models.layers import EquivariantBlock

class AttentionBlock(nn.Module):
  def __init__(self):
    super().__init__()

    self.attn  = nn.MultiheadAttention(64, 8, kdim=64, vdim=64, batch_first=True)

  def forward(self, x):
    residual = x
    x, attn = self.attn(
      x,
      x,
      x,
    )
    x += residual

    return x, attn

class ForceEncoder(nn.Module):
  '''
  '''
  def __init__(self, equivariant=False, z_dim=64, N=8, initialize=True):
    super().__init__()

    self.embed = nn.Linear(6,64)
    self.attention1 = AttentionBlock()
    self.attention2 = AttentionBlock()

    self.c4_act = gspaces.rot2dOnR2(N)
    self.in_type = enn.FieldType(self.c4_act, 64 * 64 * [self.c4_act.trivial_repr])
    self.out_type = enn.FieldType(self.c4_act, z_dim  * [self.c4_act.regular_repr])
    self.conv = EquivariantBlock(self.in_type, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize)

  def forward(self, x):
    batch_size =  x.size(0)

    x_embed = self.embed(x.view(batch_size * 64, 6)).view(batch_size, 64, 64)
    x_, _ = self.attention1(x_embed)
    x_, _ = self.attention2(x_)

    #import matplotlib.pyplot as plt
    #fig, ax = plt.subplots(nrows=2, ncols=1)
    #ax[0].plot(x[0,:,0].cpu(), label='Fx')
    #ax[0].plot(x[0,:,1].cpu(), label='Fy')
    #ax[0].plot(x[0,:,2].cpu(), label='Fz')
    #ax[0].plot(x[0,:,3].cpu(), label='Mx')
    #ax[0].plot(x[0,:,4].cpu(), label='My')
    #ax[0].plot(x[0,:,5].cpu(), label='Mz')
    #ax[1].imshow(attn.cpu().squeeze())
    #plt.show()

    x_geo = enn.GeometricTensor(x_.reshape(batch_size, -1, 1, 1), self.in_type)

    return self.conv(x_geo)
