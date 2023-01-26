import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import group
from escnn import nn as enn

from midichlorians.models.layers import EquivariantBlock

class ScaledDotProductAttention(nn.Module):
  def __init__(self, temperature):
    super().__init__()
    self.temperature = temperature

  def forward(self, q, k, v):
    attn = torch.matmul(q / self.temperature, k.transpose(2,3))

    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)

    return output, attn

class MultiheadAttention(nn.Module):
  def __init__(self, n_head, d_model, d_k, d_v, initialize=True):
    super().__init__()
    self.n_head = n_head
    self.d_k = d_k
    self.d_v = d_v

    self.c4_act = gspaces.rot2dOnR2(8)

    self.w_in_type = enn.FieldType(self.c4_act, d_model * [self.c4_act.regular_repr])
    w_out_type = enn.FieldType(self.c4_act, n_head * d_k * [self.c4_act.regular_repr])
    self.w_qs = EquivariantBlock(self.w_in_type, w_out_type, kernel_size=1, stride=1, padding=0, act=False, initialize=initialize)
    self.w_ks = EquivariantBlock(self.w_in_type, w_out_type, kernel_size=1, stride=1, padding=0, act=False, initialize=initialize)
    self.w_vs = EquivariantBlock(self.w_in_type, w_out_type, kernel_size=1, stride=1, padding=0, act=False, initialize=initialize)
    self.pool = enn.GroupPooling(w_out_type)

    self.fc_in_type = enn.FieldType(self.c4_act, n_head * d_v * [self.c4_act.regular_repr])
    self.out_type = enn.FieldType(self.c4_act, d_model * [self.c4_act.regular_repr])
    self.fc = EquivariantBlock(self.fc_in_type, self.out_type, kernel_size=1, stride=1, padding=0, act=False, initialize=initialize)
    self.out_type_2 = enn.FieldType(self.c4_act, 64 * d_model * [self.c4_act.regular_repr])

    self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

  def forward(self, q, k, v, sz_b):
    d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
    len_q, len_k, len_v = 64, 64, 64

    residual = q

    # Pass through the pre-attention projection: b x lq x (n*dv*c)
    # Seperate different heads: b x lq x n x dvc
    q = self.w_qs(q).tensor.view(sz_b, len_q, n_head, d_k*8)
    k = self.w_ks(k).tensor.view(sz_b, len_k, n_head, d_k*8)
    v = self.w_vs(v).tensor.view(sz_b, len_v, n_head, d_v*8)

    # Transpose for attention dot product: b x n x lq x dv
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    q, attn = self.attention(q, k, v)

    # Transpose to move the head dimension back: b x lq x n x dv
    # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
    q = q.transpose(1,2).contiguous().view(sz_b * 64, -1, 1, 1)
    q_geo = enn.GeometricTensor(q, self.fc_in_type)
    q = self.fc(q_geo)
    q += residual
    q_geo = enn.GeometricTensor(q.tensor.view(sz_b, -1, 1, 1), self.out_type_2)

    return q_geo, attn

class ForceEncoder(nn.Module):
  '''
  '''
  def __init__(self, z_dim=64, N=8, initialize=True):
    super().__init__()

    self.c4_act = gspaces.rot2dOnR2(N)

    self.in_type = enn.FieldType(
      self.c4_act,
      [self.c4_act.irrep(1)] + [self.c4_act.trivial_repr] + [self.c4_act.irrep(1)] + [self.c4_act.trivial_repr]
    )
    out_type = enn.FieldType(self.c4_act, 8 * [self.c4_act.regular_repr])
    self.embed = EquivariantBlock(self.in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize)
    self.attn  = MultiheadAttention(n_head=2, d_model=8, d_k=8, d_v=8, initialize=initialize)

    self.c4_act = gspaces.rot2dOnR2(N)
    self.out_type = enn.FieldType(self.c4_act, z_dim * [self.c4_act.regular_repr])
    self.conv = EquivariantBlock(self.attn.out_type_2, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize)

  def forward(self, x):
    batch_size =  x.size(0)

    x_geo = enn.GeometricTensor(x.view(batch_size * 64, 6, 1, 1), self.in_type)
    x_embed = self.embed(x_geo)
    x_attend, attn = self.attn(
      x_embed,
      x_embed,
      x_embed,
      batch_size
    )

    return self.conv(x_attend)
