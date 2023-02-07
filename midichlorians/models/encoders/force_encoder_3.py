import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import group
from escnn import nn as enn

from midichlorians.models.layers import EquivariantBlock

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len):
    super().__init__()
    self.d_model = d_model

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, 1, d_model)
    pe[0, :, 0, 0::2] = torch.sin(position * div_term)
    pe[0, :, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(1)]
    return x

class ScaledDotProductAttention(nn.Module):
  def __init__(self, temperature):
    super().__init__()
    self.temperature = temperature

  def forward(self, q, k, v):
    attn = torch.matmul(q / self.temperature, k.transpose(3,4))

    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)

    return output, attn

class MultiheadAttention(nn.Module):
  def __init__(self, n_head, d_model, d_k, d_v, initialize=True):
    super().__init__()
    self.n_head = n_head
    self.c = 8
    self.d_model = d_model
    self.d_k = d_k
    self.d_v = d_v

    self.c4_act = gspaces.rot2dOnR2(self.c)

    self.w_in_type = enn.FieldType(self.c4_act, d_model * [self.c4_act.regular_repr])
    w_out_type = enn.FieldType(self.c4_act, n_head * d_k * [self.c4_act.regular_repr])
    self.w_qs = EquivariantBlock(self.w_in_type, w_out_type, kernel_size=1, stride=1, padding=0, act=False, initialize=initialize)
    self.w_ks = EquivariantBlock(self.w_in_type, w_out_type, kernel_size=1, stride=1, padding=0, act=False, initialize=initialize)
    self.w_vs = EquivariantBlock(self.w_in_type, w_out_type, kernel_size=1, stride=1, padding=0, act=False, initialize=initialize)

    self.fc_in_type = enn.FieldType(self.c4_act, n_head * d_v * [self.c4_act.regular_repr])
    self.out_type = enn.FieldType(self.c4_act, d_model * [self.c4_act.regular_repr])
    self.fc = EquivariantBlock(self.fc_in_type, self.out_type, kernel_size=1, stride=1, padding=0, act=False, initialize=initialize)

    self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

  def forward(self, q, k, v, sz_b):
    b = q.size(0)
    d_model, d_k, d_v, n_head = self.d_model, self.d_k, self.d_v, self.n_head
    len_q, len_k, len_v = 64, 64, 64

    q_geo = enn.GeometricTensor(q.view(b * len_q, d_model*self.c, 1, 1), self.w_in_type)
    k_geo = enn.GeometricTensor(k.view(b * len_k, d_model*self.c, 1, 1), self.w_in_type)
    v_geo = enn.GeometricTensor(v.view(b * len_k, d_model*self.c, 1, 1), self.w_in_type)
    residual = q_geo

    # Pass through the pre-attention projection: b x lq x (n*dv*c)
    # Seperate different heads: b x lq x n x c x dv
    q = self.w_qs(q_geo).tensor.view(b, len_q, n_head, self.c, d_k)
    k = self.w_ks(k_geo).tensor.view(b, len_k, n_head, self.c, d_k)
    v = self.w_vs(v_geo).tensor.view(b, len_v, n_head, self.c, d_v)

    # Transpose for attention dot product: b x n x c x lq x dv
    q, k, v = q.permute(0,2,3,1,4), k.permute(0,2,3,1,4), v.permute(0,2,3,1,4)
    q, attn = self.attention(q, k, v)

    # Transpose to move the head dimension back: b x lq x n x dv x c
    # Combine the last two dimensions to concatenate all the heads together: b*lq x (n*dv*c)
    q = q.transpose(1,2).contiguous().view(b * len_q, n_head * d_model * self.c, 1, 1)
    q_geo = enn.GeometricTensor(q, self.fc_in_type)
    q_out = self.fc(q_geo)
    q_out += residual

    return q_out.tensor.view(b, len_q, d_model * self.c), attn

class ForceEncoder(nn.Module):
  '''
  '''
  def __init__(self, z_dim=64, N=8, initialize=True):
    super().__init__()

    self.c4_act = gspaces.rot2dOnR2(N)
    self.N = N
    self.d_model = 8
    self.seq_len = 64

    self.in_type = enn.FieldType(
      self.c4_act,
      [self.c4_act.irrep(1)] + [self.c4_act.trivial_repr] + [self.c4_act.irrep(1)] + [self.c4_act.trivial_repr]
    )
    out_type = enn.FieldType(self.c4_act, self.d_model * [self.c4_act.regular_repr])
    self.embed = EquivariantBlock(self.in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize)
    self.pos_encoder = PositionalEncoding(self.d_model, self.seq_len)
    self.attn  = MultiheadAttention(n_head=8, d_model=self.d_model, d_k=self.d_model, d_v=self.d_model, initialize=initialize)

    self.c4_act = gspaces.rot2dOnR2(N)
    self.fc_in_type = enn.FieldType(self.c4_act, self.seq_len * self.d_model * [self.c4_act.regular_repr])
    self.out_type = enn.FieldType(self.c4_act, z_dim * [self.c4_act.regular_repr])
    self.conv = EquivariantBlock(self.fc_in_type, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize)

  def forward(self, x):
    batch_size = x.size(0)
    seq_l = x.size(1)

    # Combine batch and seq len for embedding: b*lq x 6 x 1 x 1
    # Seperate componets of eqeuivariant tensor: b x lq x c x d
    # Want to add the same position embedding to each element in the regular represenation
    x_geo = enn.GeometricTensor(x.view(batch_size * seq_l, 6, 1, 1), self.in_type)
    x_embed = self.embed(x_geo).tensor.view(batch_size, seq_l, self.N, self.d_model)
    x_embed = self.pos_encoder(x_embed)

    # Apply attention and flatten in geo tensor: b x sq*c*d
    x_attend, attn = self.attn(
      x_embed,
      x_embed,
      x_embed,
      batch_size
    )
    x_attend = enn.GeometricTensor(x_attend.view(batch_size, -1, 1, 1), self.fc_in_type)

    return self.conv(x_attend)
