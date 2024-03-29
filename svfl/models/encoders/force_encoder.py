import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import group
from escnn import nn as enn

from svfl.models.layers import EquivariantBlock, ConvBlock, Norm

class ForceEncoder(nn.Module):
  def __init__(self, equivariant=False, z_dim=64, initialize=True, N=8):
    super().__init__()
    if equivariant:
      self.encoder = EquivForceEncoder(z_dim=z_dim, initialize=initialize, N=N)
    else:
      self.encoder = CnnForceEncoder(z_dim=z_dim)

  def forward(self, x):
    return self.encoder(x)

class ScaledDotProductAttention(nn.Module):
  def __init__(self, temperature):
    super().__init__()
    self.temperature = temperature

  def forward(self, q, k, v):
    attn = torch.matmul(q / self.temperature, k.transpose(-2,-1))

    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)

    return output, attn

class EquivMultiHeadAttention(nn.Module):
  def __init__(self, n_head, d_model, d_k, d_v, initialize=True):
    super().__init__()
    self.n_head = n_head
    self.c = 8
    self.d_model = d_model
    self.d_k = d_k
    self.d_v = d_v

    self.group = gspaces.rot2dOnR2(self.c)

    self.w_in_type = enn.FieldType(self.group, d_model * [self.group.regular_repr])
    w_out_type = enn.FieldType(self.group, n_head * d_k * [self.group.regular_repr])
    self.w_qs = EquivariantBlock(self.w_in_type, w_out_type, kernel_size=1, stride=1, padding=0, act=False, initialize=initialize)
    self.w_ks = EquivariantBlock(self.w_in_type, w_out_type, kernel_size=1, stride=1, padding=0, act=False, initialize=initialize)
    self.w_vs = EquivariantBlock(self.w_in_type, w_out_type, kernel_size=1, stride=1, padding=0, act=False, initialize=initialize)

    self.fc_in_type = enn.FieldType(self.group, n_head * d_v * [self.group.regular_repr])
    self.out_type = enn.FieldType(self.group, d_model * [self.group.regular_repr])
    self.fc = EquivariantBlock(self.fc_in_type, self.out_type, kernel_size=1, stride=1, padding=0, act=False, initialize=initialize)

    self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

  def forward(self, q, k, v):
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
    q = q.permute(0,3,1,4,2).contiguous().view(b * len_q, n_head * d_model * self.c, 1, 1)
    q_geo = enn.GeometricTensor(q, self.fc_in_type)
    q_out = self.fc(q_geo)

    return q_out.tensor.view(b, len_q, d_model * self.c), attn

class EquivForceEncoder(nn.Module):
  '''
  '''
  def __init__(self, z_dim=64, N=8, initialize=True):
    super().__init__()

    self.N = N
    self.group = gspaces.rot2dOnR2(self.N)
    self.d_model = 32
    self.seq_len = 64
    self.z_dim = z_dim

    self.in_type = enn.FieldType(
      self.group,
      [self.group.irrep(1)] + [self.group.trivial_repr] + [self.group.irrep(1)] + [self.group.trivial_repr]
    )
    out_type = enn.FieldType(self.group, self.d_model * [self.group.regular_repr])
    self.embed = EquivariantBlock(self.in_type, out_type, kernel_size=1, stride=1, padding=0, initialize=initialize)
    self.attn  = EquivMultiHeadAttention(n_head=1, d_model=self.d_model, d_k=self.d_model, d_v=self.d_model, initialize=initialize)

    self.fc_in_type = enn.FieldType(self.group, self.seq_len * self.d_model * [self.group.regular_repr])
    self.out_type = enn.FieldType(self.group, z_dim * [self.group.regular_repr])
    self.conv = EquivariantBlock(self.fc_in_type, self.out_type, kernel_size=1, stride=1, padding=0, initialize=initialize)

  def forward(self, x):
    batch_size = x.size(0)
    seq_l = x.size(1)

    # Combine batch and seq len for embedding: b*lq x 6 x 1 x 1
    # Seperate componets of eqeuivariant tensor: b x lq x c x d
    # Want to add the same position embedding to each element in the regular represenation
    x_geo = enn.GeometricTensor(x.view(batch_size * seq_l, 6, 1, 1), self.in_type)
    x_embed = self.embed(x_geo).tensor.view(batch_size, seq_l, self.N, self.d_model)

    # Apply attention and flatten in geo tensor: b x sq*c*d
    x_, _ = self.attn(
      x_embed,
      x_embed,
      x_embed,
    )
    x_ = enn.GeometricTensor(x_.view(batch_size, -1, 1, 1), self.fc_in_type)

    return self.conv(x_)

class MultiHeadAttention(nn.Module):
  def __init__(self, n_head, d_model, d_k, d_v):
    super().__init__()
    self.n_head = n_head
    self.d_model = d_model
    self.d_k = d_k
    self.d_v = d_v

    self.w_qs = ConvBlock(d_model, n_head * d_k, kernel_size=1, stride=1, padding=0, act=False)
    self.w_ks = ConvBlock(d_model, n_head * d_k, kernel_size=1, stride=1, padding=0, act=False)
    self.w_vs = ConvBlock(d_model, n_head * d_k, kernel_size=1, stride=1, padding=0, act=False)
    self.fc = ConvBlock(n_head * d_v, d_model, kernel_size=1, stride=1, padding=0, act=False)

    self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

  def forward(self, q, k, v):
    b = q.size(0)
    d_model, d_k, d_v, n_head = self.d_model, self.d_k, self.d_v, self.n_head
    len_q, len_k, len_v = 64, 64, 64

    # Pass through the pre-attention projection: b x lq x (n*dv)
    # Seperate different heads: b x lq x n x dv
    q = self.w_qs(q.view(b * len_q, d_model, 1, 1)).view(b, len_q, n_head, d_k)
    k = self.w_ks(k.view(b * len_q, d_model, 1, 1)).view(b, len_k, n_head, d_k)
    v = self.w_vs(v.view(b * len_q, d_model, 1, 1)).view(b, len_v, n_head, d_v)

    # Transpose for attention dot product: b x n x lq x dv
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    q, attn = self.attention(q, k, v)

    # Transpose to move the head dimension back: b x lq x n x dv
    # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
    q = q.transpose(1,2).contiguous().view(b * len_q, n_head * d_model, 1, 1)
    q = self.fc(q)

    return q.view(b, len_q, d_model), attn


class CnnForceEncoder(nn.Module):
  '''
  '''
  def __init__(self, z_dim=64):
    super().__init__()

    self.d_model = 32
    self.seq_len = 64
    self.z_dim = z_dim

    self.embed = ConvBlock(6, self.d_model, kernel_size=1, stride=1, padding=0)
    self.attn = MultiHeadAttention(n_head=1, d_model=self.d_model, d_k=self.d_model, d_v=self.d_model)
    self.conv = ConvBlock(self.seq_len * self.d_model, z_dim, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    batch_size =  x.size(0)

    x_embed = self.embed(x.view(batch_size * self.seq_len, 6, 1, 1)).view(batch_size, self.seq_len, self.d_model, 1, 1)
    x_, _ = self.attn(x_embed, x_embed, x_embed)

    x_ = x_.reshape(batch_size, -1, 1, 1)

    return self.conv(x_)
