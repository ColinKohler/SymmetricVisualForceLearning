import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn

class EquivariantLinearBlock(nn.Module):
  '''
  A equivariant ResNet block.
  '''
  def __init__(self, in_type, out_type, initialize=True, act=True, norm=False):
    super().__init__()
    self.norm = norm
    self.act = act

    self.fc = enn.Linear(
      in_type,
      out_type,
      initialize=initialize
    )
    if self.norm:
      self.bn = enn.InnerBatchNorm(out_type)
    if self.act:
      self.relu = enn.ReLU(out_type, inplace=True)

  def forward(self, x):
    out = self.fc(x)
    if self.norm:
      out = self.bn(out)
    if self.act:
      out = self.relu(out)

    return out

class EquivariantBlock(nn.Module):
  '''
  A equivariant ResNet block.
  '''
  def __init__(self, in_type, out_type, kernel_size=3, stride=1, padding=1, initialize=True, act=True, norm=False):
    super().__init__()
    self.norm = norm
    self.act = act

    self.conv = enn.R2Conv(
      in_type,
      out_type,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      initialize=initialize
    )
    if self.norm:
      self.bn = enn.InnerBatchNorm(out_type)
    if self.act:
      self.relu = enn.ReLU(out_type, inplace=True)

  def forward(self, x):
    out = self.conv(x)
    if self.norm:
      out = self.bn(out)
    if self.act:
      out = self.relu(out)

    return out

class ResnetBlock(nn.Module):
  '''
  A ResNet block.
  '''
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True, norm=False):
    super().__init__()
    self.norm = norm
    self.act = act

    self.conv = nn.Conv2d(
      in_channels,
      out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
    )
    if self.norm:
      self.bn = nn.BatchNorm2d(out_channels)
    if self.act:
      self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    out = self.conv(x)
    if self.norm:
      out = self.bn(out)
    if self.act:
      out = self.relu(out)

    return out

class CausalConv1d(nn.Conv1d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
    self.__padding = (kernel_size - 1) * dilation

    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=self.__padding, dilation=dilation, bias=bias)

  def forward(self, x):
    res = super().forward(x)
    if self.__padding != 0:
      return res[:, :, :-self.__padding]
    return res

class ScaledDotProductAttention(nn.Module):
  def __init__(self, temperature, attn_dropout=0.1):
    super().__init__()
    self.temperature = temperature
    self.dropout = nn.Dropout(attn_dropout)

  def forward(self, q, k, v, mask=None):
    attn = torch.matmul(q / self.temperature, k.transpose(2,3))

    if mask is not None:
      attn = attn.masked_fill(mask == 0, -1e9)

    attn = self.dropout(F.softmax(attn, dim=-1))
    output = torch.matmul(attn, v)

    return output, attn

class SelfAttention(nn.Module):
  def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
    super().__init__()
    self.n_head = n_head
    self.d_k = d_k
    self.d_v = d_v

    self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
    self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
    self.w_vs = nn.Linear(d_model, n_head * d_k, bias=False)
    self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    self.dropout = nn.Dropout(dropout)
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

  def forward(self, q, k, v, mask=None):
    d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
    sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

    residual = q

    # Pass through  the pre-attention projection: b x lq x (n*dv)
    # Seperate different heads: b x lq x n x dv
    q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
    k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
    v = self.w_vs(v).view(sz_b, len_v, n_head, d_k)

    # Transpose for attention dot product: b x n x lq x dv
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

    if mask is not None:
      mask.unsqueeze(1)  # For head axis broadcasting

    q, attn = self.attention(q, k, v, mask=mask)

    # Transpose to move the head dimension back: b x lq x n x dv
    # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
    q = q.transpose(1,2).contiguous().view(sz_b, len_q, -1)
    q = self.dropout(self.fc(q))
    q += residual

    q = self.layer_norm(q)
    return q, attn
