import torch
import e2cnn.nn
import numpy as np
import numpy.random as npr
import scipy.ndimage

def dictToCpu(state_dict):
  cpu_dict = dict()
  for k, v in state_dict.items():
    if isinstance(v, torch.Tensor):
      cpu_dict[k] = v.cpu()
    elif isinstance(v, dict):
      cpu_dict[k] = dictToCpu(v)
    elif isinstance(v, e2cnn.nn.EquivariantModule):
      cpu_dict[k] = v.cpu()
    else:
      cpu_dict[k] = v

  return cpu_dict

def normalizeObs(obs):
  obs = np.clip(obs, 0, 0.32)
  obs = obs / 0.4 * 255
  obs = obs.astype(np.uint8)

  return obs

def unnormalizeObs(obs):
  return obs / 255 * 0.4

def normalizeForce(force):
  force = np.clip(force, 0, 100)
  force = force / 100
  return force

def perturb(obs, fxy_1, fxy_2, obs_, fxy_1_, fxy_2_, dxy, set_theta_zero=False, set_trans_zero=False):
  '''

  '''
  obs_size = obs.shape[-2:]

  # Compute random rigid transform
  theta, trans, pivot = getRandomImageTransformParams(obs_size)
  if set_theta_zero:
    theta = 0.
  if set_trans_zero:
    trans = [0., 0.]
  transform = getImageTransform(theta, trans, pivot)
  transform_params = theta, trans, pivot

  rot = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
  rotated_dxy = rot.dot(dxy)
  rotated_dxy = np.clip(rotated_dxy, -1, 1)

  rotated_fxy_1 = rot.dot(fxy_1)
  rotated_fxy_2 = rot.dot(fxy_2)
  rotated_fxy_1_ = rot.dot(fxy_1_)
  rotated_fxy_2_ = rot.dot(fxy_2_)

  # Apply rigid transform to obs
  obs = scipy.ndimage.affine_transform(obs, np.linalg.inv(transform), mode='nearest', order=1)
  obs_ = scipy.ndimage.affine_transform(obs_, np.linalg.inv(transform), mode='nearest', order=1)

  return obs, rotated_fxy_1, rotated_fxy_2, obs_, rotated_fxy_1_, rotated_fxy_2_, rotated_dxy, transform_params

def getRandomImageTransformParams(obs_size):
  ''''''
  theta = npr.rand() * 2 * np.pi
  trans = npr.randint(0, obs_size[0] // 10, 2) - obs_size[0] // 20
  pivot = (obs_size[1] / 2, obs_size[0] / 2)

  return theta, trans, pivot

def getImageTransform(theta, trans, pivot=(0,0)):
  ''''''
  pivot_t_image = np.array([[1., 0., -pivot[0]],
                            [0., 1., -pivot[1]],
                            [0., 0., 1.]])
  image_t_pivot = np.array([[1., 0., pivot[0]],
                            [0., 1., pivot[1]],
                            [0., 0., 1.]])
  transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                        [np.sin(theta), np.cos(theta),  trans[1]],
                        [0., 0., 1.]])
  return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))

