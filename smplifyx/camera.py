# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import namedtuple

import torch
import torch.nn as nn
import numpy as np

from smplx.lbs import transform_mat


PerspParams = namedtuple('ModelOutput',
                         ['rotation', 'translation', 'center',
                          'focal_length'])


def create_camera(camera_type='persp', **kwargs):
    if camera_type.lower() == 'persp':
        return PerspectiveCamera(**kwargs)
    else:
        raise ValueError('Uknown camera type: {}'.format(camera_type))


class PerspectiveCamera(nn.Module):

    FOCAL_LENGTH = 5000

    def __init__(self, rotation=None, translation=None,
                 focal_length_x=None, focal_length_y=None,
                 batch_size=1, K=None, dist_coeffs=None,
                 center=None, dtype=torch.float32, **kwargs):
        super(PerspectiveCamera, self).__init__()

        self.batch_size = batch_size
        self.dtype = dtype
        # self.focal_length_x=focal_length_x
        # self.focal_length_y = focal_length_y
        # self.center = center
        # self.rotation = rotation
        # self.translation = translation
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('zero',
                             torch.zeros([batch_size], dtype=dtype))

        if focal_length_x is None or type(focal_length_x) == float:
            focal_length_x = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_x is None else
                focal_length_x,
                dtype=dtype)

        if focal_length_y is None or type(focal_length_y) == float:
            focal_length_y = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_y is None else
                focal_length_y,
                dtype=dtype)

        self.register_buffer('focal_length_x', focal_length_x)
        self.register_buffer('focal_length_y', focal_length_y)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        else:
            center = torch.tensor(center, dtype=dtype)
        self.register_buffer('center', center)
        self.register_buffer('K', torch.tensor(np.array(K).reshape(3,3), dtype=dtype))
        self.register_buffer('dist_coeffs', torch.tensor(np.array(dist_coeffs), dtype=dtype))
        if rotation is None:
            rotation = torch.eye(
                3, dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)
        else:
            rotation = torch.tensor(np.array(rotation).reshape(3, 3), dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)
        rotation = nn.Parameter(rotation, requires_grad=True)
        self.register_parameter('rotation', rotation)
        print('PerspectiveCamera: self.rotation ', self.rotation)

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)
        else:
            translation = torch.tensor([translation]* batch_size, dtype=dtype)

        translation = nn.Parameter(translation, requires_grad=True)
        self.register_parameter('translation', translation)
        print('PerspectiveCamera: self.translation ', self.translation.shape, self.translation)
    # 前向传播方法，接受3D点作为输入
    def forward(self, points):
        device = points.device

        with torch.no_grad():
            #创建相机内参矩阵，并设置焦距值
            camera_mat = torch.zeros([self.batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
            camera_mat[:, 0, 0] = self.focal_length_x
            camera_mat[:, 1, 1] = self.focal_length_y
        # print("camera_mat: ",camera_mat)
        # 计算相机的外参矩阵，包括旋转和平移
        # print("self.translation:: ",self.translation.shape,self.translation)
        camera_transform = transform_mat(self.rotation,
                                         self.translation.unsqueeze(dim=-1))


        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)
        # print("points_h: ", points_h.shape, points_h)
        # print("camera_transform.shape: ", camera_transform.shape, camera_transform)
        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])
        # print("projected_points: ", projected_points.shape, projected_points)
        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))
        # print("img_points: ", img_points.shape, img_points)
        # print("camera_mat: ", camera_mat.shape, camera_mat)
        # print("self.center: ", self.center.shape, self.center.unsqueeze(0).unsqueeze(0).shape)
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) + self.center.unsqueeze(0).unsqueeze(0)
        print("img_points: ", img_points.shape)

        return img_points
