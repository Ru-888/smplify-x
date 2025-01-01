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

import sys
import os

import time

import cv2
import numpy as np

import torch
import torch.nn as nn
from sympy.codegen.ast import float32

from mesh_viewer import MeshViewer
import utils
from smplx.lbs import transform_mat

# guess_init 函数用于初始化相机的平移向量
@torch.no_grad()   # 使用 torch.no_grad() 装饰器来停止计算梯度，因为在初始化过程中不需要反向传播
def guess_init(model,
               joints_2d,
               edge_idxs,
               focal_length=5000,
               pose_embedding=None,
               vposer=None,
               use_vposer=True,
               dtype=torch.float32,
               model_type='smpl',
               **kwargs):
    ''' Initializes the camera translation vector

        Parameters
        ----------
        model: nn.Module
            The PyTorch module of the body
        joints_2d: torch.tensor 1xJx2
            The 2D tensor of the joints
        edge_idxs: list of lists
            A list of pairs, each of which represents a limb used to estimate
            the camera translation
        focal_length: float, optional (default = 5000)
            The focal length of the camera
        pose_embedding: torch.tensor 1x32
            The tensor that contains the embedding of V-Poser that is used to
            generate the pose of the model
        dtype: torch.dtype, optional (torch.float32)
            The floating point type used
        vposer: nn.Module, optional (None)
            The PyTorch module that implements the V-Poser decoder
        Returns
        -------
        init_t: torch.tensor 1x3, dtype = torch.float32
            The vector with the estimated camera location

    '''
    # 如果使用V-Poser，解码 pose_embedding 以获得姿态参数，并将其重塑为 1xN 的张量
    body_pose = vposer.decode(
        pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
    if use_vposer and model_type == 'smpl':
        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                 dtype=body_pose.dtype,
                                 device=body_pose.device)
        body_pose = torch.cat([body_pose, wrist_pose], dim=1)
    # 使用SMPL模型生成3D关节，不返回顶点和完整姿态
    output = model(body_pose=body_pose, return_verts=False,
                   return_full_pose=False)
    joints_3d = output.joints     # 使用SMPL模型生成3D关节，不返回顶点和完整姿态
    joints_2d = joints_2d.to(device=joints_3d.device)    # 将2D关节张量移动到与3D关节相同的设备

    # 初始化两个列表，用于存储3D和2D关节之间的差异
    diff3d = []
    diff2d = []
    # 对于每条边（肢体），计算3D和2D关节之间的差异，并添加到列表中
    # print("joints_3d", joints_3d.shape, joints_3d.shape)
    # print("joints_2d", joints_2d.shape, joints_2d.shape)
    # print("edge_idxs:",edge_idxs)
    for edge in edge_idxs:
        # print("edge:",edge[0], edge[1])
        # print(joints_3d[:, edge[0]], joints_3d[:, edge[1]])
        # print("joints_2d: ", joints_2d)
        # print(joints_2d[:, edge[0]], joints_2d[:, edge[1]])
        diff3d.append(joints_3d[:, edge[0]] - joints_3d[:, edge[1]])
        diff2d.append(joints_2d[:, edge[0]] - joints_2d[:, edge[1]])
    # 将差异列表堆叠成张量
    diff3d = torch.stack(diff3d, dim=1)
    diff2d = torch.stack(diff2d, dim=1)
    # 计算2D和3D差异的长度
    length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
    length_3d = diff3d.pow(2).sum(dim=-1).sqrt()
    # 计算2D和3D长度的平均值，作为高度
    height2d = length_2d.mean(dim=1)
    height3d = length_3d.mean(dim=1)
    # 使用焦距和高度比估计相机的深度
    est_d = focal_length * (height3d / height2d)
    # print(height3d, height2d)
    # just set the z value  初始化相机平移的x和y坐标为零
    batch_size = joints_3d.shape[0]
    x_coord = torch.zeros([batch_size], device=joints_3d.device,
                          dtype=dtype)
    y_coord = x_coord.clone()
    # 将x、y坐标和估计的深度堆叠成相机平移向量
    init_t = torch.stack([x_coord, y_coord, est_d], dim=1)
    return init_t

def joints3d_joints2d(gt_3d_points, camera):
    K = camera.K.detach().cpu().numpy().astype(np.float64)
    # print("K: ", K.dtype, K)
    rotation = camera.rotation.squeeze().detach().cpu().numpy().astype(np.float64) # 旋转向量
    translation = camera.translation.squeeze().detach().cpu().numpy().astype(np.float64)  # 平移向量
    # print("rotation: ", rotation.dtype, rotation)
    # print("translation: ", translation.dtype, translation)

    dist_coeffs = camera.dist_coeffs.detach().cpu().numpy().astype(np.float64)
    print("dist_coeffs: ", dist_coeffs.dtype, dist_coeffs)
    objective_points= gt_3d_points.squeeze().detach().cpu().numpy().astype(np.float64)
    # print("objective_points: ", objective_points.dtype, objective_points)
    # 使用cv2.projectPoints将3D点投影到2D平面
    image_points, _ = cv2.projectPoints(objective_points, rotation, translation, K, dist_coeffs)

    # image_points就是投影到2D平面的像素坐标
    img_points = torch.tensor(image_points.squeeze())

    return img_points.unsqueeze(0)

'''
# FittingMonitor类，用于监控和执行3D人体姿态估计的优化过程
FittingMonitor 类，其构造函数接受以下参数：
summary_steps: 优化过程中打印摘要信息的步长。
visualize: 是否在优化过程中进行可视化。
maxiters: 最大迭代次数。
ftol: 函数值的相对变化容忍度，用于判断优化是否收敛。
gtol: 梯度的绝对值变化容忍度，用于判断优化是否收敛。
body_color: 可视化时人体模型的颜色。
model_type: 模型类型，如'smpl'。
'''
class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

    '''
    上下文管理器的一部分，用于资源管理:
    __enter__ 在进入 with 块时调用，__exit__ 在离开 with 块时调用。
    如果启用了可视化，__enter__ 会创建一个 MeshViewer 对象，__exit__ 会关闭它。
    '''
    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    '''
        # 优化方法 run_fitting
        run_fitting 方法用于执行优化过程。它接受以下参数：
        optimizer: PyTorch优化器对象。
        closure: 用于计算梯度的函数。
        params: 需要优化的参数列表。
        body_model: 人体模型的PyTorch模块。
        use_vposer: 是否使用V-Poser。
        pose_embedding: V-Poser的姿态嵌入张量。
        vposer: V-Poser模块。
        方法内部，它执行优化循环，直到达到最大迭代次数或满足收敛条件。如果启用了可视化，它还会定期更新3D模型的可视化。
        '''
    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            if self.visualize and n % self.summary_steps == 0:
                body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(
                        1, -1) if use_vposer else None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                model_output = body_model(
                    return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            prev_loss = loss.item()

        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               gt_joints3d=None, loss=None,
                               joints_conf=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               pose_embedding=None,
                               create_graph=False,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl' and use_vposer
        # 这个闭包函数 fitting_func 将在优化器的每一步中被调用，用于执行模型的前向传播、损失计算和后向传播
        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = vposer.decode(
                pose_embedding, output_type='aa').view(
                    1, -1) if use_vposer else None

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)
            # 调用身体模型进行前向传播，根据参数返回顶点和完整的姿态
            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)
            total_loss = loss(body_model_output, camera=camera,
                              gt_joints3d=gt_joints3d,
                              body_model_faces=faces_tensor,
                              joints_conf=joints_conf,
                              joint_weights=joint_weights,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              **kwargs)
            #如果 backward 为 True，则进行后向传播，计算梯度。
            if backward:
                total_loss.backward(create_graph=create_graph)

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            return total_loss

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        return SMPLifyCameraInitLoss(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


class SMPLifyLoss(nn.Module):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 expr_prior=None,
                 angle_prior=None,
                 jaw_prior=None,
                 use_joints_conf=True,
                 use_face=True, use_hands=True,
                 left_hand_prior=None, right_hand_prior=None,
                 interpenetration=True, dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0, jaw_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 reduction='sum',
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.body_pose_prior = body_pose_prior

        self.shape_prior = shape_prior

        self.interpenetration = interpenetration
        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance

        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face
        if self.use_face:
            self.expr_prior = expr_prior
            self.jaw_prior = jaw_prior

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        if self.use_hands:
            self.register_buffer('hand_prior_weight',
                                 torch.tensor(hand_prior_weight, dtype=dtype))
        if self.use_face:
            self.register_buffer('expr_prior_weight',
                                 torch.tensor(expr_prior_weight, dtype=dtype))
            self.register_buffer('jaw_prior_weight',
                                 torch.tensor(jaw_prior_weight, dtype=dtype))
        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, joints_conf,
                body_model_faces, joint_weights,
                use_vposer=False, pose_embedding=None,
                **kwargs):
        # projected_joints = camera(body_model_output.joints)

        projected_joints = body_model_output.joints
        # Calculate the weights for each joints
        weights = (joint_weights * joints_conf
                   if self.use_joints_conf else
                   joint_weights).unsqueeze(dim=-1)

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        joint_diff = self.robustifier(gt_joints - projected_joints)
        joint_loss = (torch.sum(weights ** 2 * joint_diff) *
                      self.data_weight ** 2)
        # print("SMPLifyLoss gt_joints: ", gt_joints.shape,  gt_joints)
        # print("projected_joints: ", projected_joints.shape, projected_joints)
        # print("joint_diff: ", joint_diff)
        # print("SMPLifyLoss body_model_output.joints: ", body_model_output.joints)
        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)
        else:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2

        shape_loss = torch.sum(self.shape_prior(
            body_model_output.betas)) * self.shape_weight ** 2
        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight

        # Apply the prior on the pose space of the hand
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        if self.use_hands and self.left_hand_prior is not None:
            left_hand_prior_loss = torch.sum(
                self.left_hand_prior(
                    body_model_output.left_hand_pose)) * \
                self.hand_prior_weight ** 2

        if self.use_hands and self.right_hand_prior is not None:
            right_hand_prior_loss = torch.sum(
                self.right_hand_prior(
                    body_model_output.right_hand_pose)) * \
                self.hand_prior_weight ** 2

        expression_loss = 0.0
        jaw_prior_loss = 0.0
        if self.use_face:
            expression_loss = torch.sum(self.expr_prior(
                body_model_output.expression)) * \
                self.expr_prior_weight ** 2

            if hasattr(self, 'jaw_prior'):
                jaw_prior_loss = torch.sum(
                    self.jaw_prior(
                        body_model_output.jaw_pose.mul(
                            self.jaw_prior_weight)))

        pen_loss = 0.0
        # Calculate the loss due to interpenetration
        if (self.interpenetration and self.coll_loss_weight.item() > 0):
            batch_size = projected_joints.shape[0]
            triangles = torch.index_select(
                body_model_output.vertices, 1,
                body_model_faces).view(batch_size, -1, 3, 3)

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)

            # Remove unwanted collisions
            if self.tri_filtering_module is not None:
                collision_idxs = self.tri_filtering_module(collision_idxs)

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(
                    self.coll_loss_weight *
                    self.pen_distance(triangles, collision_idxs))

        total_loss = (joint_loss + pprior_loss + shape_loss +
                      angle_prior_loss + pen_loss +
                      jaw_prior_loss + expression_loss +
                      left_hand_prior_loss + right_hand_prior_loss)
        return total_loss


class SMPLifyCameraInitLoss(nn.Module):

    def __init__(self, init_joints_idxs, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2, dtype=torch.float32,
                 **kwargs):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype

        if trans_estimation is not None:
            self.register_buffer(
                'trans_estimation',
                utils.to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            utils.to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints3d,
                **kwargs):
        # print("SMPLifyCameraInitLoss body_model_output.joints: ", body_model_output.joints)
        # print("SMPLifyCameraInitLoss gt_joints: ", gt_joints)
        # projected_joints = camera(body_model_output.joints)
        # gt_joints_projected = camera(gt_joints)
        model_out_joints = body_model_output.joints

        joint_error = torch.pow(
            torch.index_select(gt_joints3d, 1, self.init_joints_idxs) -
            torch.index_select(model_out_joints, 1, self.init_joints_idxs),
            2)
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2
        # print("joint_loss:", joint_loss)
        depth_loss = 0.0
        if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
                None):
            # print("camera.translation: ", camera.translation)
            # print("self.trans_estimation: ", self.trans_estimation)
            depth_loss = self.depth_loss_weight ** 2 * torch.sum((
                camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))

        return joint_loss + depth_loss

