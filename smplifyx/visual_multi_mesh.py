import json
import os
import pickle

import trimesh
import numpy as np
import pyrender
import torch
import smplx
from human_body_prior.tools.model_loader import load_vposer
from scipy.spatial.transform import Rotation as R
import os.path as osp
from utils import JointMapper

view = 2
frame_id = '006837'
session='12pm_Class_050823'
mesh_dir= f'/media/uq-04/ST8000/3d-Visual/results/{session}/{view}/3d/meshes/{frame_id}'
result_dir = f'/media/uq-04/ST8000/3d-Visual/results/{session}/{view}/3d/results/{frame_id}'
camera_center=[960,540]
focal_length=1800


def visualize_multiple_meshes(meshes, joints_mesh, translations, global_orients):
    """
    可视化多个OBJ文件，并根据每个mesh的位置进行调整。
    :param file_paths: OBJ文件路径列表
    """
    # 创建一个场景
    scene = pyrender.Scene()
    # 设置每个mesh的初始偏移量
    initial_offsets = []
    offset_distance = 0.1  # 设定每个 mesh 之间的最小距离

    # 读取并添加每个mesh
    for mesh, joints, translation, orient in zip(meshes, joints_mesh, translations, global_orients):

        # 创建pyrender.Mesh对象
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        # 创建一个节点并添加到场景中
        mesh_node = pyrender.Node(mesh=pyrender_mesh)

        keypoints=joints.squeeze().detach().numpy()
        center_position = keypoints.mean(axis=0)  # 计算关节的中心位置
        # 将中心位置作为偏移量进行调整
        initial_offsets.append(center_position)

        # 创建平移变换
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, 3] = translation
        print("transformation_matrix", transformation_matrix)

        # 将欧拉角转换为旋转矩阵
        rot = R.from_euler('xyz', orient)  # 假设 global_orient 的顺序是 [roll, pitch, yaw]
        rotation_matrix = rot.as_matrix()
        # 将旋转矩阵应用到变换矩阵
        transformation_matrix[:3, :3] = rotation_matrix
        print("transformation_matrix:", transformation_matrix)
        # 设置节点变换
        mesh_node.matrix = transformation_matrix
        # 将mesh加入场景，并应用变换
        scene.add_node(mesh_node)
        # # 创建关节点点云
        # keypoint_colors = np.array([[1.0, 0.0, 0.0]] * len(keypoints))  #默认为红色
        # joint_cloud = pyrender.Mesh.from_points(keypoints, colors=keypoint_colors)
        # # 创建关节点的节点
        # joint_cloud_node = pyrender.Node(mesh=joint_cloud)
        # # 添加到场景，调整关节位置
        # joint_cloud_node.matrix = transformation_matrix
        # scene.add_node(joint_cloud_node)

    # 创建一个透视相机
    camera = pyrender.camera.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=camera_center[0], cy=camera_center[1])
    # 设置相机位置
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 0, 3]  # 相机位置
    scene.add(camera, pose=camera_pose)

    # 添加光源
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    # 使用Viewer可视化
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)


# 加载 VPoser 模型
vposer_ckpt = "/home/uq-04/Workspace/SLRT/smplify-x/vposer"
vposer_ckpt = osp.expandvars(vposer_ckpt)
vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
vposer.eval()

#smplx
body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)

joint_mapper = JointMapper(body_mapping)
model_params = dict(model_path='/home/uq-04/Workspace/SLRT/smplify-x/models',
                        model_type='smplx',
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=torch.float32,
                        )
model = smplx.create(gender='neutral', **model_params)

# Read the mesh files
files = os.listdir(mesh_dir)
obj_files = [f for f in files if f.lower().endswith('.obj')]

meshes = []
pkl_files =[]
for obj_file in obj_files:
    obj_file_pth= os.path.join(mesh_dir, obj_file)
    data = trimesh.load(obj_file_pth, file_type='obj')
    meshes.append(data)
    pkl_files.append(obj_file.split('.')[0] + '.pkl')

#Read the results(.pkl ) of certain frame
joints_mesh=[]
global_orients =[]
camera_translations = []
for result_file in pkl_files:
    result_file_path = os.path.join(result_dir, result_file)
    with open(result_file_path, 'rb') as file:
        data = pickle.load(file)
    orient = torch.tensor(data['global_orient'], dtype=torch.float32)
    global_orients.append(orient)

    camera_transl = torch.tensor(data['camera_translation'], dtype=torch.float32)
    camera_translations.append(camera_transl)
    body_pose= data["body_pose"]
    pose_embedding = torch.tensor(body_pose, dtype=torch.float32)
    body_pose = vposer.decode(
        pose_embedding, output_type='aa').view(1, -1)

    output = model(body_pose=body_pose, return_verts=False,
                   return_full_pose=False)
    joints3d = output.joints
    joints_mesh.append(joints3d)
print("joints_mesh:", len(joints_mesh))


# 可视化多个mesh
visualize_multiple_meshes(meshes, joints_mesh, camera_translations, global_orients)