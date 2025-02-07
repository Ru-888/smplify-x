import json
import os
import pickle

import trimesh
import numpy as np
import pyrender
import imageio
import torch
import smplx
from trimesh.path.exchange.misc import faces_to_path

from human_body_prior.tools.model_loader import load_vposer
from scipy.spatial.transform import Rotation as R
import os.path as osp
from utils import JointMapper
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# view = 8
# frame_id = '006800'
view = 2
frame_id = '006838'
session='12pm_Class_050823'
mesh_dir= f'/media/uq-04/ST8000/3d-Visual/results/{session}/2-3d/meshes/{frame_id}/{view}/'
result_dir = f'/media/uq-04/ST8000/3d-Visual/results/{session}/2-3d/results/{frame_id}/{view}/'
camera_center=[960,540]
focal_length=1800

def align_vertices(smplx_data):
    first_person = smplx_data[0]
    for index, people in enumerate(smplx_data):
        if index != 0:
            vertices = people["vertices"]
            # 计算相对平移和旋转
            delta_translation = people['camera_translation'] - first_person['camera_translation']
            relative_rotation = np.linalg.inv(first_person['camera_rotation']).dot(people['camera_rotation'])
            # 将第其他人的顶点对齐到第一个人的坐标系
            vertices_aligned = (relative_rotation @ vertices.T).T - delta_translation
            people["vertices"] = vertices_aligned
            # 验证对齐后的结果
            print("Vertices2 Aligned Shape:", vertices_aligned.shape)
    return smplx_data

def build_world_pose(global_orient, camera_translation):
    """
    构建世界坐标系下的网格位姿矩阵
    :param global_orient: SMPL-X模型的全局旋转 (3,)
    :param camera_translation: 模型到相机的平移 (3,)
    :param camera_rotation: 相机自身的旋转 (3,3) 单矩阵阵
    :param cam_extrinsic: 相机外参矩阵 (4,4)
    :return: 4x4世界变换矩阵
    """
    # 模型旋转矩阵
    model_rot = R.from_rotvec(global_orient).as_matrix()

    # 模型到相机坐标系的变换
    model_to_cam = np.eye(4)
    model_to_cam[:3, :3] = model_rot
    model_to_cam[:3, 3] = camera_translation

    # 组合完整变换矩阵
    world_pose =  model_to_cam
    return world_pose

def visualization(smplx_data, save_dir="./screenshots"):
    # 绘制两个人的 Mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print("smplx_data", smplx_data)
    # 第一个人的 Mesh
    vertices1 = np.array(smplx_data[0]["vertices"])
    ax.scatter(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], c='blue', s=1, label='Person 1')
    for i, people in enumerate(smplx_data):
        # 对齐后的 Mesh
        if i > 0:
            vertices_aligned=people["vertices"]
            ax.scatter(vertices_aligned[:, 0], vertices_aligned[:, 1], vertices_aligned[:, 2], c='red', s=1,
                       label='Person 2 (Aligned)')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

    # 创建一个场景
    scene = pyrender.Scene(bg_color=(1.0, 1.0, 1.0), ambient_light=(0.3, 0.3, 0.3))

    for i, people in enumerate(smplx_data):
        # mesh reconstruction
        vertices = people["vertices"]
        faces = people["faces"]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # 创建pyrender.Mesh对象
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh,
            smooth=False,
            material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=(0.5, 0.5, 0.8, 1.0),
                metallicFactor=0.1,
                roughnessFactor=0.8
        ))

        # # 计算世界坐标系下的位姿矩阵
        # world_pose = build_world_pose(
        #     global_orient=people['global_orient'],
        #     camera_translation=people['camera_translation']
        # )

        # 添加到场景
        scene.add(pyrender_mesh,  name=f"person_view_{view}")
    # -------------------- 设置全局观察相机 --------------------
    # 假设全局相机位于原点，看向场景中心
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    # 动态设置观察相机
    positions = []
    for node in scene.mesh_nodes:
        positions.append(node.translation)
    center = np.mean(positions, axis=0) if positions else [0, 0, 0]

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = center + [0, np.linalg.norm(center) * 0.3, np.linalg.norm(center) * 1.5]
    scene.add(camera, pose=camera_pose)

    # 添加光源
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=np.eye(4))

    # 截图回调函数
    def save_screenshot(viewer):
        # 获取当前视图参数
        viewport_size = viewer.viewport_size
        camera_pose = viewer._camera_node.matrix

        # 创建离屏渲染器
        renderer = pyrender.OffscreenRenderer(
            viewport_width=viewport_size[0],
            viewport_height=viewport_size[1],
            point_size=1.0
        )

        # 渲染当前视角
        try:
            color, _ = renderer.render(scene, camera_pose=camera_pose)
            filename = f"{save_dir}/screenshot_{frame_id}.png"
            imageio.imwrite(filename, color)
            print(f"\033[92m截图已保存至：{frame_id}\033[0m")
        except Exception as e:
            print(f"\033[91m截图保存失败：{str(e)}\033[0m")
        finally:
            renderer.delete()

    # 调整后的完整Viewer配置
    viewer = pyrender.Viewer(
        scene,
        viewport_size=(1920, 1080),
        use_raymond_lighting=False,  # 禁用默认光源（因已手动添加）
        run_in_thread=True,  # 启用异步渲染
        registered_keys={
            's': (save_screenshot, "保存当前视角截图"),
            'S': (save_screenshot, "保存当前视角截图")
        }
    )




def main():
    # Read the mesh files
    files = os.listdir(mesh_dir)
    obj_files = [f for f in files if f.lower().endswith('.obj')]

    smplx_data = []
    for obj_file in obj_files:
        people = {}
        obj_file_pth= os.path.join(mesh_dir, obj_file)
        mesh = trimesh.load(obj_file_pth)
        people['vertices'] = mesh.vertices
        people['faces'] = mesh.faces
        print("faces:", mesh.faces)
        # Read the results(.pkl ) of certain frame
        result_file_path = os.path.join(result_dir, obj_file.split('.')[0] + '.pkl')
        with open(result_file_path, 'rb') as file:
            data = pickle.load(file)
        orient = data['global_orient']
        people['global_orient']=orient
        camera_transl = data['camera_translation']
        people['camera_translation']=camera_transl
        camera_rotation = data['camera_rotation']
        people['camera_rotation']= camera_rotation.squeeze()
        smplx_data.append(people)
        print("camera_rotations:",orient.shape, camera_transl.shape, camera_rotation.squeeze().shape, type(camera_rotation.squeeze()))

    aligned_smplx = align_vertices(smplx_data)
    visualization(aligned_smplx)

if __name__ == "__main__":
    main()