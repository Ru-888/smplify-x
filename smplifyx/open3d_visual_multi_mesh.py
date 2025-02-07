import json
import os
import numpy as np
import trimesh
import pyrender
import pickle
import imageio
import yaml

from pyrender.constants import RenderFlags
from scipy.spatial.transform import Rotation as R


def load_kepoints3d(keypoints3d_path):
    with open(keypoints3d_path, 'rb') as f:
        data = json.load(f)
    return data


def load_pkl(pkl_path):
    """加载 result.pkl 文件并提取参数"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return {
        'global_orient': data['global_orient'][0],  # (3,) 轴角
        'camera_translation': data['camera_translation'][0],  # (3,)
        'camera_rotation': data['camera_rotation'][0]  # (3,3)
    }


def load_mesh(obj_path, color=(0.5, 0.5, 0.8, 1.0)):
    """加载 .obj 文件为 pyrender.Mesh 对象"""
    tri_mesh = trimesh.load(obj_path)
    mesh = pyrender.Mesh.from_trimesh(
        tri_mesh,
        smooth=False,
        material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.1,
            roughnessFactor=0.8
        )
    )
    return mesh


def build_world_pose(global_orient, camera_translation, camera_rotation, cam_extrinsic):
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
    print("global_orient:", global_orient)
    # 模型到相机坐标系的变换
    model_to_cam = np.eye(4)
    model_to_cam[:3, :3] = model_rot
    model_to_cam[:3, 3] = camera_translation

    # 从相机坐标系到世界坐标系
    world_pose = cam_extrinsic @ model_to_cam
    return world_pose


def visualize_multiview_meshes(frame_id, obj_pkl_dicts, cam_extrinsics, keypoints3d, save_dir="./sccamera_posereenshots"):
    ''' Keypoints3d Visualization '''
    # -------------------- 新增关键点可视化函数 --------------------
    def create_joint_spheres(keypoints, person_id=0):
        """创建关节球体可视化对象"""
        joint_colors = np.array([[1.0, 0.0, 0.0, 1.0]] * 25)  # 红色球体
        joint_radius = 0.02  # 根据场景比例调整

        # 创建球体几何体
        spheres = []
        for i, (x, y, z) in enumerate(keypoints):
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                continue  # 跳过无效关键点
            # 使用trimesh创建球体
            sphere = trimesh.primitives.Sphere(
                radius=joint_radius,
                center=(x, y, z)
            )
            # 转换为pyrender的Mesh并设置颜色
            mesh = pyrender.Mesh.from_trimesh(
                sphere,
                material=pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=joint_colors[i % len(joint_colors)]
                )
            )
            spheres.append(mesh)
        return spheres

    def create_skeleton_lines(keypoints):
        """创建骨骼连线可视化对象"""
        # OpenPose25关键点连接规则
        CONNECTIONS = np.array([
            [1, 0], [2, 1], [3, 2], [4, 3],
            [5, 1], [6, 5], [7, 6],
            [8, 1], [9, 8], [10, 9],
            [11, 10], [12, 8], [13, 12], [14, 13],
            [15, 0], [16, 0], [17, 15], [18, 16], [19, 14], [20, 19], [21, 14],
            [22, 11], [23, 22], [24, 11]
        ])

        line_color = [0.0, 1.0, 0.0, 1.0]  # 绿色连线
        line_width = 1.0

        lines = []
        for (i, j) in CONNECTIONS:
            if i >= len(keypoints) or j >= len(keypoints):
                continue
            start = keypoints[i]
            end = keypoints[j]
            if np.isnan(start).any() or np.isnan(end).any():
                continue

            # 创建线段几何体
            line = pyrender.Mesh(
                [pyrender.Primitive(
                    positions=np.array([start, end]),
                    mode=pyrender.constants.GLTF.LINES,
                    color_0=line_color
                )],
                is_visible=True
            )
            lines.append(line)
        return lines

    """可视化多视角下的网格"""
    scene = pyrender.Scene(bg_color=(1.0, 1.0, 1.0), ambient_light=(0.3, 0.3, 0.3))
    camera_transl =None
    # 遍历每个视角的数据
    for view, obj_pkl_pairs in obj_pkl_dicts.items():
        print(view, len(obj_pkl_pairs))
        #OpenCV标定的外参矩阵通常将点从世界坐标系变换到相机坐标系
        cam_ext = np.linalg.inv(cam_extrinsics[view-1])
        print(view, cam_ext)
        for obj_path, pkl_path in obj_pkl_pairs:  # 直接遍历列表
            # 加载参数和网格
            params = load_pkl(pkl_path)
            mesh = load_mesh(obj_path, color=np.append(np.random.rand(3), 1.0))

            # 计算世界坐标系下的位姿矩阵
            world_pose = build_world_pose(
                global_orient=params['global_orient'],
                camera_translation=params['camera_translation'],
                camera_rotation=params['camera_rotation'],
                cam_extrinsic=cam_ext
            )
            camera_transl = params['camera_translation']
            scene.add(mesh, pose=world_pose, name=f"{frame_id}:person_view_{view}")

    # -------------------- 新增关键点渲染部分 --------------------
    if keypoints3d is not None:
        for person in keypoints3d:
            person_id = person['id']
            person_kps = np.array(person['keypoints3d'])
            kps = person_kps[:, :3]  # 提取XYZ坐标# 处理带置信度的格式 (25,3/4)
            # 添加关节球体
            spheres = create_joint_spheres(kps, person_id)
            print("spheres", len(spheres))
            for sphere in spheres:
                sphere_node = pyrender.Node(
                    mesh=sphere,
                    # translation=position,
                    name=f"joints_{person_id}"
                )
                scene.add_node(sphere_node)

            # 添加骨骼连线
            lines = create_skeleton_lines(kps)
            for line in lines:
                line_node = pyrender.Node(
                    mesh=line,
                    name=f"skeleton_{person_id}"
                )
                scene.add_node(line_node)

    # -------------------- 设置全局观察相机 --------------------
    # 假设全局相机位于原点，看向场景中心
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    # 动态设置观察相机
    positions = []
    for node in scene.mesh_nodes:
        positions.append(node.translation)
    center = np.mean(positions, axis=0) if positions else [0, 0, 0]

    camera_pose = np.eye(4)
    # camera_pose[:3, 3] = center + [0, np.linalg.norm(center) * 0.3, np.linalg.norm(center) * 1.5]
    camera_pose[:3, 3] = camera_transl
    scene.add(camera, pose=camera_pose)

    # 添加光源
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=np.eye(4))

    # 截图回调函数
    def save_screenshot(viewer):
        # 获取当前视图参数
        viewport_size = viewer.viewport_size
        camera_pose = viewer._camera_node.matrix
        print("camera_pose:  ", camera_pose)

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
        run_in_thread=True, # 启用异步渲染
        registered_keys = {
            's': (save_screenshot, "保存当前视角截图"),
            'S': (save_screenshot, "保存当前视角截图")
        }
    )

# 示例调用
if __name__ == "__main__":

    # views=[2,8]
    # frame_id = '006837'
    # session = '12pm_Class_050823'
    # kepoinyts3d = f'/media/uq-04/ST8000/3d-Visual/{session}/3d_json/006800.json'
    views = [1]
    frame_id = '000000'
    session = 'feng'
    kepoinyts3d = f'/media/uq-04/ST8000/3d-Visual/{session}/test-dwpose-f-track/3d_json/{frame_id}.json'
    # cam_extrinsics = np.array([
    #     [[-0.118633, -0.990798, 0.065165, -0.093025],
    #      [-0.475979, -0.000851, -0.879456, 0.779975],
    #      [0.871419, -0.135350, -0.471497, 3.694555],
    #      [0.0, 0.0, 0.0, 1.0]],
    #     [[-0.800171, -0.596351, 0.063965, -0.017410],
    #      [-0.254180, 0.240575, -0.936758, 0.567165],
    #      [0.543249, -0.765826, -0.344082, 5.478421],
    #      [0.0, 0.0, 0.0, 1.0]],
    #     [[-0.898062, 0.439740, -0.010705, 1.301304],
    #      [0.160393, 0.304710, -0.938843, 0.858324],
    #      [-0.409585, -0.844856, -0.344179, 5.620791],
    #      [0.0, 0.0, 0.0, 1.0]],
    #     [[-0.998685, -0.029209, -0.042144, 0.526303],
    #      [0.026747, 0.404480, -0.914156, 0.793217],
    #      [0.043748, -0.914080, -0.403166, 4.604659],
    #      [0.0, 0.0, 0.0, 1.0]],
    #     [[0.999700, 0.010259, 0.022240, -0.449803],
    #      [0.024491, -0.410265, -0.911637, 0.956946],
    #      [-0.000228, 0.911909, -0.410393, 4.321124],
    #      [0.0, 0.0, 0.0, 1.0]],
    #     [[0.047894, 0.998726, 0.015903, -0.684285],
    #      [0.521832, -0.011442, -0.852972, 0.596665],
    #      [-0.851703, 0.049151, -0.521715, 4.234497],
    #      [0.0, 0.0, 0.0, 1.0]],
    #     [[0.746017, 0.665191, 0.031307, -0.248103],
    #      [0.231386, -0.214845, -0.948843, 1.099851],
    #      [-0.624435, 0.715096, -0.314194, 5.600614],
    #      [0.0, 0.0, 0.0, 1.0]],
    #     [[0.928410, -0.371383, 0.011403, 1.298628],
    #      [-0.109671, -0.303229, -0.946586, 0.936235],
    #      [0.355004, 0.877569, -0.322251, 5.024762],
    #      [0.0, 0.0, 0.0, 1.0]]
    # ])

    cam_extrinsics = np.load(f'/media/uq-04/ST8000/3d-Visual/{session}/extrinsics.npy')

    obj_pkl_dicts = {}

    keypoints3d = load_kepoints3d(kepoinyts3d)

    # Read the mesh files
    for view in views:
        mesh_dir = f'/media/uq-04/ST8000/3d-Visual/{session}/mesh/{view}/meshes/{frame_id}/'
        result_dir = f'/media/uq-04/ST8000/3d-Visual/{session}/mesh/{view}/results/{frame_id}/'
        print("mesh_dir", mesh_dir)
        print("result_dir", result_dir)
        files = os.listdir(mesh_dir)
        obj_paths = [os.path.join(mesh_dir, f) for f in files if f.lower().endswith('.obj')]
        person_ids = [f.split(".")[0] for f in files if f.lower().endswith('.obj')]
        print("person_ids", person_ids)
        print("obj_paths", obj_paths)
        res_files = os.listdir(result_dir)
        pkl_paths = [os.path.join(result_dir, f) for f in res_files if f.lower().endswith('.pkl')]
        print("pkl_paths", pkl_paths)
        v_obj_pkl_pairs = []
        for obj_path, pkl_path in zip(obj_paths, pkl_paths):
            v_obj_pkl_pairs.append((obj_path, pkl_path))
        obj_pkl_dicts[view] = v_obj_pkl_pairs

    visualize_multiview_meshes(frame_id, obj_pkl_dicts, cam_extrinsics, keypoints3d)


