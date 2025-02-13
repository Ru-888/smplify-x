import bpy
import numpy as np
import pickle
import os
from pathlib import Path
from mathutils import Matrix, Vector, Euler
from scipy.spatial.transform import Rotation as R
import math

views = [2, 8]
frame_id = '006801'
session = '12pm_Class_050823'
align_frame = [45, 37, 29, 23, 21, 15, 7, 0]  # 12pm_Class_050823
kepoinyts3d = f'/media/uq-04/ST8000/3d-Visual/{session}/test-dwpose-f-track/3d_json/{frame_id}.json'
OUTPUT_DIR = f'/Users/ru/Desktop/GYM/3d-Visual/result/{session}/output/'  # 渲染图保存目录


class SMPLXMultiViewVisualizer:
    def __init__(self, views, root_dir, cam_extrinsics):
        """
        初始化可视化器
        mesh_dir: 存放obj文件的目录
        result_dir: 存放姿态参数pkl文件的目录
        cam_extrinsics: 包含两个相机外参的字典 {cam_id: 4x4 matrix}
        """
        self.views = views
        self.root_dir = Path(root_dir)
        self.cam_extrinsics = cam_extrinsics

        # 清空场景并设置渲染参数
        self.clear_scene()
        self.setup_render_params()

    def clear_scene(self):
        """清空Blender场景"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def setup_render_params(self):
        """设置渲染参数"""
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'

        # 设置相机参数
        camera_data = bpy.data.cameras.new(name='Global_Camera')

        # 设置渲染分辨率
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        scene.render.pixel_aspect_x = 1.0
        scene.render.pixel_aspect_y = 1.0

        # 设置相机基本参数
        camera_data.type = 'PERSP'  # 使用透视相机
        camera_data.sensor_width = 36.0  # 35mm全画幅传感器

        # 先使用一个临时焦距，后面会根据场景大小调整
        camera_data.lens = 1800 * camera_data.sensor_width / 1920

        # 设置较大的视距范围
        camera_data.clip_start = 0.1
        camera_data.clip_end = 1000.0

        return camera_data

    def load_pose_params(self, person_id, view):
        """从pkl文件加载姿态参数"""
        pkl_path = Path(
            f'{self.root_dir}/{session}/{view}/results/{str(align_frame[view - 1] + int(frame_id)).zfill(6)}/{person_id}.pkl')
        print("pkl_path: ", pkl_path)
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def set_object_transform(self, obj, global_orientation, camera_translation, extrinsics, view):
        """
        设置物体的世界坐标系变换
        Args:
            obj: Blender物体
            global_orientation: 全局旋转向量 (3,)
            camera_translation: 相机坐标系下的平移向量 (3,)
            extrinsics: 相机外参矩阵 (4,4)，从世界坐标系到相机坐标系的变换
            view: 视角编号
        """
        # 1. 首先构建物体在相机坐标系下的变换矩阵
        rotation_matrix = R.from_rotvec(global_orientation).as_matrix()
        object_to_camera = np.eye(4)
        object_to_camera[:3, :3] = rotation_matrix
        object_to_camera[:3, 3] = camera_translation

        # 2. 计算相机到世界坐标系的变换矩阵
        extrinsic = np.linalg.inv(extrinsics)

        # 3. 将物体从相机坐标系变换到世界坐标系
        world_transform = extrinsic @ object_to_camera

        # 4. 如果是第二个视角，添加Z轴旋转
        if view == self.views[1]:  # 第二个视角
            # 获取物体的当前位置
            current_position = world_transform[:3, 3]

            # 创建Z轴旋转矩阵（逆时针80度）
            z_rotation = np.eye(4)
            z_rotation[:3, :3] = R.from_euler('z', 80, degrees=True).as_matrix()

            # 1. 将物体移动到原点
            translate_to_origin = np.eye(4)
            translate_to_origin[:3, 3] = -current_position

            # 2. 应用旋转
            # 3. 移回原来的位置
            translate_back = np.eye(4)
            translate_back[:3, 3] = current_position

            # 组合变换：移动到原点 -> 旋转 -> 移回原位
            world_transform = translate_back @ z_rotation @ translate_to_origin @ world_transform

        # 5. 转换为Blender矩阵格式并应用
        world_matrix = Matrix(world_transform.tolist())
        obj.matrix_world = world_matrix

        # 打印调试信息
        if view == self.views[1]:
            print(f"Rotating mesh around its position: {current_position}")
            print(f"Final transformation matrix:\n{world_transform}")

        return obj

    def import_mesh(self, obj_path, person_id, view):
        """导入单个mesh并应用变换"""
        # 导入obj文件
        bpy.ops.wm.obj_import(filepath=str(obj_path))
        obj = bpy.context.selected_objects[0]

        # 加载对应的姿态参数
        pose_data = self.load_pose_params(person_id, view)
        extrinsics = self.cam_extrinsics[view - 1]

        # 打印调试信息
        print(f"Processing view {view}, person {person_id}")
        print(f"Extrinsics matrix:\n{extrinsics}")

        # 应用全局旋转和平移
        global_orient = pose_data['global_orient'][0]  # shape: (3,)
        camera_translation = pose_data['camera_translation'][0]  # shape: (3,)

        print(f"Global orientation: {global_orient}")
        print(f"Camera translation: {camera_translation}")

        obj = self.set_object_transform(
            obj,
            global_orient,
            camera_translation,
            extrinsics,
            view  # 添加view参数
        )
        return obj

    def create_camera(self, location, rotation, name):
        """创建相机"""
        cam_data = bpy.data.cameras.new(name=f'Camera_{name}')
        cam_obj = bpy.data.objects.new(f'Camera_{name}', cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)

        cam_obj.location = location
        # 明确指定使用XYZ顺序的Euler旋转
        cam_obj.rotation_euler = Euler(rotation, 'XYZ')

        # 打印详细的旋转信息
        print(f"Camera rotation order: XYZ")
        print(f"Camera rotation (radians): {rotation}")
        print(
            f"Camera rotation (degrees): ({np.degrees(rotation[0])}, {np.degrees(rotation[1])}, {np.degrees(rotation[2])})")

        return cam_obj

    def render_scene(self, output_path, resolution=(1920, 1080)):
        """渲染场景"""
        scene = bpy.context.scene
        scene.render.resolution_x = resolution[0]
        scene.render.resolution_y = resolution[1]

        # 设置输出路径
        scene.render.filepath = str(output_path)

        # 渲染图像
        bpy.ops.render.render(write_still=True)

    def get_global_bounding_box(self, obj):
        """获取物体全局坐标系下的包围盒顶点"""
        return [obj.matrix_world @ Vector(v) for v in obj.bound_box]

    def calculate_scene_bounds(self, meshes):
        """计算场景中所有网格的全局包围盒"""
        min_coord = Vector((float('inf'),) * 3)
        max_coord = Vector((-float('inf'),) * 3)

        for obj in meshes:
            if obj.type == 'MESH':
                bbox = self.get_global_bounding_box(obj)
                for v in bbox:
                    min_coord.x = min(min_coord.x, v.x)
                    min_coord.y = min(min_coord.y, v.y)
                    min_coord.z = min(min_coord.z, v.z)
                    max_coord.x = max(max_coord.x, v.x)
                    max_coord.y = max(max_coord.y, v.y)
                    max_coord.z = max(max_coord.z, v.z)

        return min_coord, max_coord

    def setup_scene(self):
        """设置场景，导入所有mesh并创建相机"""
        meshes = {}
        imported_persons = set()
        all_meshes = []  # 用于存储所有导入的mesh对象

        # 遍历每个相机视角
        print("views:", views)
        for view in views:
            mesh_files = []
            mesh_dir = Path(
                f'{self.root_dir}/{session}/{view}/meshes/{str(align_frame[view - 1] + int(frame_id)).zfill(6)}/')
            print("mesh_dir:", mesh_dir)

            # 导入该相机下的所有mesh
            for obj_file in sorted(mesh_dir.glob('*.obj')):
                print("obj_file", obj_file)
                person_id = obj_file.stem.split('.')[0]  # 假设文件名格式为 frame_XXXX.obj

                # 如果不是第一个视角且person_id已经导入过，则跳过
                if view != self.views[0] and person_id in imported_persons:
                    print(f"Skipping duplicate mesh for person {person_id} in view {view}")
                    continue

                # 导入mesh并记录person_id
                mesh = self.import_mesh(obj_file, person_id, view)
                mesh_files.append(mesh)
                all_meshes.append(mesh)  # 添加到所有mesh列表
                imported_persons.add(person_id)

                # 打印调试信息
                print(f"Imported mesh for person {person_id} from view {view}")

            if mesh_files:  # 只有当有mesh被导入时才添加到字典
                meshes[view] = mesh_files

        # 打印导入统计信息
        print(f"Total imported persons: {len(imported_persons)}")
        print(f"Imported person IDs: {sorted(list(imported_persons))}")

        # 配置参数
        RESOLUTION = (1920, 1080)
        SAFE_MARGIN = 1.05  # 减小安全边距到5%
        DEFAULT_SENSOR_WIDTH = 36.0

        # 获取场景包围盒
        min_coord, max_coord = self.calculate_scene_bounds(all_meshes)
        if min_coord == Vector((float('inf'),) * 3):
            print("未找到人体网格！")
            return meshes, None

        # 计算包围盒中心和尺寸
        center = (min_coord + max_coord) * 0.5
        size = max_coord - min_coord
        max_dimension = max(size.x, size.y, size.z) * SAFE_MARGIN

        # 创建相机
        camera_data = bpy.data.cameras.new(name='Global_Camera')
        global_cam = bpy.data.objects.new('Global_Camera', camera_data)
        bpy.context.scene.collection.objects.link(global_cam)

        # 设置相机参数
        camera_data.type = 'PERSP'
        camera_data.sensor_width = DEFAULT_SENSOR_WIDTH

        # 计算视野参数
        aspect_ratio = RESOLUTION[0] / RESOLUTION[1]

        # 根据宽高比自动选择适配方向
        if aspect_ratio >= 1:
            fov_ratio = (size.x * aspect_ratio, size.y)
        else:
            fov_ratio = (size.x, size.y / aspect_ratio)

        required_fov = 2 * math.atan(max(fov_ratio) / (2 * max_dimension))

        # 设置相机焦距
        camera_data.lens = (DEFAULT_SENSOR_WIDTH / (2 * math.tan(required_fov / 2)))

        # 调整相机位置为偏上水平视角
        # 减小相机到场景中心的距离
        camera_distance = max_dimension * 1.15  # 从2.5减小到1.5，使相机更靠近场景

        # 调整相机高度和水平偏移
        height_offset = max_dimension * 0.6  # 降低相机高度（从0.8减小到0.6）
        horizontal_offset = camera_distance * 0.6  # 减小水平偏移（从0.8减小到0.6）

        # 将相机放置在场景中心的斜前上方
        camera_pos = center + Vector((
            -horizontal_offset * 1.2,  # 向左偏移，稍微增加以获得更好的侧面视角
            -horizontal_offset * 0.8,  # 向后偏移，稍微减小以靠近场景
            height_offset  # 保持相同的高度
        ))

        # 设置相机位置
        global_cam.location = camera_pos

        # 使用look_at函数让相机朝向场景中心
        look_at_direction = center - camera_pos
        rot_quat = look_at_direction.to_track_quat('-Z', 'Y')

        # 应用旋转
        global_cam.rotation_mode = 'QUATERNION'
        global_cam.rotation_quaternion = rot_quat

        # 微调焦距以确保视野合适
        # 给定的焦距基础上稍微减小，以获得更大的视野
        camera_data.lens = camera_data.lens * 0.9

        # 设为活动相机
        bpy.context.scene.camera = global_cam

        # 打印调试信息
        print(f"Scene bounds: min={min_coord}, max={max_coord}")
        print(f"Scene center: {center}")
        print(f"Scene size: {size}")
        print(f"Camera position: {global_cam.location}")
        print(f"Camera distance: {camera_distance}")
        print(f"Camera focal length: {camera_data.lens:.2f}mm")

        return meshes, global_cam


def main():
    # 示例相机外参（需要根据实际数据替换）
    cam_extrinsics = np.array([
        [[-0.118633, -0.990798, 0.065165, -0.093025],
         [-0.475979, -0.000851, -0.879456, 0.779975],
         [0.871419, -0.135350, -0.471497, 3.694555],
         [0.0, 0.0, 0.0, 1.0]],
        [[-0.800171, -0.596351, 0.063965, -0.017410],
         [-0.254180, 0.240575, -0.936758, 0.567165],
         [0.543249, -0.765826, -0.344082, 5.478421],
         [0.0, 0.0, 0.0, 1.0]],
        [[-0.898062, 0.439740, -0.010705, 1.301304],
         [0.160393, 0.304710, -0.938843, 0.858324],
         [-0.409585, -0.844856, -0.344179, 5.620791],
         [0.0, 0.0, 0.0, 1.0]],
        [[-0.998685, -0.029209, -0.042144, 0.526303],
         [0.026747, 0.404480, -0.914156, 0.793217],
         [0.043748, -0.914080, -0.403166, 4.604659],
         [0.0, 0.0, 0.0, 1.0]],
        [[0.999700, 0.010259, 0.022240, -0.449803],
         [0.024491, -0.410265, -0.911637, 0.956946],
         [-0.000228, 0.911909, -0.410393, 4.321124],
         [0.0, 0.0, 0.0, 1.0]],
        [[0.047894, 0.998726, 0.015903, -0.684285],
         [0.521832, -0.011442, -0.852972, 0.596665],
         [-0.851703, 0.049151, -0.521715, 4.234497],
         [0.0, 0.0, 0.0, 1.0]],
        [[0.746017, 0.665191, 0.031307, -0.248103],
         [0.231386, -0.214845, -0.948843, 1.099851],
         [-0.624435, 0.715096, -0.314194, 5.600614],
         [0.0, 0.0, 0.0, 1.0]],
        [[0.928410, -0.371383, 0.011403, 1.298628],
         [-0.109671, -0.303229, -0.946586, 0.936235],
         [0.355004, 0.877569, -0.322251, 5.024762],
         [0.0, 0.0, 0.0, 1.0]]
    ])

    # 创建可视化器
    visualizer = SMPLXMultiViewVisualizer(
        views=views,
        root_dir=f'/Users/ru/Desktop/GYM/3d-Visual/result/',
        cam_extrinsics=cam_extrinsics
    )

    # 设置场景
    meshes, global_cam = visualizer.setup_scene()

    # 渲染结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_img=os.path.join(OUTPUT_DIR, f'{frame_id}.png')
    visualizer.render_scene(output_img)


if __name__ == "__main__":
    main()