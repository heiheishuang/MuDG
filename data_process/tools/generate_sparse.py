import os
import pickle
import numpy as np
import pyrender
from copy import deepcopy

from .merge_points import fetch_ply, store_ply


def generate_dynamic_sparse(path, cameras=['camera_FRONT']):
    import cv2

    pt_file = os.path.join(path, "scenario.pt")
    data = pickle.load(open(pt_file, "rb"))

    sparse_dir = os.path.join(path, "sparse")
    depth_dir = os.path.join(path, "depth")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    bg_lidar_pcd_path = os.path.join(path, 'objects', "background.ply")
    obj_info_path = os.path.join(path, "objects_info.pkl")
    obj_info = pickle.load(open(obj_info_path, "rb"))
    obj_vis = process_obj_info(obj_info)

    observers = data['observers']

    all_frames = data['observers']['lidar_TOP']['n_frames']

    cameras = [(name, observers[name]) for name in cameras]
    for camera_name, camera in cameras:

        camera_path = os.path.join(sparse_dir, camera_name)
        depth_path = os.path.join(depth_dir, camera_name)
        os.makedirs(camera_path, exist_ok=True)
        os.makedirs(depth_path, exist_ok=True)

        for index in range(0, all_frames):
            height = int(camera['data']['hw'][index][0])
            width = int(camera['data']['hw'][index][1])
            K = camera['data']['intr'][index]  # [3, 3]
            c2w = camera['data']['c2w'][index]

            print("current_frame: ", index)

            lidar_xyz, lidar_rgb, _ = fetch_ply(bg_lidar_pcd_path)

            py_camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.0001,
                                                  zfar=200)
            nc = pyrender.Node(camera=py_camera, matrix=np.eye(4))
            scene = pyrender.Scene(ambient_light=[.0, .0, .0], bg_color=[0.0, 0.0, 0.0])
            mesh = pyrender.Mesh.from_points(lidar_xyz, lidar_rgb)
            scene.add(mesh)
            scene.add_node(nc)

            render_generator = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=2.5)
            T_cv2gl = np.eye(4, dtype=np.float32)
            T_cv2gl[1, 1] *= -1.
            T_cv2gl[2, 2] *= -1.

            T_vc2w = c2w @ T_cv2gl
            scene.set_pose(nc, pose=T_vc2w)
            bg_color_map, bg_depth_map = render_generator.render(scene, flags=pyrender.constants.RenderFlags.FLAT)

            sparse_map_filename = os.path.join(camera_path, "{:08}".format(index) + "_bg.jpg")
            bg_sparse_map = cv2.cvtColor(bg_color_map, cv2.COLOR_RGB2BGR)
            cv2.imwrite(sparse_map_filename, bg_sparse_map)

            depth_map_path = os.path.join(depth_path, "{:08}".format(index) + '_bg')
            np.save(depth_map_path, bg_depth_map)

            #### render obj
            obj_xyz, obj_rgb = merge_all_obj(obj_info, obj_vis, frame=index)
            py_camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.0001,
                                                  zfar=200)
            nc = pyrender.Node(camera=py_camera, matrix=np.eye(4))
            scene = pyrender.Scene(ambient_light=[.0, .0, .0], bg_color=[0.0, 0.0, 0.0])
            mesh = pyrender.Mesh.from_points(obj_xyz, obj_rgb)
            scene.add(mesh)
            scene.add_node(nc)

            render_generator = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=4)
            T_cv2gl = np.eye(4, dtype=np.float32)
            T_cv2gl[1, 1] *= -1.
            T_cv2gl[2, 2] *= -1.

            T_vc2w = c2w @ T_cv2gl
            scene.set_pose(nc, pose=T_vc2w)
            obj_color_map, obj_depth_map = render_generator.render(scene, flags=pyrender.constants.RenderFlags.FLAT)

            sparse_map_filename = os.path.join(camera_path, "{:08}".format(index) + "_obj.jpg")
            obj_sparse_map = cv2.cvtColor(obj_color_map, cv2.COLOR_RGB2BGR)
            cv2.imwrite(sparse_map_filename, obj_sparse_map)

            depth_map_path = os.path.join(depth_path, "{:08}".format(index) + "_obj")
            np.save(depth_map_path, obj_depth_map)

            # merge obj and bg using mask
            mask = np.all(obj_sparse_map > 0, axis=2).astype(np.uint8) * 1
            mask = cv2.dilate(mask, np.ones((5, 5)), iterations=3)

            mask_path = os.path.join(camera_path, "{:08}".format(index) + "_mask.jpg")
            cv2.imwrite(mask_path, mask * 255)

            sparse_map_filename = os.path.join(camera_path, "{:08}".format(index) + ".jpg")
            color_map = bg_color_map * (1 - mask[:, :, None]) + obj_color_map * mask[:, :, None]
            color_map = color_map.astype(np.uint8)
            color_map = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
            cv2.imwrite(sparse_map_filename, color_map)

            depth_map_path = os.path.join(depth_path, "{:08}".format(index))
            depth_map = bg_depth_map * (1 - mask) + obj_depth_map * mask
            np.save(depth_map_path, depth_map)


def generate_virtual_dynamic_sparse(path, cameras=['camera_FRONT']):
    import cv2

    pt_file = os.path.join(path, "scenario.pt")
    data = pickle.load(open(pt_file, "rb"))

    sparse_dir = os.path.join(path, "virtual_sparse")
    depth_dir = os.path.join(path, "virtual_depth")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    bg_lidar_pcd_path = os.path.join(path, 'objects', "background.ply")
    obj_info_path = os.path.join(path, "objects_info.pkl")
    obj_info = pickle.load(open(obj_info_path, "rb"))
    obj_vis = process_obj_info(obj_info)

    observers = data['observers']

    all_frames = data['observers']['lidar_TOP']['n_frames']

    cameras = [(name, observers[name]) for name in cameras]
    for camera_name, camera in cameras:

        camera_path = os.path.join(sparse_dir, camera_name)
        depth_path = os.path.join(depth_dir, camera_name)
        os.makedirs(camera_path, exist_ok=True)
        os.makedirs(depth_path, exist_ok=True)

        for index in range(0, all_frames):
            height = int(camera['data']['hw'][index][0])
            width = int(camera['data']['hw'][index][1])
            K = camera['data']['intr'][index]  # [3, 3]
            c2w = camera['data']['c2w'][index]

            print("current_frame: ", index)
            lidar_xyz, lidar_rgb, _ = fetch_ply(bg_lidar_pcd_path)

            py_camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.0001, zfar=200)

            ref_c2w = deepcopy(c2w)
            virtual_c2ws = generate_virtual_pose(ref_c2w, with_ori_pose=True)

            for virtual_index in range(1, len(virtual_c2ws)):
                c2w = virtual_c2ws[virtual_index]

                # render bg
                nc = pyrender.Node(camera=py_camera, matrix=np.eye(4))
                scene = pyrender.Scene(ambient_light=[.0, .0, .0], bg_color=[0.0, 0.0, 0.0])
                mesh = pyrender.Mesh.from_points(lidar_xyz, lidar_rgb)
                scene.add(mesh)
                scene.add_node(nc)

                render_generator = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=2.5)
                T_cv2gl = np.eye(4, dtype=np.float32)
                T_cv2gl[1, 1] *= -1.
                T_cv2gl[2, 2] *= -1.

                T_vc2w = c2w @ T_cv2gl
                scene.set_pose(nc, pose=T_vc2w)
                bg_color_map, bg_depth_map = render_generator.render(scene, flags=pyrender.constants.RenderFlags.FLAT)

                sparse_map_filename = os.path.join(camera_path,  "{:08}_{}".format(index, virtual_index) + "_bg.jpg")
                bg_sparse_map = cv2.cvtColor(bg_color_map, cv2.COLOR_RGB2BGR)
                cv2.imwrite(sparse_map_filename, bg_sparse_map)

                depth_map_path = os.path.join(depth_path,  "{:08}_{}".format(index, virtual_index) + '_bg')
                np.save(depth_map_path, bg_depth_map)

                #### render obj
                obj_xyz, obj_rgb = merge_all_obj(obj_info, obj_vis, frame=index)
                nc = pyrender.Node(camera=py_camera, matrix=np.eye(4))
                scene = pyrender.Scene(ambient_light=[.0, .0, .0], bg_color=[0.0, 0.0, 0.0])
                mesh = pyrender.Mesh.from_points(obj_xyz, obj_rgb)
                scene.add(mesh)
                scene.add_node(nc)

                render_generator = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=4)
                T_cv2gl = np.eye(4, dtype=np.float32)
                T_cv2gl[1, 1] *= -1.
                T_cv2gl[2, 2] *= -1.

                T_vc2w = c2w @ T_cv2gl
                scene.set_pose(nc, pose=T_vc2w)
                obj_color_map, obj_depth_map = render_generator.render(scene, flags=pyrender.constants.RenderFlags.FLAT)

                sparse_map_filename = os.path.join(camera_path,  "{:08}_{}".format(index, virtual_index) + "_obj.jpg")
                obj_sparse_map = cv2.cvtColor(obj_color_map, cv2.COLOR_RGB2BGR)
                cv2.imwrite(sparse_map_filename, obj_sparse_map)

                depth_map_path = os.path.join(depth_path,  "{:08}_{}".format(index, virtual_index) + "_obj")
                np.save(depth_map_path, obj_depth_map)

                # merge obj and bg using mask
                mask = np.all(obj_sparse_map > 0, axis=2).astype(np.uint8) * 1
                mask = cv2.dilate(mask, np.ones((5, 5)), iterations=3)

                mask_path = os.path.join(camera_path,  "{:08}_{}".format(index, virtual_index) + "_mask.jpg")
                cv2.imwrite(mask_path, mask * 255)

                sparse_map_filename = os.path.join(camera_path,  "{:08}_{}".format(index, virtual_index) + ".jpg")
                color_map = bg_color_map * (1 - mask[:, :, None]) + obj_color_map * mask[:, :, None]
                color_map = color_map.astype(np.uint8)
                color_map = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
                cv2.imwrite(sparse_map_filename, color_map)

                depth_map_path = os.path.join(depth_path,  "{:08}_{}".format(index, virtual_index))
                depth_map = bg_depth_map * (1 - mask) + obj_depth_map * mask
                np.save(depth_map_path, depth_map)


def process_obj_info(obj_info):
    visibilitys = []
    for obj in obj_info:
        visibility = obj['visibility']  # (183,)
        visibilitys.append(visibility)

    visibilitys = np.stack(visibilitys, axis=0)  # (N, 183)
    visibilitys = visibilitys.T  # (183, N)

    return visibilitys


def merge_all_obj(obj_info, obj_vis, frame):
    lidar_xyz = []
    lidar_rgb = []

    visibility = obj_vis[frame]  # N
    for index, vis in enumerate(visibility):
        if vis == 1:
            obj = obj_info[index]
            id = obj['id']
            transform_obj = obj['transform_obj'][frame]
            obj_xyz = obj['point_cloud']['points']
            obj_rgb = obj['point_cloud']['colors']
            obj_xyz = obj_xyz @ transform_obj[:3, :3].T + transform_obj[:3, 3]
            lidar_xyz.append(obj_xyz)
            lidar_rgb.append(obj_rgb)
            print("id: ", id, "length", len(obj_xyz))

    if not lidar_xyz or not lidar_rgb:
        return np.array([[1000, 1000, 1000]]), np.array([[0, 0, 0]])

    lidar_xyz = np.concatenate(lidar_xyz, axis=0)
    lidar_rgb = np.concatenate(lidar_rgb, axis=0)
    return lidar_xyz, lidar_rgb


def generate_virtual_pose(ori_c2w, random_shift=2.0, with_ori_pose=False):
    """基于原始c2w生成新的c2w"""
    # random_shift = np.random.uniform(low=1.0, high=3.5)  # 1m ~ 3.5m之间随机偏移
    # random_shift_dir = np.random.choice([-1.0, 1.0])  # 向左或向右偏移的方向, 目前仅考虑左右偏移
    # random_rot = np.random.uniform(low=0.0, high=30.0)  # 旋转角度
    # random_rot_flag = np.random.choice([0.0, 1.0])  # 是否旋转
    # random_rot_dir = np.random.choice([-1.0, 1.0])  # 旋转的方向, 目前仅考虑yaw角方向的旋转

    ret_vc2w = [] if not with_ori_pose else [ori_c2w]
    # shift, 这里手动的生成左右各偏移一次
    for random_shift_dir in [-1.0, 1.0]:
        vcam2cam = np.eye(4)
        vcam2cam[0, 3] += round(random_shift_dir * random_shift, 4)
        vir_c2w = ori_c2w @ vcam2cam
        ret_vc2w.append(vir_c2w)

    return ret_vc2w