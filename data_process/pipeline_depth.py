import os

import numpy as np
import open3d as o3d
import cv2
import pyrender
import pickle

from argparse import ArgumentParser

from tools.process_lidar import trans_local2global, get_color_from_camera
from tools.process_lidar import segment_object_pcd

from tools.generate_sparse import process_obj_info, merge_all_obj

def load_lidar_data(base_path, obj_info_path):

    # load the lidar data which is removed the object points

    file = open(os.path.join(base_path, "scenario.pt"), "rb")
    data = pickle.load(file)

    lidar_dir = os.path.join(base_path, "lidars")
    sensor = "lidar_TOP"
    lidar = data['observers'][sensor]

    all_frames = data['observers']['lidar_TOP']['n_frames']
    obj_info = pickle.load(open(obj_info_path, "rb"))

    points = []
    colors = []

    for index in range(0, all_frames):

        lidar_path = os.path.join(lidar_dir, "lidar_TOP", "{:08}".format(index) + ".npz")
        lidar_data = np.load(lidar_path)
        rays_o = lidar_data['rays_o']
        rays_d = lidar_data['rays_d']
        ranges = lidar_data['ranges']

        l2w = lidar['data']['l2w'][index]
        rays_o, rays_d, ranges = trans_local2global(rays_o, rays_d, ranges, l2w, offset=None)

        xyz = rays_o + rays_d * ranges[:, np.newaxis]  # (N, 3)
        cls, mask = get_color_from_camera(xyz, index, data['observers'], base_path)
        cls = cls[mask]
        xyz = xyz[mask]

        # remove the object points
        for obj in obj_info:
            if obj["visibility"][index] == 0:
                continue
            mask_frames, _ = segment_object_pcd(obj["bbox"][index], obj["transform_obj"][index], xyz)
            xyz = xyz[~mask_frames]
            cls = cls[~mask_frames]

        points.append(xyz)
        colors.append(cls)

    return points, colors


def get_6frames_lidar(index, points, colors):
    all_frames = len(points)

    indexs = [index - 2, index - 1, index, index + 1, index + 2, index + 3]
    indexs = [i if i >= 0 and i < all_frames else index for i in indexs]

    merge_p = []
    merge_c = []
    for i in indexs:
        merge_p.append(points[i])
        merge_c.append(colors[i])

    return np.concatenate(merge_p, axis=0), np.concatenate(merge_c, axis=0)


def generate_sparse_depth_data(base_path, sparse_dir, depth_dir, points, colors, cameras=['camera_FRONT']):
    file = open(os.path.join(base_path, "scenario.pt"), "rb")
    data = pickle.load(file)

    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    observers = data['observers']

    obj_info_path = os.path.join(base_path, "objects_info.pkl")
    obj_info = pickle.load(open(obj_info_path, "rb"))
    obj_vis = process_obj_info(obj_info)

    all_frames = data['observers']['lidar_TOP']['n_frames']

    cameras = [(name, observers[name]) for name in cameras]
    for camera_name, camera in cameras:
        camera_path = os.path.join(sparse_dir, camera_name)
        depth_path = os.path.join(depth_dir, camera_name)
        os.makedirs(camera_path, exist_ok=True)
        os.makedirs(depth_path, exist_ok=True)

        for index in range(all_frames):
            lidar_xyz, lidar_rgb = get_6frames_lidar(index, points, colors)

            height = int(camera['data']['hw'][index][0])
            width = int(camera['data']['hw'][index][1])
            K = camera['data']['intr'][index]  # [3, 3]
            c2w = camera['data']['c2w'][index]

            print("current_frame: ", index)

            # merge background and object points
            obj_xyz, obj_rgb = merge_all_obj(obj_info, obj_vis, frame=index)

            lidar_xyz = np.concatenate([lidar_xyz, obj_xyz], axis=0)
            lidar_rgb = np.concatenate([lidar_rgb / 255.0 , obj_rgb], axis=0)


            ## remove hidden points part
            cam_pose = c2w[:, 3]

            pc_lidar_i_cam_ori = o3d.geometry.PointCloud()
            pc_lidar_i_cam_ori.points = o3d.utility.Vector3dVector(lidar_xyz)
            pc_lidar_i_cam_ori.colors = o3d.utility.Vector3dVector(lidar_rgb)

            _, pt_map = pc_lidar_i_cam_ori.hidden_point_removal([cam_pose[0], cam_pose[1], cam_pose[2]], 100000)
            pc_lidar_i_cam_ori = pc_lidar_i_cam_ori.select_by_index(pt_map)

            lidar_xyz = np.asarray(pc_lidar_i_cam_ori.points)
            lidar_rgb = np.asarray(pc_lidar_i_cam_ori.colors)

            ## remove hidden points part

            py_camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.0001, zfar=200)
            nc = pyrender.Node(camera=py_camera, matrix=np.eye(4))
            scene = pyrender.Scene(ambient_light=[.0, .0, .0], bg_color=[0.0, 0.0, 0.0])
            mesh = pyrender.Mesh.from_points(lidar_xyz, lidar_rgb)
            scene.add(mesh)
            scene.add_node(nc)

            render_generator = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=2)
            T_cv2gl = np.eye(4, dtype=np.float32)
            T_cv2gl[1, 1] *= -1.
            T_cv2gl[2, 2] *= -1.

            T_vc2w = c2w @ T_cv2gl
            scene.set_pose(nc, pose=T_vc2w)
            color_map, depth_map = render_generator.render(scene, flags=pyrender.constants.RenderFlags.FLAT)

            sparse_map_filename = os.path.join(camera_path, "{:08}".format(index) + ".jpg")
            sparse_map = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
            cv2.imwrite(sparse_map_filename, sparse_map)

            depth_map_path = os.path.join(depth_path, "{:08}".format(index))
            np.save(depth_map_path, depth_map)



if __name__ == "__main__":
    parser = ArgumentParser("Gernerate dense depth")
    parser.add_argument("--data_path", required=True, type=str)

    args = parser.parse_args()

    camera_name = ['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT']

    images_path = os.path.join(args.data_path, "images")
    semantic_path = os.path.join(args.data_path, "semantic")

    render_base = args.data_path
    depths_lidar_path = os.path.join(render_base, "six_frames_depth")
    depths_sparse_path = os.path.join(render_base, "six_frames_sparse")

    # load the lidar data which is removed the dynamic object points
    obj_info_path = os.path.join(render_base, "objects_info.pkl")
    points_all, colors_all = load_lidar_data(render_base, obj_info_path)

    generate_sparse_depth_data(render_base, depths_sparse_path, depths_lidar_path, points_all, colors_all, camera_name)

    print("save sparse depth Done! ")


    print("All Done!")

