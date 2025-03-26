import os
import pickle
import numpy as np
import open3d as o3d
from PIL import Image
from typing import NamedTuple

from .merge_points import store_ply, fetch_ply


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class ObjectInfo(NamedTuple):
    id: int
    class_name: str
    visibility: np.array
    bbox: np.array
    transform_obj: np.array
    point_cloud: BasicPointCloud
    ply_path: str


def trans_local2global(rays_o, rays_d, ranges, l2w, offset):
    rays_d = rays_d @ l2w[:3, :3].T  # [1, N, 3]
    rays_o = rays_o @ l2w[:3, :3].T + l2w[:3, 3]  # [1, N, 3,]

    if offset:
        rays_o -= offset
    return rays_o[0], rays_d[0], ranges[0]


def downsample_points(voxel_size, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    return np.asarray(pcd.points), np.asarray(pcd.colors)


def get_color_from_camera(xyz, frame_id, observers, path):
    """
    Get color from camera.
    """
    image_dir = os.path.join(path, "images")
    cls = np.zeros((xyz.shape[0], 3))
    all_mask = np.zeros(xyz.shape[0], dtype=bool)
    for sensor in observers.keys():
        if sensor in ['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT',
                      'camera_SIDE_LEFT', 'camera_SIDE_RIGHT']:
            c2w = observers[sensor]['data']['c2w'][frame_id]
            K = observers[sensor]['data']['intr'][frame_id]  # [3, 3]
            h = observers[sensor]['data']['hw'][frame_id][0]
            w = observers[sensor]['data']['hw'][frame_id][1]
            R_c2w = c2w[:3, :3]
            t_c2w = c2w[:3, 3]
            R_w2c = np.linalg.inv(R_c2w)
            t_w2c = -R_w2c @ t_c2w
            xyz_cam = xyz @ R_w2c.T + t_w2c  # [N, 3,]

            depth_mask = xyz_cam[:, 2] > 0

            xyz_cam = xyz_cam[:, :] / xyz_cam[:, 2, None]
            xy = xyz_cam @ K.T

            xy = xy.astype(np.int32)
            mask = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            mask = mask & depth_mask
            all_mask = all_mask | mask
            xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)

            image_path = os.path.join(image_dir, sensor, "{:08}".format(frame_id) + ".jpg")
            image = Image.open(image_path)
            image = np.array(image)

            cls[mask] = image[xy[:, 1], xy[:, 0]][mask]
    return cls, all_mask


def segment_obj_from_lidar(path, observers, transform_obj, bbox, visibility, start_f=0, end_f=99, voxel_size=0.3):
    lidar_dir = os.path.join(path, "lidars")
    sensor = "lidar_TOP"
    lidar = observers[sensor]

    obj_points = []
    obj_colors = []
    for index in range(start_f, end_f + 1):
        if visibility[index] == 0:
            # object is not visible
            continue

        lidar_path = os.path.join(lidar_dir, sensor, "{:08}".format(index) + ".npz")
        lidar_data = np.load(lidar_path)
        rays_o = lidar_data['rays_o']
        rays_d = lidar_data['rays_d']
        ranges = lidar_data['ranges']

        l2w = lidar['data']['l2w'][index]
        rays_o, rays_d, ranges = trans_local2global(rays_o, rays_d, ranges, l2w, offset=None)

        xyz = rays_o + rays_d * ranges[:, np.newaxis]  # (N, 3)
        cls, mask = get_color_from_camera(xyz, index, observers, path)
        if voxel_size > 0:
            pts, cls = downsample_points(voxel_size=voxel_size, points=xyz[mask], colors=cls[mask])
        else:
            pts = xyz[mask]
            cls = cls[mask]

        mask_frames, pts_l = segment_object_pcd(bbox[index], transform_obj[index], pts)
        obj_colors.append(cls[mask_frames])
        obj_points.append(pts_l[mask_frames])

    return obj_points, obj_colors


def segment_object_pcd(bbox, translation_obj, pcd):
    """
    Calculate the bounding box of the object.
    """
    points_w = pcd

    R_l2w = translation_obj[:3, :3]
    t_l2w = translation_obj[:3, 3]
    R_w2l = np.linalg.inv(R_l2w)
    t_w2l = -R_w2l @ t_l2w

    points_l = points_w @ R_w2l.T + t_w2l  # [N, 3,]
    mask_x = np.logical_and(points_l[:, 0] > -bbox[0] / 2, points_l[:, 0] < +bbox[0] / 2)
    mask_y = np.logical_and(points_l[:, 1] > -bbox[1] / 2, points_l[:, 1] < +bbox[1] / 2)
    # TODO: add z-axis mask from zys， reserving the points on the road
    mask_z = np.logical_and(points_l[:, 2] > -bbox[2] / 2 + 0.25, points_l[:, 2] < +bbox[2] / 2)
    obj_mask = mask_x * mask_y * mask_z
    return obj_mask, points_l


def save_object_from_pt(path, pt_file, start, end, voxel_size=0.1, verbose=False):
    obj_infos = []

    file = open(pt_file, "rb")
    data = pickle.load(file)

    all_frames = data['observers']['lidar_TOP']['n_frames']
    objects = data['objects']

    print("The scene has these objects: ")

    for k, obj in objects.items():
        obj_id = obj['id']
        class_name = obj['class_name']
        if class_name != "Pedestrian" and class_name != "Vehicle":
            continue
        segments = obj['segments']

        all_transform = np.zeros((all_frames, 4, 4))
        all_scale = np.zeros((all_frames, 3))
        all_visibility = np.zeros(all_frames)

        for segment in segments:
            start_frame = segment['start_frame']
            n_frames = segment['n_frames']
            transform = segment['data']['transform']
            scale = segment['data']['scale']

            all_transform[start_frame:start_frame + n_frames] = transform
            all_scale[start_frame:start_frame + n_frames] = scale
            all_visibility[start_frame:start_frame + n_frames] = 1

        visibility = all_visibility[start:end + 1]
        bbox = all_scale[start:end + 1]
        transform_obj = all_transform[start:end + 1]

        if not is_object_motion(transform_obj, visibility):
            continue

        obj_pcd, obj_cls = segment_obj_from_lidar(path, data['observers'], transform_obj, bbox, visibility,
                                                  voxel_size=-1)

        if obj_pcd:
            points = np.concatenate(obj_pcd, axis=0)
            colors = np.concatenate(obj_cls, axis=0)
            if voxel_size > 0:
                points, colors = downsample_points(voxel_size=voxel_size, points=points, colors=colors)

            obj_pcd = BasicPointCloud(points, colors / 255.0, np.zeros((points.shape[0], 3)))

            if len(obj_pcd.points) < 100:
                continue

            print(f"Object {obj_id}", class_name, len(obj_pcd))
            ply_path = os.path.join(path, "objects", f"{obj_id}.ply")  # world frames
            store_ply(ply_path, points, colors)

            point_cloud = {'points': points, 'colors': colors / 255.0, 'normals': np.zeros((points.shape[0], 3))}
            obj_info = {'id': obj_id, 'class_name': class_name, 'visibility': visibility, 'bbox': bbox,
                        'transform_obj': transform_obj, 'point_cloud': point_cloud, 'ply_path': ply_path}

            obj_infos.append(obj_info)

            # obj_infos.append(ObjectInfo(id=obj_id, class_name=class_name, visibility=visibility, bbox=bbox,
            #                             transform_obj=transform_obj, point_cloud=obj_pcd,
            #                             ply_path=ply_path))

    with open(os.path.join(path, "objects_info.pkl"), "wb") as f:
        pickle.dump(obj_infos, f)


def save_background_from_pt(path, pt_file, obj_info_path, voxel_size=0.1):
    file = open(pt_file, "rb")
    data = pickle.load(file)

    lidar_dir = os.path.join(path, "lidars")
    sensor = "lidar_TOP"
    lidar = data['observers'][sensor]

    obj_info = pickle.load(open(obj_info_path, "rb"))

    all_frames = data['observers']['lidar_TOP']['n_frames']

    print("Saving the background all points to a ply file. ")

    all_xyz = []
    all_cls = []

    for index in range(all_frames):
        lidar_path = os.path.join(lidar_dir, sensor, "{:08}".format(index) + ".npz")
        lidar_data = np.load(lidar_path)
        rays_o = lidar_data['rays_o']
        rays_d = lidar_data['rays_d']
        ranges = lidar_data['ranges']

        l2w = lidar['data']['l2w'][index]
        rays_o, rays_d, ranges = trans_local2global(rays_o, rays_d, ranges, l2w, offset=None)

        xyz = rays_o + rays_d * ranges[:, np.newaxis]  # (N, 3)
        cls, mask = get_color_from_camera(xyz, index, data['observers'], path)
        cls = cls[mask]
        xyz = xyz[mask]

        # remove the object points
        for obj in obj_info:
            if obj["visibility"][index] == 0:
                continue
            mask_frames, _ = segment_object_pcd(obj["bbox"][index], obj["transform_obj"][index], xyz)
            xyz = xyz[~mask_frames]
            cls = cls[~mask_frames]

        all_xyz.append(xyz)
        all_cls.append(cls)

    all_xyz = np.concatenate(all_xyz, axis=0)
    all_cls = np.concatenate(all_cls, axis=0)
    if  voxel_size > 0:
        all_xyz, all_cls = downsample_points(voxel_size=voxel_size, points=all_xyz, colors=all_cls)

    bg_pcd = BasicPointCloud(all_xyz, all_cls / 255.0, np.zeros((all_xyz.shape[0], 3)))
    ply_path = os.path.join(path, "objects", "background.ply")  # world frames
    store_ply(ply_path, bg_pcd.points, bg_pcd.colors * 255)


def is_object_motion(translations, visibilities):
    first_frame = last_frame = -1
    for index in range(len(visibilities)):
        if first_frame == -1 and visibilities[index] == 1:
            first_frame = index
        if visibilities[index] == 1:
            last_frame = index

    first_pos = translations[first_frame]
    last_pos = translations[last_frame]
    dist = np.linalg.norm(last_pos - first_pos)

    if dist > 0.5:
        return True
    else:
        return False
