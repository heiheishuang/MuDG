import json
import os.path

import numpy as np
import cv2
from argparse import ArgumentParser
from megfile import smart_open, smart_path_join, smart_listdir, smart_exists

oss_read_root = "./datasets/waymo"
all_clip = "./data/scene_data.json"
all_item = "./data/all_multi_frames.json"
camera_list = ["camera_FRONT", "camera_FRONT_LEFT", "camera_FRONT_RIGHT"]

with smart_open(all_clip, "r") as f:
    json_all = f.readlines()

all_clip_items = []
all_items = []

for index in range(len(json_all)):
    clip = json_all[index][:-1]
    clip_path = smart_path_join(oss_read_root, clip)

    try:
        for cam_index in range(len(camera_list)):
            image_path = smart_path_join(clip_path, "images", camera_list[cam_index])
            sparse_path = smart_path_join(clip_path, "sparse", camera_list[cam_index])

            sem_dense_path = smart_path_join(clip_path, "semantic_dense", camera_list[cam_index])
            # sem_sparse_path = smart_path_join(clip_path, "semantic_sparse", camera_list[cam_index])

            # normal_dense_path = smart_path_join(clip_path, "normals_dense", camera_list[cam_index])

            depth_dense_path = smart_path_join(clip_path, "remove_hidden_points_depth_processed_depthlab", camera_list[cam_index])
            depth_sparse_path = smart_path_join(clip_path, "depth", camera_list[cam_index])

            all_images = smart_listdir(image_path)
            assert all_images == sorted(all_images)
            for img_index, image_name in enumerate(all_images):

                if img_index - 8 < 0 or img_index + 8 > len(all_images):
                    continue

                img_frames = [all_images[index] for index in range(img_index - 8, img_index + 8)]

                image_item = {
                    "dense_color_base": smart_path_join("..", image_path),
                    "sparse_color_base": smart_path_join("..", sparse_path),
                    "dense_semantic_base": smart_path_join("..", sem_dense_path),
                    # "sparse_semantic_base": smart_path_join("..", sem_sparse_path),
                    # "dense_normal_base": smart_path_join("..", normal_dense_path),
                    "dense_depth_base": smart_path_join("..", depth_dense_path),
                    "sparse_depth_base": smart_path_join("..", depth_sparse_path),
                    "frames": img_frames,
                }

                all_items.append(image_item)
        print(clip_path)

    except Exception as e:
        print("error", clip_path)
        continue

if os.path.exists(all_item):
    os.remove(all_item)
with open(all_item, 'w') as f:
    for item_data in all_items:
        f.write(str(item_data) + '\n')


val_filename_ls_path = "./data/val_multi_frames.json"
train_filename_ls_path = "./data/train_multi_frames.json"
if os.path.exists(val_filename_ls_path):
    os.remove(val_filename_ls_path)
if os.path.exists(train_filename_ls_path):
    os.remove(train_filename_ls_path)

with open(all_item, "r") as f:
    metadata = f.readlines()

for index in range(len(metadata)):
    item = metadata[index]

    if index % 200 == 0:
        with open(val_filename_ls_path, "a") as f:
            f.write(item)
    else:
        with open(train_filename_ls_path, "a") as f:
            f.write(item)

print("Done!")
