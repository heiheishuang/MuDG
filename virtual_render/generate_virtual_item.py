import os
import cv2
import numpy as np
import tqdm
import argparse

from megfile import smart_open, smart_path_join, smart_listdir, smart_exists

oss_data_path = "./datasets/waymo"
camera_list = ["camera_FRONT"]


def generate_frames_save_list(data_path, list_name, camera_names=['camera_FRONT']):
    clip_path = smart_path_join(data_path, list_name)

    all_items = []

    for camera_name in camera_names:
        image_path = smart_path_join(clip_path, 'images', camera_name)
        sparse_path = smart_path_join(clip_path, 'sparse', camera_name)

        sem_dense_path = smart_path_join(clip_path, 'semantic_dense', camera_name)
        # sem_sparse_path = smart_path_join(clip_path, 'semantic_sparse', camera_name)

        virtual_sparse_path = smart_path_join(clip_path, 'virtual_sparse', camera_name)
        virtual_depth_path = smart_path_join(clip_path, "virtual_depth", camera_name)

        all_images = smart_listdir(image_path)
        assert all_images == sorted(all_images)

        for img_index, image_name in enumerate(all_images):

            if img_index - 8 < 0 or img_index + 8 > len(all_images):
                continue

            img_frames = [all_images[index] for index in range(img_index - 8, img_index + 8)]

            image_item = {
                "dense_color_base": image_path,
                "sparse_color_base": sparse_path,
                "dense_semantic_base": sem_dense_path,
                # "sparse_semantic_base": sem_sparse_path,
                "frames": img_frames,
                "virtual_sparse_path": virtual_sparse_path,
                "virtual_depth_path": virtual_depth_path
            }
            all_items.append(image_item)

    return all_items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate videos for rendered results')
    parser.add_argument('--scene_name', type=str,
                        default='segment-15365')
    args = parser.parse_args()

    all_item_path = "./virtual_render/virtual_data/" + args.scene_name[8:13] + "-virtual_data_frames.json"

    all_items = generate_frames_save_list(oss_data_path, args.scene_name, camera_list)

    if os.path.exists(all_item_path):
        os.remove(all_item_path)
    with smart_open(all_item_path, 'w') as f:
        for item_data in all_items:
            f.write(str(item_data) + '\n')

    print(args.scene_name, "Done!")
