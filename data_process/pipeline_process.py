import os
import pickle
from argparse import ArgumentParser

from tools.process_lidar import save_object_from_pt, save_background_from_pt
from tools.generate_sparse import generate_dynamic_sparse, generate_virtual_dynamic_sparse
from tools.semantic_tools import convert_pfm2rgb
from datetime import datetime


if __name__ == "__main__":

    parser = ArgumentParser("Points process, merge sfm points and lidar points")
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--render_virtual", required=False, action="store_true")
    args = parser.parse_args()

    now = datetime.now()

    path = args.data_path
    pt_file = os.path.join(path, "scenario.pt")

    file = open(pt_file, "rb")
    data = pickle.load(file)
    all_frames = data['observers']['lidar_TOP']['n_frames']

    camera_name = ['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT',
                   'camera_SIDE_LEFT', 'camera_SIDE_RIGHT']

    # save obj models & background model
    # obj will be saved in path/objects, background will be saved in path/objects/background.ply
    obj_path = os.path.join(path, "objects")
    obj_info_path = os.path.join(path, "objects_info.pkl")
    os.makedirs(obj_path, exist_ok=True)
    save_object_from_pt(path, pt_file, start=0, end=all_frames - 1, voxel_size=-1)
    save_background_from_pt(path, pt_file, obj_info_path=obj_info_path, voxel_size=-1)

    # save sparse image
    # print("Render sparse from lidar points")
    sparse_camera_name = ['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT']
    generate_dynamic_sparse(path, sparse_camera_name)

    # save virtual sparse image
    if args.render_virtual:
        print("Render virtual from lidar points")
        generate_virtual_dynamic_sparse(path, sparse_camera_name)

    # process semantic
    print("Process semantic map")
    semantic_path = os.path.join(path, "semantic")
    semantic_dense_path = os.path.join(path, "semantic_dense")
    convert_pfm2rgb(semantic_path, semantic_dense_path, sparse_camera_name)

    print("Merge points success!")
    print("Time taken: ", datetime.now() - now)




