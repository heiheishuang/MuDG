import cv2
import numpy as np
import torch

from megfile import smart_open, smart_path_join

def get_color_frames(sample, image_size=(256, 256), transform=None, move_id=1):
    caption = "A photo a of driving scene."

    dense_color_base = sample["dense_color_base"]
    sparse_color_base = sample["virtual_sparse_path"]

    frames = sample["frames"]
    dense_frames_path = [smart_path_join(dense_color_base, frame) for frame in frames]
    if move_id is not None:
        sparse_frames_path = [smart_path_join(sparse_color_base, frame[:-4] + f"_{move_id}.jpg") for frame in frames]
    else:
        sparse_frames_path = [smart_path_join(sparse_color_base, frame) for frame in frames]

    dense_frames = []
    sparse_frames = []
    h, w = image_size  # 576, 1024
    for dense_frame_path, sparse_frame_path in zip(dense_frames_path, sparse_frames_path):
        dense_data = smart_open(dense_frame_path, 'rb').read()
        dense_frame = cv2.imdecode(np.frombuffer(dense_data, np.uint8), cv2.IMREAD_ANYCOLOR)
        dense_frame = cv2.cvtColor(dense_frame, cv2.COLOR_BGR2RGB)
        dense_frame = cv2.resize(dense_frame, (w, h), interpolation=cv2.INTER_LINEAR)
        dense_frames.append(dense_frame)

        sparse_data = smart_open(sparse_frame_path, 'rb').read()
        sparse_frame = cv2.imdecode(np.frombuffer(sparse_data, np.uint8), cv2.IMREAD_ANYCOLOR)
        sparse_frame = cv2.cvtColor(sparse_frame, cv2.COLOR_BGR2RGB)
        sparse_frame = cv2.resize(sparse_frame, (w, h), interpolation=cv2.INTER_NEAREST)
        sparse_frames.append(sparse_frame)

    dense_frames = np.stack(dense_frames, axis=0)
    sparse_frames = np.stack(sparse_frames, axis=0)

    ## change the first frame of the sparse_frames
    sparse_frames[0] = dense_frames[0]

    ## process data
    dense_frames = torch.tensor(dense_frames).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]
    sparse_frames = torch.tensor(sparse_frames).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]

    if transform is not None:
        frames = torch.cat([dense_frames, sparse_frames], dim=0)
        frames = transform(frames)
        dense_frames = frames[:3]
        sparse_frames = frames[3:]

    ## turn frames tensors to [-1, 1]
    dense_frames = (dense_frames / 255 - 0.5) * 2
    sparse_frames = (sparse_frames / 255 - 0.5) * 2

    data = {
        'dense_frames': dense_frames,
        "sparse_frames": sparse_frames,
        'caption': caption,
        'fps': torch.tensor([10]),
        'class_label': torch.tensor([0])
    }

    return data


def get_sparse_depth(sample, image_size=(256, 256), transform=None, move_id=1):
    sparse_depth_base = sample["virtual_depth_path"]

    frames = sample["frames"]
    if move_id is not None:
        depth_frames_path = [smart_path_join(sparse_depth_base, frame[:-4] + f"_{move_id}.npy") for frame in frames]
    else:
        depth_frames_path = [smart_path_join(sparse_depth_base, frame[:-4] + ".npy") for frame in frames]

    depth_frames = []
    h, w = image_size  # 576, 1024

    for depth_path in depth_frames_path:
        depth_data = np.load(smart_open(depth_path, 'rb'), encoding='bytes', allow_pickle=True)
        depth_data = cv2.resize(depth_data, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
        depth_data = np.repeat(depth_data, 3, axis=2)
        depth_frames.append(depth_data)

    depth_frames = np.stack(depth_frames, axis=0)

    sparse_depths = torch.tensor(depth_frames).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]

    if transform is not None:
        sparse_depths = transform(sparse_depths)

    ## turn frames tensors to [-1,1]
    sparse_depths = torch.clamp(sparse_depths, 0, 100)
    sparse_depths = (sparse_depths / 100 - 0.5) * 2

    return sparse_depths


def get_depth_frames(sample, image_size=(256, 256), transform=None, move_id=1):
    caption = "A photo a of driving scene."

    dense_color_base = sample["dense_color_base"]
    sparse_color_base = sample["virtual_sparse_path"]

    frames = sample["frames"]
    dense_frames_path = [smart_path_join(dense_color_base, frame) for frame in frames]
    if move_id is not None:
        sparse_frames_path = [smart_path_join(sparse_color_base, frame[:-4] + f"_{move_id}.jpg") for frame in frames]
    else:
        sparse_frames_path = [smart_path_join(sparse_color_base, frame) for frame in frames]

    dense_frames = []
    sparse_frames = []
    h, w = image_size  # 576, 1024
    for dense_frame_path, sparse_frame_path in zip(dense_frames_path, sparse_frames_path):
        dense_data = smart_open(dense_frame_path, 'rb').read()
        dense_frame = cv2.imdecode(np.frombuffer(dense_data, np.uint8), cv2.IMREAD_ANYCOLOR)
        dense_frame = cv2.cvtColor(dense_frame, cv2.COLOR_BGR2RGB)
        dense_frame = cv2.resize(dense_frame, (w, h), interpolation=cv2.INTER_LINEAR)
        dense_frames.append(dense_frame)

        sparse_data = smart_open(sparse_frame_path, 'rb').read()
        sparse_frame = cv2.imdecode(np.frombuffer(sparse_data, np.uint8), cv2.IMREAD_ANYCOLOR)
        sparse_frame = cv2.cvtColor(sparse_frame, cv2.COLOR_BGR2RGB)
        sparse_frame = cv2.resize(sparse_frame, (w, h), interpolation=cv2.INTER_NEAREST)
        sparse_frames.append(sparse_frame)

    dense_frames = np.stack(dense_frames, axis=0)
    sparse_frames = np.stack(sparse_frames, axis=0)

    ## change the first frame of the sparse_frames
    sparse_frames[0] = dense_frames[0]

    ## process data
    dense_frames = torch.tensor(dense_frames).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]
    sparse_frames = torch.tensor(sparse_frames).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]

    if transform is not None:
        frames = torch.cat([dense_frames, sparse_frames], dim=0)
        frames = transform(frames)
        dense_frames = frames[:3]
        sparse_frames = frames[3:]

    ## turn frames tensors to [-1, 1]
    dense_frames = (dense_frames / 255 - 0.5) * 2
    sparse_frames = (sparse_frames / 255 - 0.5) * 2

    data = {
        'dense_frames': dense_frames,
        "sparse_frames": sparse_frames,
        'caption': caption,
        'fps': torch.tensor([10]),
        'class_label': torch.tensor([500])
    }

    return data

def get_semantic_frames(sample, image_size=(256, 256), transform=None, move_id=1):
    caption = "A photo a of driving scene."

    dense_color_base = sample["dense_color_base"]
    sparse_color_base = sample["virtual_sparse_path"]

    frames = sample["frames"]
    dense_frames_path = [smart_path_join(dense_color_base, frame) for frame in frames]
    if move_id is not None:
        sparse_frames_path = [smart_path_join(sparse_color_base, frame[:-4] + f"_{move_id}.jpg") for frame in frames]
    else:
        sparse_frames_path = [smart_path_join(sparse_color_base, frame) for frame in frames]

    dense_frames = []
    sparse_frames = []
    h, w = image_size  # 576, 1024
    for dense_frame_path, sparse_frame_path in zip(dense_frames_path, sparse_frames_path):
        dense_data = smart_open(dense_frame_path, 'rb').read()
        dense_frame = cv2.imdecode(np.frombuffer(dense_data, np.uint8), cv2.IMREAD_ANYCOLOR)
        dense_frame = cv2.cvtColor(dense_frame, cv2.COLOR_BGR2RGB)
        dense_frame = cv2.resize(dense_frame, (w, h), interpolation=cv2.INTER_LINEAR)
        dense_frames.append(dense_frame)

        sparse_data = smart_open(sparse_frame_path, 'rb').read()
        sparse_frame = cv2.imdecode(np.frombuffer(sparse_data, np.uint8), cv2.IMREAD_ANYCOLOR)
        sparse_frame = cv2.cvtColor(sparse_frame, cv2.COLOR_BGR2RGB)
        sparse_frame = cv2.resize(sparse_frame, (w, h), interpolation=cv2.INTER_NEAREST)
        sparse_frames.append(sparse_frame)

    dense_frames = np.stack(dense_frames, axis=0)
    sparse_frames = np.stack(sparse_frames, axis=0)

    ## change the first frame of the sparse_frames
    sparse_frames[0] = dense_frames[0]

    ## process data
    dense_frames = torch.tensor(dense_frames).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]
    sparse_frames = torch.tensor(sparse_frames).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]

    if transform is not None:
        frames = torch.cat([dense_frames, sparse_frames], dim=0)
        frames = transform(frames)
        dense_frames = frames[:3]
        sparse_frames = frames[3:]

    ## turn frames tensors to [-1, 1]
    dense_frames = (dense_frames / 255 - 0.5) * 2
    sparse_frames = (sparse_frames / 255 - 0.5) * 2

    data = {
        'dense_frames': dense_frames,
        "sparse_frames": sparse_frames,
        'caption': caption,
        'fps': torch.tensor([10]),
        'class_label': torch.tensor([1])
    }

    return data
