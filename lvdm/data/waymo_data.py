import os
import random
from tqdm import tqdm
import pandas as pd
from decord import VideoReader, cpu

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from megfile import smart_open, smart_path_join
import numpy as np
import cv2


class Waymo(Dataset):
    def __init__(self,
                 video_length=3,
                 resolution=[256, 512],
                 spatial_transform=None,
                 crop_resolution=None,
                 filename_ls_path=None,
                 train_labels=["color", "semantic", "depth"],
                 ):
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.filename_ls_path = filename_ls_path
        self.train_labels = train_labels
        self._load_metadata()

        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                ])
            elif spatial_transform == "resize_center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                ])
            elif spatial_transform == "resize":
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        with open(self.filename_ls_path, "r") as f:
            self.metadata = f.readlines()

    def _get_semantic(self, index):
        ## get frames until success
        index = index % len(self.metadata)
        sample = eval(self.metadata[index])

        caption = "A photo a of driving scene."


        dense_semantic_base = sample["dense_semantic_base"]
        sparse_semantic_base = sample["sparse_color_base"]
        dense_color_base = sample["dense_color_base"]

        frames = sample["frames"]
        dense_frames_path = [smart_path_join(dense_semantic_base, frame) for frame in frames]
        sparse_frames_path = [smart_path_join(sparse_semantic_base, frame) for frame in frames]
        color_frames_path = [smart_path_join(dense_color_base, frame) for frame in frames]

        color_frames = []
        dense_frames = []
        sparse_frames = []
        h, w = self.resolution
        for dense_path, sparse_path, color_path in zip(dense_frames_path, sparse_frames_path, color_frames_path):
            dense_data = smart_open(dense_path, 'rb').read()
            dense_frame = cv2.imdecode(np.frombuffer(dense_data, np.uint8), cv2.IMREAD_ANYCOLOR)
            dense_frame = cv2.cvtColor(dense_frame, cv2.COLOR_BGR2RGB)
            dense_frame = cv2.resize(dense_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            dense_frames.append(dense_frame)

            sparse_data = smart_open(sparse_path, 'rb').read()
            sparse_frame = cv2.imdecode(np.frombuffer(sparse_data, np.uint8), cv2.IMREAD_ANYCOLOR)
            sparse_frame = cv2.cvtColor(sparse_frame, cv2.COLOR_BGR2RGB)
            sparse_frame = cv2.resize(sparse_frame, (w, h), interpolation=cv2.INTER_NEAREST)
            sparse_frames.append(sparse_frame)

            color_data = smart_open(color_path, 'rb').read()
            color_frame = cv2.imdecode(np.frombuffer(color_data, np.uint8), cv2.IMREAD_ANYCOLOR)
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            color_frame = cv2.resize(color_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            color_frames.append(color_frame)

        dense_frames = np.stack(dense_frames, axis=0)
        sparse_frames = np.stack(sparse_frames, axis=0)
        color_frames = np.stack(color_frames, axis=0)

        ## change the first frame of the sparse_frames
        sparse_frames[0] = color_frames[0]

        ## process data
        dense_frames = torch.tensor(dense_frames).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]
        sparse_frames = torch.tensor(sparse_frames).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]

        if self.spatial_transform is not None:
            frames = torch.cat([dense_frames, sparse_frames], dim=0)
            frames = self.spatial_transform(frames)
            dense_frames = frames[:3]
            sparse_frames = frames[3:6]

        if self.resolution is not None:
            assert (dense_frames.shape[2], dense_frames.shape[3]) == (self.resolution[0], self.resolution[1]), \
                f'frames={dense_frames.shape}, self.resolution={self.resolution}'

        ## turn frames tensors to [-1,1]
        dense_frames = (dense_frames / 255 - 0.5) * 2
        sparse_frames = (sparse_frames / 255 - 0.5) * 2

        data = {
            'dense_frames': dense_frames,
            "sparse_frames": sparse_frames,
            'caption': caption,
            'fps': 10,
            'class_label': torch.tensor([1])
        }

        return data

    def _get_color(self, index):
        ## get frames until success
        index = index % len(self.metadata)
        sample = eval(self.metadata[index])

        caption = "A photo a of driving scene."

        dense_color_base = sample["dense_color_base"]
        sparse_color_base = sample["sparse_color_base"]

        frames = sample["frames"]
        dense_frames_path = [smart_path_join(dense_color_base, frame) for frame in frames]
        sparse_frames_path = [smart_path_join(sparse_color_base, frame) for frame in frames]

        dense_frames = []
        sparse_frames = []
        h, w = self.resolution
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

        if self.spatial_transform is not None:
            frames = torch.cat([dense_frames, sparse_frames], dim=0)
            frames = self.spatial_transform(frames)
            dense_frames = frames[:3]
            sparse_frames = frames[3:]

        if self.resolution is not None:
            assert (dense_frames.shape[2], dense_frames.shape[3]) == (self.resolution[0], self.resolution[1]), \
                f'frames={dense_frames.shape}, self.resolution={self.resolution}'

        ## turn frames tensors to [-1,1]
        dense_frames = (dense_frames / 255 - 0.5) * 2
        sparse_frames = (sparse_frames / 255 - 0.5) * 2

        data = {
            'dense_frames': dense_frames,
            "sparse_frames": sparse_frames,
            'caption': caption,
            'fps': 10,
            'class_label': torch.tensor([0])
        }

        return data

    def _get_normal(self, index):
        ## get frames until success
        index = index % len(self.metadata)
        sample = eval(self.metadata[index])

        caption = "A photo a of driving scene."

        dense_color_base = sample["dense_color_base"]
        sparse_color_base = sample["sparse_color_base"]
        dense_normal_base = sample["dense_normal_base"]

        frames = sample["frames"]
        dense_frames_path = [smart_path_join(dense_color_base, frame) for frame in frames]
        sparse_frames_path = [smart_path_join(sparse_color_base, frame) for frame in frames]
        normal_frames_path = [smart_path_join(dense_normal_base, frame[:-4] + ".npy") for frame in frames]

        color_frames = []
        sparse_frames = []
        normal_frames = []
        h, w = self.resolution
        for dense_path, sparse_path, normal_path in zip(dense_frames_path, sparse_frames_path, normal_frames_path):
            dense_data = smart_open(dense_path, 'rb').read()
            dense_frame = cv2.imdecode(np.frombuffer(dense_data, np.uint8), cv2.IMREAD_ANYCOLOR)
            dense_frame = cv2.cvtColor(dense_frame, cv2.COLOR_BGR2RGB)
            dense_frame = cv2.resize(dense_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            color_frames.append(dense_frame)

            sparse_data = smart_open(sparse_path, 'rb').read()
            sparse_frame = cv2.imdecode(np.frombuffer(sparse_data, np.uint8), cv2.IMREAD_ANYCOLOR)
            sparse_frame = cv2.cvtColor(sparse_frame, cv2.COLOR_BGR2RGB)
            sparse_frame = cv2.resize(sparse_frame, (w, h), interpolation=cv2.INTER_NEAREST)
            sparse_frames.append(sparse_frame)

            normal_data = np.load(smart_open(normal_path, 'rb'), encoding='bytes', allow_pickle=True)[0]
            normal_data = cv2.resize(normal_data, (w, h), interpolation=cv2.INTER_LINEAR)
            normal_frames.append(normal_data)

        color_frames = np.stack(color_frames, axis=0)
        sparse_frames = np.stack(sparse_frames, axis=0)
        normal_frames = np.stack(normal_frames, axis=0)

        ## change the first frame of the sparse_frames
        dense_frames = normal_frames
        sparse_frames[0] = color_frames[0]

        ## process data
        dense_frames = torch.tensor(dense_frames).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]
        sparse_frames = torch.tensor(sparse_frames).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]

        if self.spatial_transform is not None:
            frames = torch.cat([dense_frames, sparse_frames], dim=0)
            frames = self.spatial_transform(frames)
            dense_frames = frames[:3]
            sparse_frames = frames[3:6]

        if self.resolution is not None:
            assert (dense_frames.shape[2], dense_frames.shape[3]) == (self.resolution[0], self.resolution[1]), \
                f'frames={dense_frames.shape}, self.resolution={self.resolution}'

        ## turn frames tensors to [-1,1]
        # dense normal is already in [-1,1]
        sparse_frames = (sparse_frames / 255 - 0.5) * 2

        data = {
            'dense_frames': dense_frames,
            'sparse_frames': sparse_frames,
            'caption': caption,
            'fps': 10,
            'class_label': torch.tensor([1000])
        }

        return data

    def _get_depth(self, index):
        ## get frames until success
        index = index % len(self.metadata)
        sample = eval(self.metadata[index])

        caption = "A photo a of driving scene."

        dense_color_base = sample["dense_color_base"]
        sparse_color_base = sample["sparse_color_base"]
        dense_depth_base = sample["dense_depth_base"]

        frames = sample["frames"]
        dense_frames_path = [smart_path_join(dense_color_base, frame) for frame in frames]
        sparse_frames_path = [smart_path_join(sparse_color_base, frame) for frame in frames]
        depth_frames_path = [smart_path_join(dense_depth_base, frame[:-4] + ".npy") for frame in frames]

        color_frames = []
        sparse_frames = []
        depth_frames = []
        h, w = self.resolution
        for dense_path, sparse_path, depth_path in zip(dense_frames_path, sparse_frames_path, depth_frames_path):
            dense_data = smart_open(dense_path, 'rb').read()
            dense_frame = cv2.imdecode(np.frombuffer(dense_data, np.uint8), cv2.IMREAD_ANYCOLOR)
            dense_frame = cv2.cvtColor(dense_frame, cv2.COLOR_BGR2RGB)
            dense_frame = cv2.resize(dense_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            color_frames.append(dense_frame)

            sparse_data = smart_open(sparse_path, 'rb').read()
            sparse_frame = cv2.imdecode(np.frombuffer(sparse_data, np.uint8), cv2.IMREAD_ANYCOLOR)
            sparse_frame = cv2.cvtColor(sparse_frame, cv2.COLOR_BGR2RGB)
            sparse_frame = cv2.resize(sparse_frame, (w, h), interpolation=cv2.INTER_NEAREST)
            sparse_frames.append(sparse_frame)

            depth_data = np.load(smart_open(depth_path, 'rb'), encoding='bytes', allow_pickle=True)
            depth_data = cv2.resize(depth_data, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
            depth_data = np.repeat(depth_data, 3, axis=2)
            depth_frames.append(depth_data)

        color_frames = np.stack(color_frames, axis=0)
        sparse_frames = np.stack(sparse_frames, axis=0)
        depth_frames = np.stack(depth_frames, axis=0)

        ## change the first frame of the sparse_frames
        dense_frames = depth_frames
        sparse_frames[0] = color_frames[0]

        ## process data
        dense_frames = torch.tensor(dense_frames).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]
        sparse_frames = torch.tensor(sparse_frames).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]

        if self.spatial_transform is not None:
            frames = torch.cat([dense_frames, sparse_frames], dim=0)
            frames = self.spatial_transform(frames)
            dense_frames = frames[:3]
            sparse_frames = frames[3:6]

        if self.resolution is not None:
            assert (dense_frames.shape[2], dense_frames.shape[3]) == (self.resolution[0], self.resolution[1]), \
                f'frames={dense_frames.shape}, self.resolution={self.resolution}'

        ## turn frames tensors to [-1,1]
        dense_frames = torch.clamp(dense_frames, 0, 100) / 100.0
        dense_frames = (dense_frames - 0.5) * 2
        sparse_frames = (sparse_frames / 255 - 0.5) * 2

        data = {
            'dense_frames': dense_frames,
            "sparse_frames": sparse_frames,
            'caption': caption,
            'fps': 10,
            'class_label': torch.tensor([500])
        }

        return data

    def get_label(self):
        label = None

        if len(self.train_labels) == 1:
            label = self.train_labels[0]
        elif len(self.train_labels) == 2:
            label = np.random.rand()
            label = self.train_labels[0] if label > 0.5 else self.train_labels[1]
        elif len(self.train_labels) == 3:
            label = np.random.rand()
            if 0 < label < 0.25:
                if "normal" in self.train_labels:
                    label = 'normal'
                else:
                    label = "depth"
            elif 0.25 < label < 0.50:
                label = 'semantic'
            elif 0.5 < label < 1.00:
                label = 'color'

        return label

    def _get_sparse_depth(self, index):
        ## get frames until success
        index = index % len(self.metadata)
        sample = eval(self.metadata[index])

        sparse_depth_base = sample["sparse_depth_base"]

        frames = sample["frames"]
        depth_frames_path = [smart_path_join(sparse_depth_base, frame[:-4] + ".npy") for frame in frames]

        depth_frames = []
        h, w = self.resolution
        for depth_path in depth_frames_path:
            depth_data = np.load(smart_open(depth_path, 'rb'), encoding='bytes', allow_pickle=True)
            depth_data = cv2.resize(depth_data, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
            depth_data = np.repeat(depth_data, 3, axis=2)
            depth_frames.append(depth_data)

        depth_frames = np.stack(depth_frames, axis=0)

        sparse_depths = torch.tensor(depth_frames).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]

        if self.spatial_transform is not None:
            sparse_depths = self.spatial_transform(sparse_depths)

        ## turn frames tensors to [-1,1]
        sparse_depths = torch.clamp(sparse_depths, 0, 100)
        sparse_depths = (sparse_depths / 100 - 0.5) * 2

        return sparse_depths


    def __getitem__(self, index):
        label = self.get_label()

        data = None
        if label == 'color':
            data = self._get_color(index)
        elif label == 'semantic':
            data = self._get_semantic(index)
        elif label == 'normal':
            data = self._get_normal(index)
        elif label == 'depth':
            data = self._get_depth(index)

        depth_condition = self._get_sparse_depth(index)
        data["sparse_depth"] = depth_condition

        return data

    def __len__(self):
        return len(self.metadata)

