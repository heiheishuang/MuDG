import os
import sys

import numpy as np

import torch
import torchvision

import PIL
from typing import List, Optional, Tuple, Union


def save_virtual_color_results(prompt, samples, filename, fakedir, gts, sparses, base_index, fps=10,
                               dir_name='virtual_samples_separate'):
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    video = samples
    savedir = fakedir

    # b,c,t,h,w
    video = video.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    n = video.shape[0]
    for i in range(n):
        grid = video[i, ...]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0)  # thwc

        sample_path = os.path.join(savedir.replace('samples', dir_name))

        os.makedirs(sample_path, exist_ok=True)

        for index in range(1, grid.shape[0]):
            result = grid[index].permute(2, 0, 1)
            dense = ((gts[i][:, index].cpu() + 1) / 2 * 255).to(torch.uint8)
            sparse = ((sparses[i][:, index].cpu() + 1) / 2 * 255).to(torch.uint8)

            torchvision.io.write_png(result, os.path.join(sample_path, f'color_re_{base_index + index}.png'), 0)
            torchvision.io.write_png(dense, os.path.join(sample_path, f'color_gt_{base_index + index}.png'), 0)
            torchvision.io.write_png(sparse, os.path.join(sample_path, f'color_sp_{base_index + index}.png'), 0)
            all_result = torch.stack([dense, result, sparse], dim=2).view(3, result.shape[1], result.shape[2] * 3)
            torchvision.io.write_png(all_result, os.path.join(sample_path, f'color_all_{base_index + index}.png'), 0)


def save_virtual_depth_results(prompt, samples, filename, fakedir, gts, sparses, base_index, fps=10, is_virtual=False,
                               dir_name='virtual_samples_separate'):
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        for i in range(n):
            grid = video[i, ...]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0)  # thwc

            sample_path = os.path.join(savedirs[idx].replace('samples', dir_name))
            depth_path = os.path.join(savedirs[idx].replace('samples', 'depth'))

            os.makedirs(sample_path, exist_ok=True)
            os.makedirs(depth_path, exist_ok=True)

            for index in range(1, grid.shape[0]):
                result_pred = torch.mean(grid[index].permute(2, 0, 1).float(), dim=0, keepdim=True) / 255  # 1 h w
                np.save(os.path.join(depth_path, f'depth_re_{base_index + index}.npy'), result_pred.numpy())

                result = torch.tensor(np.array(visualize_depth(result_pred.cpu().numpy())[0])).permute(2, 0, 1)

                gt = (torch.mean(gts[i][:, index], dim=0, keepdim=True) + 1) / 2
                depth_gt = gt.cpu().numpy()
                np.save(os.path.join(depth_path, f'depth_gt_{base_index + index}.npy'), depth_gt)

                if is_virtual:
                    dense = ((gts[i][:, index].cpu() + 1) / 2 * 255).to(torch.uint8)
                else:
                    dense = torch.tensor(np.array(visualize_depth(gt.cpu().numpy())[0])).permute(2, 0, 1)

                sparse = ((sparses[i][:, index].cpu() + 1) / 2 * 255).to(torch.uint8)

                torchvision.io.write_png(result, os.path.join(sample_path, f'color_re_{base_index + index}.png'), 0)
                torchvision.io.write_png(dense, os.path.join(sample_path, f'color_gt_{base_index + index}.png'), 0)
                torchvision.io.write_png(sparse, os.path.join(sample_path, f'color_sp_{base_index + index}.png'), 0)
                all_result = torch.stack([dense, result, sparse], dim=2).view(3, result.shape[1], result.shape[2] * 3)
                torchvision.io.write_png(all_result, os.path.join(sample_path, f'color_all_{base_index + index}.png'), 0)


def save_virtual_semantic_results(prompt, samples, filename, fakedir, gts, sparses, base_index, fps=10,
                                  dir_name='virtual_samples_separate'):
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        for i in range(n):
            grid = video[i, ...]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0)  # thwc

            sample_path = os.path.join(savedirs[idx].replace('samples', dir_name))
            os.makedirs(sample_path, exist_ok=True)
            semantic_path = os.path.join(savedirs[idx].replace('samples', 'semantic'))
            os.makedirs(semantic_path, exist_ok=True)

            for index in range(1, grid.shape[0]):
                result = grid[index].permute(2, 0, 1)
                vis_pred, semantic_pred = visualize_semantic(result, return_pt=True)
                np.save(os.path.join(semantic_path, f'semantic_re_{base_index + index}.npy'), semantic_pred.numpy())

                dense = ((gts[i][:, index].cpu() + 1) / 2 * 255).to(torch.uint8)
                vis_gt, semantic_gt = visualize_semantic(dense, return_pt=True)
                np.save(os.path.join(semantic_path, f'semantic_gt_{base_index + index}.npy'), semantic_gt.numpy())

                sparse = ((sparses[i][:, index].cpu() + 1) / 2 * 255).to(torch.uint8)

                torchvision.io.write_png(vis_pred, os.path.join(sample_path, f'color_re_{base_index + index}.png'), 0)
                torchvision.io.write_png(dense, os.path.join(sample_path, f'color_gt_{base_index + index}.png'), 0)
                torchvision.io.write_png(sparse, os.path.join(sample_path, f'color_sp_{base_index + index}.png'), 0)
                all_result = torch.stack([dense, vis_pred, sparse], dim=2).view(3, result.shape[1], result.shape[2] * 3)
                torchvision.io.write_png(all_result, os.path.join(sample_path, f'color_all_{base_index + index}.png'), 0)


def colormap(
        image: Union[np.ndarray, torch.Tensor],
        cmap: str = "Spectral",
        bytes: bool = False,
        _force_method: Optional[str] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Converts a monochrome image into an RGB image by applying the specified colormap. This function mimics the
    behavior of matplotlib.colormaps, but allows the user to use the most discriminative color maps ("Spectral",
    "binary") without having to install or import matplotlib. For all other cases, the function will attempt to use
    the native implementation.

    Args:
        image: 2D tensor of values between 0 and 1, either as np.ndarray or torch.Tensor.
        cmap: Colormap name.
        bytes: Whether to return the output as uint8 or floating point image.
        _force_method:
            Can be used to specify whether to use the native implementation (`"matplotlib"`), the efficient custom
            implementation of the select color maps (`"custom"`), or rely on autodetection (`None`, default).

    Returns:
        An RGB-colorized tensor corresponding to the input image.
    """
    if not (torch.is_tensor(image) or isinstance(image, np.ndarray)):
        raise ValueError("Argument must be a numpy array or torch tensor.")
    if _force_method not in (None, "matplotlib", "custom"):
        raise ValueError("_force_method must be either `None`, `'matplotlib'` or `'custom'`.")

    supported_cmaps = {
        "binary": [
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
        ],
        "Spectral": [  # Taken from matplotlib/_cm.py
            (0.61960784313725492, 0.003921568627450980, 0.25882352941176473),  # 0.0 -> [0]
            (0.83529411764705885, 0.24313725490196078, 0.30980392156862746),
            (0.95686274509803926, 0.42745098039215684, 0.2627450980392157),
            (0.99215686274509807, 0.68235294117647061, 0.38039215686274508),
            (0.99607843137254903, 0.8784313725490196, 0.54509803921568623),
            (1.0, 1.0, 0.74901960784313726),
            (0.90196078431372551, 0.96078431372549022, 0.59607843137254901),
            (0.6705882352941176, 0.8666666666666667, 0.64313725490196083),
            (0.4, 0.76078431372549016, 0.6470588235294118),
            (0.19607843137254902, 0.53333333333333333, 0.74117647058823533),
            (0.36862745098039218, 0.30980392156862746, 0.63529411764705879),  # 1.0 -> [K-1]
        ],
    }

    def method_matplotlib(image, cmap, bytes=False):
        import matplotlib

        arg_is_pt, device = torch.is_tensor(image), None
        if arg_is_pt:
            image, device = image.cpu().numpy(), image.device

        if cmap not in matplotlib.colormaps:
            raise ValueError(
                f"Unexpected color map {cmap}; available options are: {', '.join(list(matplotlib.colormaps.keys()))}"
            )

        cmap = matplotlib.colormaps[cmap]
        out = cmap(image, bytes=bytes)  # [?,4]
        out = out[..., :3]  # [?,3]

        if arg_is_pt:
            out = torch.tensor(out, device=device)

        return out

    def method_custom(image, cmap, bytes=False):
        arg_is_np = isinstance(image, np.ndarray)
        if arg_is_np:
            image = torch.tensor(image)
        if image.dtype == torch.uint8:
            image = image.float() / 255
        else:
            image = image.float()

        is_cmap_reversed = cmap.endswith("_r")
        if is_cmap_reversed:
            cmap = cmap[:-2]

        if cmap not in supported_cmaps:
            raise ValueError(
                f"Only {list(supported_cmaps.keys())} color maps are available without installing matplotlib."
            )

        cmap = supported_cmaps[cmap]
        if is_cmap_reversed:
            cmap = cmap[::-1]
        cmap = torch.tensor(cmap, dtype=torch.float, device=image.device)  # [K,3]
        K = cmap.shape[0]

        pos = image.clamp(min=0, max=1) * (K - 1)
        left = pos.long()
        right = (left + 1).clamp(max=K - 1)

        d = (pos - left.float()).unsqueeze(-1)
        left_colors = cmap[left]
        right_colors = cmap[right]

        out = (1 - d) * left_colors + d * right_colors

        if bytes:
            out = (out * 255).to(torch.uint8)

        if arg_is_np:
            out = out.numpy()

        return out

    if _force_method is None and torch.is_tensor(image) and cmap == "Spectral":
        return method_custom(image, cmap, bytes)

    out = None
    if _force_method != "custom":
        out = method_matplotlib(image, cmap, bytes)

    if _force_method == "matplotlib" and out is None:
        raise ImportError("Make sure to install matplotlib if you want to use a color map other than 'Spectral'.")

    if out is None:
        out = method_custom(image, cmap, bytes)

    return out


def visualize_depth(
        depth: Union[
            np.ndarray,
            torch.Tensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.Tensor],
        ],
        val_min: float = 0.0,
        val_max: float = 1.0,
        color_map: str = "Spectral",
) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
    """
    Visualizes depth maps, such as predictions of the `MarigoldDepthPipeline`.

    Args:
        depth (`Union[PIL.Image.Image, np.ndarray, torch.Tensor, List[PIL.Image.Image], List[np.ndarray],
            List[torch.Tensor]]`): Depth maps.
        val_min (`float`, *optional*, defaults to `0.0`): Minimum value of the visualized depth range.
        val_max (`float`, *optional*, defaults to `1.0`): Maximum value of the visualized depth range.
        color_map (`str`, *optional*, defaults to `"Spectral"`): Color map used to convert a single-channel
                  depth prediction into colored representation.

    Returns: `PIL.Image.Image` or `List[PIL.Image.Image]` with depth maps visualization.
    """
    if depth.ndim == 2:
        depth = depth[None, ...]

    if val_max <= val_min:
        raise ValueError(f"Invalid values range: [{val_min}, {val_max}].")

    def visualize_depth_one(img, idx=None):
        img = torch.from_numpy(img)
        if val_min != 0.0 or val_max != 1.0:
            img = (img - val_min) / (val_max - val_min)
        img = colormap(img, cmap=color_map, bytes=True)  # [H,W,3]
        img = PIL.Image.fromarray(img.cpu().numpy())
        return img

    if depth is None or isinstance(depth, list) and any(o is None for o in depth):
        raise ValueError("Input depth is `None`")

    return [visualize_depth_one(img, idx) for idx, img in enumerate(depth)]


def visualize_semantic(semantic, return_pt=False):
    # semantic: 3, H, W

    color_map = {
        0: [255, 120, 50],
        1: [255, 192, 203],
        2: [255, 255, 0],
        3: [0, 150, 245],
        4: [0, 255, 255],
        5: [255, 127, 0],
        6: [255, 0, 0],
        7: [255, 240, 150],
        8: [135, 60, 0],
        9: [160, 32, 240],
        10: [255, 0, 255],
        11: [139, 137, 137],
        12: [75, 0, 75],
        13: [150, 240, 80],
        14: [230, 230, 250],
        15: [0, 175, 0],
        16: [0, 255, 127],
        17: [222, 155, 161],
        18: [140, 62, 69],
    }

    color_map_array = np.array(list(color_map.values()))  # 21, 3
    semantic = semantic.unsqueeze(0).repeat(19, 1, 1, 1).permute(2, 3, 0, 1)  # H, W, 21, 3

    distance = np.linalg.norm(semantic - color_map_array, axis=3)
    semantic = np.argmin(distance, axis=2)  # H, W

    lut = np.array([color_map[i] for i in range(19)], dtype=np.uint8)
    output = lut[semantic]

    if return_pt:
        output = torch.tensor(output).permute(2, 0, 1)
        semantic = torch.tensor(semantic)

    return output, semantic
