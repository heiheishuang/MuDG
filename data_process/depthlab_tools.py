import argparse
import logging
import os
import random
import numpy as np
import torch
from PIL import Image
import PIL
from typing import List, Optional, Tuple, Union
from tqdm.auto import tqdm
from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
)
import cv2
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_2d_condition_main import UNet2DConditionModel_main
from src.models.projection import My_proj
from transformers import CLIPVisionModelWithProjection
from inference.depthlab_pipeline import DepthLabPipeline
from utils.seed_all import seed_all
from utils.image_util import get_filled_for_latents
import re


def read_semantic_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')

    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def process_sky(depth_path, semantic_path):
    """
    Process sky region in depth map.
    Args:
        depth_path: depth map path
        semantic_path: semantic map path
    Returns:
        depth map with sky region processed
    """

    depth = np.load(depth_path)
    semantic, _ = read_semantic_pfm(semantic_path)

    mask = semantic == 10  # sky mask
    depth[mask] = 100

    depth = np.clip(depth, 0, 100)

    vis = visualize_depth(depth / 100)

    return depth, vis


def process_black(depth_path):
    """
    Process sky region in depth map.
    Args:
        depth_path: depth map path
    Returns:
        depth map with sky region processed
    """

    depth = np.load(depth_path)
    semantic = np.load(depth_path)
    mask = semantic == 0  # sky mask

    depth = np.clip(depth, 0, 100)

    vis = visualize_depth(depth / 100)

    vis = np.array(vis)
    vis[mask] = [0, 0, 0]
    vis = PIL.Image.fromarray(vis)

    return depth, vis


def align_depth(lidar_depth, unscaled_depth):
    """
    Aligns scaled depth map with unscaled depth map.
    Args:
        lidar_depth: scaled depth map
        unscaled_depth: unscaled depth map
    Returns:
        aligned depth map
    """

    if unscaled_depth.ndim == 3:
        unscaled_depth = unscaled_depth.squeeze(0)

    # align scaled depth with unscaled depth
    mask = np.logical_and(lidar_depth > 0, unscaled_depth > 0)
    lidar_depth = lidar_depth[mask]
    unaligned_depth = unscaled_depth[mask]

    A = np.vstack([unaligned_depth, np.ones(len(unaligned_depth))]).T
    m, c = np.linalg.lstsq(A, lidar_depth, rcond=None)[0]
    aligned_depth = m * unscaled_depth + c

    return aligned_depth


def depth_to_color(depth, color_map="Spectral"):
    depth_pred_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_pred_colored = visualize_depth(
        depth_pred_norm, 0, 1, color_map=color_map
    )

    return depth_pred_colored


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
    if val_max <= val_min:
        raise ValueError(f"Invalid values range: [{val_min}, {val_max}].")

    def visualize_depth_one(img):
        img = torch.from_numpy(img)
        if val_min != 0.0 or val_max != 1.0:
            img = (img - val_min) / (val_max - val_min)
        img = colormap(img, cmap=color_map, bytes=True)  # [H,W,3]
        img = PIL.Image.fromarray(img.cpu().numpy())
        return img

    if depth is None or isinstance(depth, list) and any(o is None for o in depth):
        raise ValueError("Input depth is `None`")

    return visualize_depth_one(depth)


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


def generate_depth_map(pipe, image_path, depth_path):
    input_image = Image.open(image_path)
    depth_numpy = np.load(depth_path)

    mask = 1 - (depth_numpy > 0).astype(np.int64)

    if args.refine is not True:
        depth_numpy = get_filled_for_latents(mask, depth_numpy)

    pipe_out = pipe(
        input_image,
        denosing_steps=denoise_steps,
        processing_res=processing_res,
        match_input_res=True,
        batch_size=1,
        color_map="Spectral",
        show_progress_bar=True,
        depth_numpy_origin=depth_numpy,
        mask_origin=mask,
        guidance_scale=1,
        normalize_scale=args.normalize_scale,
        strength=args.strength,
        blend=True)

    depth_pred: np.ndarray = pipe_out.depth_np

    return depth_pred, pipe_out.depth_colored


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,  # quantitative evaluation uses 50 steps
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--normalize_scale",
        type=float,
        default=1,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='./checkpoints/marigold-depth-v1-0',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default='./checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--denoising_unet_path",
        type=str,
        default='./checkpoints/DepthLab/denoising_unet.pth',
        help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
        default='./checkpoints/DepthLab/mapping_layer.pth',
        help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--reference_unet_path",
        type=str,
        default='./checkpoints/DepthLab/reference_unet.pth',
        help="Path to depth inpainting model."
    )

    parser.add_argument(
        "--blend",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument("--data_path", required=True, type=str)

    args = parser.parse_args()
    denoise_steps = args.denoise_steps
    processing_res = args.processing_res
    seed = args.seed

    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Model --------------------

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                        subfolder='vae')
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,
                                                 subfolder='text_encoder')
    denoising_unet = UNet2DConditionModel_main.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                                               in_channels=12, sample_size=96,
                                                               low_cpu_mem_usage=False,
                                                               ignore_mismatched_sizes=True)
    reference_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                                          in_channels=4, sample_size=96,
                                                          low_cpu_mem_usage=False,
                                                          ignore_mismatched_sizes=True)
    image_enc = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    mapping_layer = My_proj()

    mapping_layer.load_state_dict(
        torch.load(args.mapping_path, map_location="cpu"),
        strict=False,
    )
    mapping_device = torch.device("cuda")
    mapping_layer.to(mapping_device)
    reference_unet.load_state_dict(
        torch.load(args.reference_unet_path, map_location="cpu"),
    )
    denoising_unet.load_state_dict(
        torch.load(args.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer')
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    pipe = DepthLabPipeline(reference_unet=reference_unet,
                            denoising_unet=denoising_unet,
                            mapping_layer=mapping_layer,
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            image_enc=image_enc,
                            scheduler=scheduler,
                            ).to('cuda')
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    # -------------------- Inference and saving --------------------

    # render waymo depth
    camera_name = ['camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT']
    data_path = args.data_path

    images_path = os.path.join(data_path, "images")
    semantic_path = os.path.join(data_path, "semantic")

    depths_lidar_path = os.path.join(data_path, "six_frames_depth")
    depths_dense_path = os.path.join(data_path, "six_frames_depth_dense")
    depths_aligned_path = os.path.join(data_path, "six_frames_depth_aligned")
    depths_vis_path = os.path.join(data_path, "six_frames_depth_vis")
    depths_processed_path = os.path.join(data_path, "six_frames_depth_processed")

    for camera in camera_name:
        camera_path = os.path.join(images_path, camera)
        image_names = sorted(os.listdir(camera_path))

        if not os.path.exists(os.path.join(depths_dense_path, camera)):
            os.makedirs(os.path.join(depths_dense_path, camera))
        if not os.path.exists(os.path.join(depths_vis_path, camera)):
            os.makedirs(os.path.join(depths_vis_path, camera))
        if not os.path.exists(os.path.join(depths_aligned_path, camera)):
            os.makedirs(os.path.join(depths_aligned_path, camera))
        if not os.path.exists(os.path.join(depths_processed_path, camera)):
            os.makedirs(os.path.join(depths_processed_path, camera))

        for image_name in image_names:
            image_path = os.path.join(camera_path, image_name)
            lidar_depth_path = os.path.join(depths_lidar_path, camera, image_name[:-4] + ".npy")


            # vis sparse depth
            _, vis = process_black(lidar_depth_path)
            vis.save(os.path.join(depths_vis_path, camera, image_name.replace(".jpg", "_sparse.png")))

            # generate depth
            depth, vis = generate_depth_map(pipe, image_path, lidar_depth_path)
            np.save(os.path.join(depths_dense_path, camera, image_name[:-4]), depth)
            vis.save(os.path.join(depths_vis_path, camera, image_name.replace(".jpg", ".png")))

            # align depth
            lidar_depth = np.load(lidar_depth_path)
            scaled_depth = align_depth(lidar_depth, depth)
            np.save(os.path.join(depths_aligned_path, camera, image_name[:-4]), scaled_depth)
            vis_aligned = depth_to_color(scaled_depth)
            vis_aligned.save(os.path.join(depths_vis_path, camera, image_name.replace(".jpg", "_aligned.png")))

            # process depth
            depth_aligned_path = os.path.join(depths_aligned_path, camera, image_name[:-4] + ".npy")
            depth_processed_path = os.path.join(depths_processed_path, camera, image_name[:-4] + ".npy")
            semantic_mask = os.path.join(semantic_path, camera, 'seg', image_name[:-4] + ".pfm")
            depth_processed, vis = process_sky(depth_aligned_path, semantic_mask)
            np.save(depth_processed_path, depth_processed)
            vis.save(os.path.join(depths_vis_path, camera, image_name.replace(".jpg", "_processed.png")))

    print("All Done!")
