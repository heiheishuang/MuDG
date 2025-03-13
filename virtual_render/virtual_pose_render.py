import argparse, os, sys
import datetime, time
from omegaconf import OmegaConf
from einops import rearrange
from collections import OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import instantiate_from_config
import random

from virtual_render.eval_tools import save_virtual_color_results, save_virtual_depth_results, save_virtual_semantic_results
from virtual_render.data_tools import get_color_frames, get_sparse_depth, get_depth_frames, get_semantic_frames

from megfile import smart_open
import cv2
import numpy as np

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k, v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]] = state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model


def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def image_guided_synthesis(model, prompts, sparse_x, sparse_depth, class_label, noise_shape, n_samples=1, ddim_steps=50,
                           ddim_eta=1., unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False,
                           multiple_cond_cfg=False, timestep_spacing='uniform', guidance_rescale=0.0, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = sparse_x.shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""] * batch_size

    cond_frame_index = 0
    img = sparse_x[:, :, cond_frame_index]  # bchw
    img_emb = model.embedder(img)  ## blc
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        sparse_z = get_latent_z(model, sparse_x)  # b c t h w
        sparse_depth_z = get_latent_z(model, sparse_depth)

        kwargs.update({"sparse_x": sparse_z})
        kwargs.update({"class_label": class_label})  # color

        img_cat_cond = torch.cat([sparse_z, sparse_depth_z], dim=1)

        cond["c_concat"] = [img_cat_cond]  # b c 1 h w

    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img))  ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb, img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    for _ in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=batch_size,
                                             shape=noise_shape[1:],
                                             verbose=False,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             cfg_img=cfg_img,
                                             mask=cond_mask,
                                             x0=cond_z0,
                                             fs=fs,
                                             timestep_spacing=timestep_spacing,
                                             guidance_rescale=guidance_rescale,
                                             **kwargs
                                             )

        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def run_inference_multi(args, gpu_num, gpu_no):
    ## model config
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())

    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'Inference with {n_frames} frames')
    noise_shape = [args.bs, channels, n_frames, h, w]

    fakedir = os.path.join(args.savedir, "samples")
    fakedir_separate = os.path.join(args.savedir, "samples_separate")

    os.makedirs(fakedir_separate, exist_ok=True)

    ## prompt file setting
    assert os.path.exists(args.val_files), "Error: val file Not Found!"

    video_size = (args.height, args.width)
    transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size)])

    with open(args.val_files, "r") as f:
        metadata = f.readlines()

    # for debug
    # metadata = metadata[:24]

    index = 0
    sample_0 = eval(metadata[index])
    # dense_list, sparse_list, index_list, prompt_list, label_list, sparse_depth_list = get_color_infer_data(
    #     index,
    #     sample_0,
    #     video_size,
    #     transform
    # )

    color_data_0 = get_color_frames(sample_0, image_size=video_size, transform=transform)
    depth_data_0 = get_depth_frames(sample_0, image_size=video_size, transform=transform)
    semantic_data_0 = get_semantic_frames(sample_0, image_size=video_size, transform=transform)

    dense_list = [color_data_0["dense_frames"], depth_data_0["dense_frames"], semantic_data_0["dense_frames"]]
    sparse_list = [color_data_0["sparse_frames"], depth_data_0["sparse_frames"], semantic_data_0["sparse_frames"]]
    index_list = [index, index, index]
    prompt_list = ["A photo a of driving scene.", "A photo a of driving scene.", "A photo a of driving scene."]
    label_list = [color_data_0['class_label'], depth_data_0['class_label'], semantic_data_0['class_label']]
    sparse_depth_list = [get_sparse_depth(sample_0, image_size=video_size, transform=transform),
                         get_sparse_depth(sample_0, image_size=video_size, transform=transform),
                         get_sparse_depth(sample_0, image_size=video_size, transform=transform)]

    num_samples = len(metadata)

    start = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        color_loop_frames = []
        depth_loop_frames = []
        semantic_loop_frames = []
        while index < num_samples:
            prompts = prompt_list
            sparses = sparse_list
            denses = dense_list
            filenames = index_list
            class_labels = label_list
            sparse_depths = sparse_depth_list

            sparses = torch.stack(sparses, dim=0).to("cuda")
            denses = torch.stack(denses, dim=0).to("cuda")
            class_labels = torch.stack(class_labels, dim=0).to("cuda")
            sparse_depths = torch.stack(sparse_depths, dim=0).to("cuda")


            batch_samples = image_guided_synthesis(
                model, prompts, sparses, sparse_depths, class_labels, noise_shape, args.n_samples,
                args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale,
                args.cfg_img, args.fps, args.text_input,
                args.multiple_cond_cfg, args.timestep_spacing, args.guidance_rescale
            ) # 1 1 c t h w

            batch_samples = torch.clamp(batch_samples.float(), -1., 1.)

            old_index = index
            index = index + args.video_length // 2

            color_re_images = None
            for nn, label in enumerate(label_list):
                if label == 0:
                    save_virtual_color_results(prompts[nn], batch_samples[nn], filenames[nn], fakedir, denses, sparses,
                                               fps=args.fps, base_index=old_index, dir_name='virtual_color')

                    for re_index in range(1, args.video_length // 2 + 1):
                        re_image_data = ((batch_samples[nn, 0, :, re_index] + 1) / 2 * 255).int().cpu()
                        gt_image_data = ((denses[0, :, re_index] + 1) / 2 * 255).int().cpu()

                        all_images = torch.stack([re_image_data, gt_image_data], dim=2).view(3, args.height,
                                                                                             args.width * 2)
                        color_loop_frames.append(all_images)

                    color_re_images = []
                    for re_index in range(args.video_length // 2, args.video_length):
                        re_image_data = batch_samples[nn, 0, :, re_index].cpu()
                        color_re_images.append(re_image_data)

                    if index >= num_samples:
                        continue

                    sample_i = eval(metadata[index])
                    color_re_images = torch.stack(color_re_images).permute(1, 0, 2, 3)  # [c,t,h,w]
                    color_data_i = get_color_frames(sample_i, image_size=video_size, transform=transform)
                    color_data_i["sparse_frames"][:, 0:args.video_length // 2] = color_re_images[0: args.video_length // 2]
                    color_data_i["sparse_frames"][:, 0] = color_data_i["dense_frames"][:, 0]

                elif label == 500:
                    save_virtual_depth_results(prompts[nn], batch_samples[nn], filenames[nn], fakedir, denses, sparses,
                                               fps=args.fps, base_index=old_index, is_virtual=True, dir_name='virtual_depth')

                    sample_path = os.path.join(fakedir.replace('samples', 'virtual_depth'))
                    for re_index in range(1, args.video_length // 2 + 1):
                        re_image_path = os.path.join(sample_path, f'color_re_{re_index + old_index}.png')
                        re_image_data = smart_open(re_image_path, 'rb').read()
                        re_image_data = cv2.imdecode(np.frombuffer(re_image_data, np.uint8), cv2.IMREAD_ANYCOLOR)
                        re_image_data = cv2.cvtColor(re_image_data, cv2.COLOR_BGR2RGB)
                        re_image_data = cv2.resize(re_image_data, (args.width, args.height),
                                                   interpolation=cv2.INTER_LINEAR)
                        re_image_data = torch.tensor(re_image_data).permute(2, 0, 1).float()  # [h,w,c] -> [c,h,w]

                        gt_image_path = os.path.join(sample_path, f'color_gt_{re_index + old_index}.png')
                        gt_image_data = smart_open(gt_image_path, 'rb').read()
                        gt_image_data = cv2.imdecode(np.frombuffer(gt_image_data, np.uint8), cv2.IMREAD_ANYCOLOR)
                        gt_image_data = cv2.cvtColor(gt_image_data, cv2.COLOR_BGR2RGB)
                        gt_image_data = cv2.resize(gt_image_data, (args.width, args.height),
                                                   interpolation=cv2.INTER_LINEAR)
                        gt_image_data = torch.tensor(gt_image_data).permute(2, 0, 1).float()  # [h,w,c] -> [c,h,w]
                        all_images = torch.stack([re_image_data, gt_image_data], dim=2).view(3, args.height,
                                                                                             args.width * 2)
                        depth_loop_frames.append(all_images)

                    if index >= num_samples:
                        continue

                    sample_i = eval(metadata[index])
                    depth_data_i = get_depth_frames(sample_i, image_size=video_size, transform=transform)
                    # if color_re_images is not None:
                    #     depth_data_i["sparse_frames"][:, 0:args.video_length // 2] = \
                    #         color_re_images[0: args.video_length // 2]

                elif label == 1:
                    save_virtual_semantic_results(prompts[nn], batch_samples[nn], filenames[nn], fakedir, denses, sparses,
                                                  fps=args.fps, base_index=old_index, dir_name='virtual_semantic')

                    sample_path = os.path.join(fakedir.replace('samples', 'virtual_semantic'))
                    for re_index in range(1, args.video_length // 2 + 1):
                        re_image_path = os.path.join(sample_path, f'color_re_{re_index + old_index}.png')
                        re_image_data = smart_open(re_image_path, 'rb').read()
                        re_image_data = cv2.imdecode(np.frombuffer(re_image_data, np.uint8), cv2.IMREAD_ANYCOLOR)
                        re_image_data = cv2.cvtColor(re_image_data, cv2.COLOR_BGR2RGB)
                        re_image_data = cv2.resize(re_image_data, (args.width, args.height),
                                                   interpolation=cv2.INTER_LINEAR)
                        re_image_data = torch.tensor(re_image_data).permute(2, 0, 1).float()  # [h,w,c] -> [c,h,w]

                        gt_image_path = os.path.join(sample_path, f'color_gt_{re_index + old_index}.png')
                        gt_image_data = smart_open(gt_image_path, 'rb').read()
                        gt_image_data = cv2.imdecode(np.frombuffer(gt_image_data, np.uint8), cv2.IMREAD_ANYCOLOR)
                        gt_image_data = cv2.cvtColor(gt_image_data, cv2.COLOR_BGR2RGB)
                        gt_image_data = cv2.resize(gt_image_data, (args.width, args.height),
                                                   interpolation=cv2.INTER_LINEAR)
                        gt_image_data = torch.tensor(gt_image_data).permute(2, 0, 1).float()  # [h,w,c] -> [c,h,w]
                        all_images = torch.stack([re_image_data, gt_image_data], dim=2).view(3, args.height,
                                                                                             args.width * 2)
                        semantic_loop_frames.append(all_images)

                    if index >= num_samples:
                        continue

                    sample_i = eval(metadata[index])
                    semantic_data_i = get_semantic_frames(sample_i, image_size=video_size, transform=transform)
                    # if color_re_images is not None:
                    #     semantic_data_i["sparse_frames"][:, 0:args.video_length // 2] = \
                    #         color_re_images[0: args.video_length // 2]


            if index >= num_samples:
                break

            dense_list = [color_data_i["dense_frames"], depth_data_i["dense_frames"], semantic_data_i["dense_frames"]]
            sparse_list = [color_data_i["sparse_frames"], depth_data_i["sparse_frames"], semantic_data_i["sparse_frames"]]
            index_list = [index, index, index]
            prompt_list = ["A photo a of driving scene.", "A photo a of driving scene." ,"A photo a of driving scene."]
            label_list = [color_data_i['class_label'], depth_data_i['class_label'], semantic_data_i['class_label']]
            sparse_depth_list = [get_sparse_depth(sample_i, image_size=video_size, transform=transform),
                                 get_sparse_depth(sample_i, image_size=video_size, transform=transform),
                                 get_sparse_depth(sample_i, image_size=video_size, transform=transform)]

    # save all videos -----------------------------------------
    color_loop_frames = torch.stack(color_loop_frames, dim=0)
    torchvision.io.write_video(os.path.join(args.savedir, f'color_all_compare.mp4'), color_loop_frames.permute(0, 2, 3, 1),
                               fps=args.fps, video_codec='h264', options={'crf': '10'})
    depth_loop_frames = torch.stack(depth_loop_frames, dim=0)
    torchvision.io.write_video(os.path.join(args.savedir, f'depth_all_compare.mp4'), depth_loop_frames.permute(0, 2, 3, 1),
                               fps=args.fps, video_codec='h264', options={'crf': '10'})
    semantic_loop_frames = torch.stack(semantic_loop_frames, dim=0)
    torchvision.io.write_video(os.path.join(args.savedir, f'semantic_all_compare.mp4'), semantic_loop_frames.permute(0, 2, 3, 1),
                               fps=args.fps, video_codec='h264', options={'crf': '10'})

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_files", type=str, required=True, help="val file list")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt", )
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM", )
    parser.add_argument("--ddim_eta", type=float, default=1.0,
                        help="eta for ddim sampling (0.0 yields deterministic sampling)", )
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--fps", type=int, default=10,
                        help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger  motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0,
                        help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--video_length", type=int, default=3, help="inference video length")
    parser.add_argument("--negative_prompt", action='store_true', default=False, help="negative prompt")
    parser.add_argument("--text_input", action='store_true', default=False, help="input text to I2V model or not")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False,
                        help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform",
                        help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.0,
                        help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", action='store_true', default=False,
                        help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")
    return parser


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@DynamiCrafter cond-Inference: %s" % now)
    parser = get_parser()
    args = parser.parse_args()

    seed = args.seed
    if seed < 0:
        seed = random.randint(0, 2 ** 31)
    seed_everything(seed)
    rank, gpu_num = 0, 1
    run_inference_multi(args, gpu_num, rank)
