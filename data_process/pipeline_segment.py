'''
multi-cam multi-process verision to perform semseg.
'''
import os, sys
import math
import time
import argparse
import torch
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool
# sys.path.append(str(Path(__file__).parent))
from third_party.SegFormer.mmseg.apis import inference_segmentor, init_segmentor
from third_party.SegFormer.mmseg.core.evaluation import get_palette
import numpy as np


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


def save_results_all(model, img, result, palette=None, save_path='./', result_name='demo.jpg'):
    """save the results (visualized and raw results) in save_path
    """

    '''
    if hasattr(model, 'module'):
        model = model.module
    img, color_seg = model.show_result(img, result, palette=palette, show=False)
    '''

    # kernel = np.ones((4,4))
    # mask_d = cv2.dilate(mask, kernel, iterations=1)
    # mask_d = mask

    # mask = result[0] >= 8  # equal to seg
    # mask = np.array(mask, dtype=np.uint8)

    '''
    # cv2.imwrite(f'{save_path}/visualize/{result_name}', mmcv.bgr2rgb(img))
    # cv2.imwrite(f'{save_path}/color_seg/{result_name}', mmcv.bgr2rgb(color_seg))
    '''
    # cv2.imwrite(f'{save_path}/mask/{result_name}', mask*255)
    save_pfm(f"{save_path}/seg/{result_name.split('.')[0]}.pfm", np.array(result[0], dtype=np.float32))


def mp(params):
    img_root, save_root, scale, id_s, id_e, config, checkpoint, device, palette, cam_name = params

    model = init_segmentor(config, checkpoint, device=device)
    img_list = sorted(os.listdir(img_root))

    t0 = time.time()
    for img in tqdm(img_list[id_s:id_e], desc=f"SemSeg-{cam_name}"):
        img_fullname = os.path.join(img_root, img)
        result = inference_segmentor(model, img_fullname)

        # save the single result
        save_results_all(model, img_fullname, result, get_palette(palette),
                         save_path=save_root, result_name=img)
    print(f"\n ({device}) Segment image-id {id_s}-{id_e - 1} cost time = {time.time() - t0} s.")


def main(input_root, save_root_, config, checkpoint, scale=1, gpu_num=1, mp_num=1, palette='cityscapes'):
    print(f"Batch SemSeg using {gpu_num} GPUs with {mp_num} process per GPU!")
    cam_list = sorted(os.listdir(input_root))

    for cam in cam_list:
        img_root = os.path.join(input_root, cam)
        save_root = os.path.join(save_root_, cam)

        assert os.path.isdir(img_root), f"{img_root} doesn't exist!"
        os.makedirs(save_root, exist_ok=True)
        os.makedirs(os.path.join(save_root, 'mask'), exist_ok=True)
        os.makedirs(os.path.join(save_root, 'seg'), exist_ok=True)

        img_list = sorted(os.listdir(img_root))
        inter_num = math.ceil(len(img_list) / (gpu_num * mp_num))

        params = []
        total_idx = 0
        for gpu_idx in range(gpu_num):
            for mp_idx in range(mp_num):
                device = f"cuda:{gpu_idx}"
                id_s, id_e = total_idx * inter_num, (total_idx + 1) * inter_num
                id_e = len(img_list) if id_e >= len(img_list) else id_e
                params.append([img_root, save_root, scale, id_s, id_e, config, checkpoint, device, palette, cam])
                total_idx += 1

        partial_func = partial(mp)
        p = Pool(gpu_num * mp_num)
        p.map(partial_func, params)
        p.close()
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch semseg images')

    parser.add_argument('--data_root',
                        type=str,
                        required=True,
                        help='root path to tf_data.')
    parser.add_argument('--resize_scale',
                        type=float,
                        default=1.0,
                        help='resize scale of the input image.')
    parser.add_argument('--config_file',
                        type=str,
                        default='./third_party/SegFormer/local_configs/segformer/B3/segformer.b3.1024x1024.city.160k.py',
                        help='path to config file.')
    parser.add_argument('--ckpt_file',
                        type=str,
                        default='./third_party/SegFormer/checkpoints/segformer.b3.1024x1024.city.160k.pth',
                        help='path to checkpoint file.')

    args = parser.parse_args()

    config_file = args.config_file
    ckpt_file = args.ckpt_file
    base_input_root = os.path.join(args.data_root, 'images')
    base_save_root = os.path.join(args.data_root, 'semantic')

    torch.multiprocessing.set_start_method('spawn')
    main(base_input_root, base_save_root, config_file, ckpt_file, scale=1, gpu_num=1, mp_num=1)
