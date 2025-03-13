import os, re
from omegaconf import OmegaConf
import logging
mainlogger = logging.getLogger('mainlogger')

import torch
import torch.nn as nn
from collections import OrderedDict

def init_workspace(name, logdir, model_config, lightning_config, rank=0):
    workdir = os.path.join(logdir, name)
    ckptdir = os.path.join(workdir, "checkpoints")
    cfgdir = os.path.join(workdir, "configs")
    loginfo = os.path.join(workdir, "loginfo")

    # Create logdirs and save configs (all ranks will do to avoid missing directory error if rank:0 is slower)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(loginfo, exist_ok=True)

    if rank == 0:
        if "callbacks" in lightning_config and 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
            os.makedirs(os.path.join(ckptdir, 'trainstep_checkpoints'), exist_ok=True)
        OmegaConf.save(model_config, os.path.join(cfgdir, "model.yaml"))
        OmegaConf.save(OmegaConf.create({"lightning": lightning_config}), os.path.join(cfgdir, "lightning.yaml"))
    return workdir, ckptdir, cfgdir, loginfo

def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None

def get_trainer_callbacks(lightning_config, config, logdir, ckptdir, logger):
    default_callbacks_cfg = {
        "model_checkpoint": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch}",
                "verbose": True,
                "save_last": False,
            }
        },
        "batch_logger": {
            "target": "callbacks.ImageLogger",
            "params": {
                "save_dir": logdir,
                "batch_frequency": 1000,
                "max_images": 4,
                "clamp": True,
            }
        },    
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                "log_momentum": False
            }
        },
        "cuda_callback": {
            "target": "callbacks.CUDACallback"
        },
    }

    ## optional setting for saving checkpoints
    monitor_metric = check_config_attribute(config.model.params, "monitor")
    if monitor_metric is not None:
        mainlogger.info(f"Monitoring {monitor_metric} as checkpoint metric.")
        default_callbacks_cfg["model_checkpoint"]["params"]["monitor"] = monitor_metric
        default_callbacks_cfg["model_checkpoint"]["params"]["save_top_k"] = 3
        default_callbacks_cfg["model_checkpoint"]["params"]["mode"] = "min"

    if 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
        mainlogger.info('Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint': {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                                                   'params': {
                                                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                                                        "filename": "{epoch}-{step}",
                                                        "verbose": True,
                                                        'save_top_k': -1,
                                                        'every_n_train_steps': 10000,
                                                        'save_weights_only': True
                                                    }
                                                }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    return callbacks_cfg

def get_trainer_logger(lightning_config, logdir, on_debug):
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
                "name": "tensorboard",
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
    }
    os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)
    default_logger_cfg = default_logger_cfgs["tensorboard"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return logger_cfg

def get_trainer_strategy(lightning_config):
    default_strategy_dict = {
        "target": "pytorch_lightning.strategies.DDPShardedStrategy"
    }
    if "strategy" in lightning_config:
        strategy_cfg = lightning_config.strategy
        return strategy_cfg
    else:
        strategy_cfg = OmegaConf.create()

    strategy_cfg = OmegaConf.merge(default_strategy_dict, strategy_cfg)
    return strategy_cfg

def load_checkpoints(model, model_cfg):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), "Error: Pre-trained checkpoint NOT found at:%s"%pretrained_ckpt
        mainlogger.info(">>> Load weights from pretrained checkpoint")

        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        try:
            if 'state_dict' in pl_sd.keys():

                if model_cfg.params.unet_config.params.in_channels == 8:
                    unet_param = model_cfg.params.unet_config.params
                    replace_unet_conv_in(pl_sd["state_dict"], unet_param, model)

                if model_cfg.params.unet_config.params.class_label_condition:
                    unet_param = model_cfg.params.unet_config.params
                    add_class_embed(pl_sd["state_dict"], unet_param, model)

                model.load_state_dict(pl_sd["state_dict"], strict=True)
                mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)
            else:
                # deepspeed
                new_pl_sd = OrderedDict()
                for key in pl_sd['module'].keys():
                    new_pl_sd[key[16:]]=pl_sd['module'][key]
                model.load_state_dict(new_pl_sd, strict=True)
        except:
            model.load_state_dict(pl_sd)
    else:
        mainlogger.info(">>> Start training from scratch")

    return model

def add_class_embed(model_param, unet_param, model):
    # replace the first layer to accept 8 in_channels
    _weight_0 = model_param['model.diffusion_model.time_embed.0.weight'].clone()  # [1280, 320]
    _bias_0 = model_param['model.diffusion_model.time_embed.0.bias'].clone()  # [1280]
    _weight_2 = model_param['model.diffusion_model.time_embed.2.weight'].clone()  # [1280, 1280]
    _bias_2 = model_param['model.diffusion_model.time_embed.2.bias'].clone()  # [1280]

    model_param['model.diffusion_model.class_embed.0.weight'] = _weight_0
    model_param['model.diffusion_model.class_embed.0.bias'] = _bias_0
    model_param['model.diffusion_model.class_embed.2.weight'] = _weight_2
    model_param['model.diffusion_model.class_embed.2.bias'] = _bias_2

    linear_0 = nn.Linear(unet_param.model_channels, 1280)
    linear_0.weight = nn.Parameter(_weight_0)
    linear_0.bias = nn.Parameter(_bias_0)

    linear_2 = nn.Linear(1280, 1280)
    linear_2.weight = nn.Parameter(_weight_2)
    linear_2.bias = nn.Parameter(_bias_2)

    model.model.diffusion_model.class_embed[0] = linear_0
    model.model.diffusion_model.class_embed[2] = linear_2

    return


def replace_unet_conv_in(model_param, unet_param, model):
    # replace the first layer to accept 8 in_channels
    _weight = model_param['model.diffusion_model.input_blocks.0.0.weight'].clone()  # [320, 8, 3, 3]
    _bias = model_param['model.diffusion_model.input_blocks.0.0.bias'].clone()  # [320]
    _weight = torch.cat([_weight, _weight[:, 4:8, :, :]], dim=1)  # [320, 12, 3, 3]
    # half the activation magnitude
    _weight *= 0.5

    # update model_param
    model_param['model.diffusion_model.input_blocks.0.0.weight'] = _weight
    model_param['model.diffusion_model.input_blocks.0.0.bias'] = _bias

    # new conv_in channel
    _n_convin_out_channel = unet_param.out_channels
    _new_conv_in = nn.Conv2d(
        12, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _new_conv_in.weight = nn.Parameter(_weight)
    _new_conv_in.bias = nn.Parameter(_bias)

    model.model.diffusion_model.input_blocks[0][0] = _new_conv_in
    logging.info("Unet conv_in layer is replaced")
    return


def set_logger(logfile, name='mainlogger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s: %(message)s"))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger