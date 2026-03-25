"""
Utility functions for logging and wandb management.

This module provides functions to parse configuration files, create directories,
set random seeds, and manage logging with wandb. It includes a Logger class that
handles logging of metrics, model checkpoints, and video recordings during training
and evaluation.
"""

import os
import shutil
import uuid
import json
import random

import torch
import numpy as np

import torch.nn as nn
from typing import Callable

import timm
import wandb

from PIL import Image
from omegaconf import OmegaConf


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def freeze_all(model: nn.Module):
    """
    Freeze all parameters in the model.

    Args:
        model (nn.Module): The model to freeze.
    """

    for p in model.parameters():
        p.requires_grad = False


def set_trainable(module: nn.Module, trainable: bool = True):
    """
    Set the requires_grad attribute of all parameters in the module.

    Args:
        module (nn.Module): The module to modify.
        trainable (bool): If True, set requires_grad to True, else False.
    """

    for p in module.parameters():
        p.requires_grad = trainable


def _get_head_module(model: nn.Module):
    """
    Get the head module of a model, if it exists.
    """
    # timm uses get_classifier() for the final head
    if hasattr(model, "get_classifier"):
        cls = model.get_classifier()
        if isinstance(cls, nn.Module):
            return cls
    # fallbacks
    for attr in ("head", "classifier", "fc", "mlp_head"):
        if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Module):
            return getattr(model, attr)
    return None


def unfreeze_head_and_last_norm(model: nn.Module):
    head = _get_head_module(model)
    if head is not None:
        set_trainable(head, True)

    # DINOv2 ViTs usually have a final norm at model.norm or model.fc_norm
    for attr in ("norm", "fc_norm"):
        if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Module):
            set_trainable(getattr(model, attr), True)


def get_vit_backbone(name: str, weights=None, ft: bool=True) -> nn.Module:
    """
    Get a vision backbone model from timm.

    Args:
        name (str): Name of the model to load from timm.
        weights: Pretrained weights to load. If None, uses default pretrained weights.
        ft (bool): If True, the model is set up for fine-tuning.
    """
    # create the rgb model
    rgb_model = timm.create_model(name, pretrained=True)

    if 'resnet' in name:
        rgb_model.fc = torch.nn.Identity()

        # print trainable vs total parameters
        total_params = sum(p.numel() for p in rgb_model.parameters())
        trainable_params = sum(p.numel() for p in rgb_model.parameters() if p.requires_grad)

        print(f"======================= Parameter Report of ResNet Backbone =======================")
        print(f"ResNet: Total parameters: {total_params:,} | Trainable parameters: {trainable_params:,}")
        print(f"===================================================================================")

    elif 'dinov2' in name:
        if ft:
            freeze_all(rgb_model)
            unfreeze_head_and_last_norm(rgb_model)

        # print trainable vs total parameters
        total_params = sum(p.numel() for p in rgb_model.parameters())
        trainable_params = sum(p.numel() for p in rgb_model.parameters() if p.requires_grad)

        print(f"======================= Parameter Report of Dino Backbone =======================")
        print(f"DinoV2: Total parameters: {total_params:,} | Trainable parameters: {trainable_params:,}")
        print(f"=================================================================================")

    return rgb_model


def parse_cfg(cfg_path: str) -> OmegaConf:
    """Parses a config file and returns an OmegaConf object."""
    base = OmegaConf.load(cfg_path)
    cli = OmegaConf.from_cli()
    for k,v in cli.items():
        if v == None:
            cli[k] = True
    base.merge_with(cli)
    return base


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def set_seed(seed: int):
    """Set random seeds for PyTorch, NumPy, and Python's random module for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Logger:
    """Primary logger object. Logs in wandb."""
    def __init__(self, log_dir, cfg, config=None):
        """
        Initializes the Logger object.
        This sets up the logging directories for metrics, models,
        and initializes wandb for logging.

        Args:
            log_dir (str): The directory where logs will be stored.
            cfg (OmegaConf): Configuration object containing logging parameters.
        """
        self._log_dir = make_dir(log_dir)
        self._cfg = cfg

        # date and time based uuid
        self._uuid = f"{uuid.uuid4()}"

        self._model_dir = make_dir(self._log_dir / 'models' / self._uuid)
        self._metrics_dir = self._model_dir  # save metrics in models dir

        # save config file to models directory
        if config is not None:
            shutil.copy(config, self._model_dir / 'config.yaml')

        # initialize wandb
        wandb.init(
            config=OmegaConf.to_container(cfg),
            project=cfg.project,
            group=cfg.group,
            name=cfg.exp_name,
            id=self._uuid,
            mode=cfg.wandb_mode,
            dir=self._log_dir
        )
        self._wandb = wandb

    def log(self, d: dict, category: str):
        """
        Log a dictionary of metrics to wandb and print them to the console.

        Args:
            d (dict): Dictionary containing metrics to log.
            category (str): Category of the metrics, either 'train' or 'eval'.
        """
        assert category in ['train', 'eval']
        assert 'step' in d

        print(f"[{d['step']}]", " | ".join(f"{k} {v:.6f}" for k, v in d.items() if k != 'step'))

        with (self._metrics_dir / f"{self._cfg.exp_name}.jsonl").open("a") as f:
            f.write(json.dumps({"step": d['step'], **d}) + "\n")
        _d = dict()

        for k, v in d.items():
            _d[category + "/" + k] = v

        self._wandb.log(_d, step=d['step'])

    def save_agent(self, agent=None, identifier='final'):
        if agent:
            fp = self._model_dir / f'model_{str(identifier)}.pt'
            agent.save(fp)

    def finish(self):
        """
        Finish the wandb run. This should be called at the end of training or evaluation.
        """
        if self._wandb:
            self._wandb.finish()


def crop_resize(img, is_depth=False, output_size=(224, 224), output_dtype=None):
    """
    Crop and resize images -> center crop to aspect ratio -> resize (like apply_transform).

    Args:
        img (PIL.Image or np.ndarray): Input image.
        is_depth (bool): If True, use nearest-neighbor interpolation (for depth maps).
        output_size (tuple): Target (width, height).
        output_dtype (np.dtype): Optionally cast the result to this dtype.

    Returns:
        np.ndarray: Cropped and resized image as numpy array.
    """

    if is_depth:
        # squeeze the last dimension if depth
        img = img.squeeze()

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if is_depth:
        img = img.convert("F")

    # get dimensions
    width, height = img.size
    min_dim = min(width, height)

    # define center crop box
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    # choose interpolation method
    method = Image.NEAREST if is_depth else Image.BILINEAR

    # crop and resize
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size, method)

    # convert back to numpy
    img_np = np.array(img)

    # optional dtype cast
    if output_dtype is not None and img_np.dtype != np.dtype(output_dtype):
        img_np = img_np.astype(output_dtype)

    return img_np