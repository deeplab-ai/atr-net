# -*- coding: utf-8 -*-
"""Extract features with PyTorch ResNet-101."""

import cv2
import numpy as np
import torch
from torch import nn
from scipy.misc import imread

from config import USE_CUDA, PATHS
from src.tools.coco_classes import coco_classes
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.faster_rcnn.resnet import resnet

CLASSES = np.asarray([
    coco_classes[c] for c in sorted(list(coco_classes.keys()))])
USE_CUDA = USE_CUDA and torch.cuda.is_available()


def get_image_blob(img_filename, im_scale):
    """
    Convert an image into a network input.

    Inputs:
        - img_filename: str, image filename
        - im_scale: float, image scaling to match a specific size
    Returns:
        - blob: array, a data blob holding an image pyramid
    """
    # Read and transform image to BGR
    img = np.array(imread(img_filename))
    if len(img.shape) == 2:
        img = np.stack((img, img, img), axis=2)
    img = img[:, :, ::-1]  # rgb -> bgr

    # Preprocess image
    img = img.astype(np.float32, copy=True)
    img -= cfg.PIXEL_MEANS

    # Rescale
    img = cv2.resize(
        img, None, None, fx=im_scale, fy=im_scale,
        interpolation=cv2.INTER_LINEAR)
    zeros = np.zeros((cfg.TEST.MAX_SIZE, cfg.TEST.MAX_SIZE, 3), np.float32)
    zeros[:img.shape[0], :img.shape[1], :img.shape[2]] = img
    return zeros


def _load():
    """Load a feature extractor."""
    cfg_from_file(PATHS['faster_rcnn_path'] + 'res101_ls.yml')
    cfg_from_list([
        'ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'
    ])

    cfg.USE_GPU_NMS = USE_CUDA
    cfg.CUDA = USE_CUDA
    cfg.TRAIN.BATCH_SIZE = 256
    cfg.MAX_NUM_GT_BOXES = 64
    cfg.USE_GPU_NMS = USE_CUDA
    cfg.CUDA = USE_CUDA

    load_name = PATHS['faster_rcnn_path'] + "faster_rcnn_1_10_14657.pth"
    print("load checkpoint %s" % (load_name))
    if USE_CUDA:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name,
                                map_location=lambda storage, loc: storage)
    return checkpoint


class BaseFeatureExtractor(nn.Module):
    """Extract ResNet RoI pooling features."""

    def __init__(self):
        """Initialize class using checkpoint."""
        super().__init__()
        checkpoint = _load()
        _net = resnet(CLASSES, 101, class_agnostic=False)
        _net.create_architecture()
        _net.load_state_dict(checkpoint['model'])
        if USE_CUDA:
            _net.cuda()
        _net.eval()
        for param in _net.parameters():
            param.requires_grad = False
        self.base_net = _net.RCNN_base

    def forward(self, im_blob):
        """Forward pass."""
        return self.base_net(im_blob.permute(0, 3, 1, 2))
