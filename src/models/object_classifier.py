# -*- coding: utf-8 -*-
"""Object Classification module for multiple SGG datasets."""

from time import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

from model.faster_rcnn.resnet import resnet
from src.utils.train_test_utils import ObjClsTrainTester
from src.tools.early_stopping_scheduler import EarlyStopping
from src.tools.feature_extractors import _load


class ObjectClassifier(nn.Module):
    """
    Object Classifier based on Faster-RCNN to benefit from RoI proposal.

    Inputs:
        - num_classes: int, number of object class (does not include bg)
        - train_top: boolean, whether to train layers before classifier
        - train_rpn: boolean, whether to train region proposal network
    """

    def __init__(self, num_classes, train_top, train_rpn):
        """Initialize class using checkpoint."""
        super().__init__()
        self._use_rpn = train_rpn

        # Load and freeze Faster-RCNN
        checkpoint = _load()
        _net = resnet(np.arange(81), 101, class_agnostic=False)
        _net.create_architecture()
        _net.load_state_dict(checkpoint['model'])
        _net.eval()
        for param in _net.parameters():
            param.requires_grad = False

        # Create architecture and unfreeze top-net's weights
        self.rpn_net = _net.RCNN_rpn
        if train_rpn:
            for name, param in self.rpn_net.named_parameters():
                if 'bn' not in name:
                    param.requires_grad = True
        self.rpn_target_net = _net.RCNN_proposal_target
        self.roi_net = _net.RCNN_roi_align
        self.top_net = _net.RCNN_top
        if train_top:
            for name, param in self.top_net.named_parameters():
                if 'bn' not in name:
                    param.requires_grad = True
        self.old_logit_layer = _net.RCNN_cls_score
        self.logit_layer = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

    def forward(self, base_features, im_info, gt_boxes):
        """
        Forward pass.

        Inputs:
            - base_features: tensor (1, channels, height, scale)
            - im_info: tensor (n_img, 3),
                each row: (scaled height, scaled width, scale)
            - gt_boxes: tensor (1, n_obj, 5) of gt rois,
                each row: [xmin, ymin, xmax, ymax, gt_label]
        """
        # Obtain rois from RPN
        rpn_loss = 0
        if self._use_rpn:
            rois, rpn_cls_loss, rpn_bbox_loss = \
                self.rpn_net(base_features, im_info, gt_boxes, 0)
            rpn_loss = rpn_bbox_loss + rpn_cls_loss

        # Use ground truth boxes for refining
        if self.training and self._use_rpn:
            rois, rois_label, _, _, _ = self.rpn_target_net(rois, gt_boxes, 0)
            rois = rois.view(-1, 5)
            rois_label = rois_label.view(-1).long()
            rois = rois[rois_label > 0]
            rois_label = rois_label[rois_label > 0] - 1
        else:
            rois = gt_boxes.view(-1, 5)
            rois = rois[rois.sum(1) > 0][:, (4, 0, 1, 2, 3)]
            rois_label = rois[:, 0].clone().long() - 1
            rois[:, 0] = 0.0

        # RoI pooling
        pooled_features = self.roi_net(base_features, rois.view(-1, 5))

        # Compute object classification scores
        top_features = self.top_net(pooled_features)
        scores = self.logit_layer(top_features.mean(3).mean(2))

        return pooled_features, top_features, scores, rpn_loss, rois_label

    def train(self, mode=True):
        """Override train to prevent modules from being trainable."""
        nn.Module.train(self, mode=mode)
        self.rpn_net.train()  # always on train to obtain rpn_loss
        self.top_net.eval()  # always on eval to disable batch normalization


class TrainTester(ObjClsTrainTester):
    """Extends ObjClsTrainTester."""

    def __init__(self, net, config, features):
        """Initialize instance."""
        super().__init__(net, config, features)

    def _net_forward(self, batch, step):
        return self.net(
            self.data_loader.get('base_features', batch, step),
            self.data_loader.get('image_info', batch, step),
            self.data_loader.get('object_rcnn_rois', batch, step))

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        _, _, scores, rpn_loss, rois_label = self._net_forward(batch, step)
        loss = self.criterion(scores, rois_label)
        if self.config.train_rpn:
            loss = loss + rpn_loss
        return loss

    def _net_outputs(self, batch, step):
        """Get network outputs for current batch."""
        _, _, scores, _, _ = self._net_forward(batch, step)
        return scores


def train_test(config):
    """Train and test a net."""
    config.logger.debug(
        'Tackling %s for %d classes' % (config.task, config.num_classes))
    net = ObjectClassifier(
        num_classes=config.num_classes,
        train_top=config.train_top,
        train_rpn=config.train_rpn)
    features = {'images', 'image_info', 'object_rcnn_rois'}
    train_tester = TrainTester(net, config, features)
    logit_params = [
        param for name, param in net.named_parameters()
        if 'logit' in name and param.requires_grad]
    other_net_params = [
        param for name, param in net.named_parameters()
        if 'logit' not in name and param.requires_grad]
    optimizer = optim.Adam(
        [
            {'params': other_net_params, 'lr': 0.0002},
            {'params': logit_params}
        ], lr=0.002, weight_decay=config.weight_decay)
    if config.use_early_stopping:
        scheduler = EarlyStopping(
            optimizer, factor=0.3, patience=1, max_decays=2)
    else:
        scheduler = MultiStepLR(optimizer, [4, 8], gamma=0.3)
    t_start = time()
    train_tester.train(
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(reduction='none'),
        scheduler=scheduler,
        epochs=30 if config.use_early_stopping else 10)
    config.logger.info('Training time: ' + str(time() - t_start))
    train_tester.net.mode = 'test'
    t_start = time()
    train_tester.test()
    config.logger.info('Test time: ' + str(time() - t_start))
