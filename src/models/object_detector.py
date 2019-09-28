# -*- coding: utf-8 -*-
"""Object Detection module for multiple SGG datasets."""

from time import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

from config import USE_CUDA
from model.faster_rcnn.resnet import resnet
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.roi_layers import nms
from src.utils.train_test_utils import ObjDetTrainTester
from src.tools.early_stopping_scheduler import EarlyStopping
from src.tools.feature_extractors import _load


class ObjectDetector(nn.Module):
    """
    Object Detector based on Faster-RCNN.

    Inputs:
        - num_classes: int, number of object class (does not include bg)
        - train_top: boolean, whether to train layers before classifier
        - train_rpn: boolean, whether to train region proposal network
    """

    def __init__(self, num_classes, train_top=False, train_rpn=False):
        """Initialize class using checkpoint."""
        super().__init__()
        num_classes += 1  # class 0 is now the background!
        self.mode = 'train'

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
        self.logit_layer = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )
        self.bbox_layer = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 4 * num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, base_features, im_info, gt_boxes=None):
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
        rois, rpn_cls_loss, rpn_bbox_loss = \
            self.rpn_net(base_features, im_info, gt_boxes, 0)
        rpn_loss = rpn_bbox_loss + rpn_cls_loss

        # Use ground truth boxes for refining
        rois_label = None
        if self.mode == 'train':
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = \
                self.rpn_target_net(rois, gt_boxes, None)
            rois_label = rois_label.view(-1).long()

        # RoI pooling
        pooled_features = self.roi_net(base_features, rois.view(-1, 5))

        # Feed to top layers
        pooled_features = self.top_net(pooled_features).mean(3).mean(2)

        # Compute bbox offset
        bbox_pred = self.bbox_layer(pooled_features)
        if self.mode == 'train':
            # select the corresponding columns according to roi labels
            bbox_pred = torch.gather(
                bbox_pred.view(-1, int(bbox_pred.size(1) / 4), 4), 1,
                rois_label.view(-1, 1, 1).expand(rois_label.size(0), 1, 4)
            ).squeeze(1)

        # Compute object classification scores
        scores = self.logit_layer(pooled_features)

        # Compute bbox regression L1 loss
        rcnn_bbox_loss = 0
        if self.mode == 'train':
            rcnn_bbox_loss = _smooth_l1_loss(
                bbox_pred, rois_target.view(-1, rois_target.size(2)),
                rois_inside_ws.view(-1, rois_inside_ws.size(2)),
                rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        bbox_pred = bbox_pred.view(1, rois.size(1), -1)

        if self.mode == 'test':
            scores = self.softmax(scores)
            detections = postprocess_dets(scores, bbox_pred, rois, im_info)
            scores = detections[:, 4]
            bbox_pred = detections[:, :4]
            rois_label = detections[:, 5]

        return scores, bbox_pred, rpn_loss, rcnn_bbox_loss, rois_label

    def train(self, mode=True):
        """Override train to prevent modules from being trainable."""
        nn.Module.train(self, mode=mode)
        self.rpn_net.train()  # always on train to obtain rpn_loss
        self.top_net.eval()  # always on eval to disable batch normalization


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights,
                    bbox_outside_weights, sigma=1.0, dim=[1]):
    """Bounding box regression loss."""
    sigma = sigma ** 2
    box_diff = bbox_inside_weights * (bbox_pred - bbox_targets)
    smooth_l1_sign = (torch.abs(box_diff) < 1. / sigma).detach().float()
    in_box_loss = (
        torch.pow(box_diff, 2) * (sigma / 2.) * smooth_l1_sign
        + (torch.abs(box_diff) - (0.5 / sigma)) * (1. - smooth_l1_sign))
    box_loss = bbox_outside_weights * in_box_loss
    for i in sorted(dim, reverse=True):
        box_loss = box_loss.sum(i)
    box_loss = box_loss.mean()
    return box_loss


def postprocess_dets(scores, bboxes, rois, im_info):
    """
    Postprocess detections to get meaningful results.

    Inputs:
        - scores: tensor, (N, num_classes + 1)
        - bboxes: tensor, (1, N, 4 * (num_classes + 1))
        - rois: tensor, (1, N, 5)
        - im_info: tensor, (1, 3)
    Outputs:
        - tensor (Ndets, 6), like (xmin, ymin, xmax, ymax, score, class)
    """
    num_classes = scores.shape[1]  # including bg
    use_cuda = USE_CUDA and torch.cuda.is_available()

    # Apply bounding-box regression deltas
    std = torch.FloatTensor((0.1, 0.1, 0.2, 0.2))
    std = std.cuda() if use_cuda else std
    bboxes = bboxes.view(-1, 4) * std
    bboxes = bboxes.view(1, -1, 4 * num_classes)
    bboxes = bbox_transform_inv(rois[:, :, 1:5], bboxes, 1)
    bboxes = clip_boxes(bboxes, im_info, 1)
    bboxes /= im_info[0][-1]
    bboxes = bboxes[0]  # (N, 4 * (num_classes + 1))

    # Class-wise nms
    detections = []
    for cid in range(1, num_classes):
        inds = torch.nonzero(scores[:, cid] > 0.05).view(-1)
        if inds.numel() > 0:
            cls_scores = scores[:, cid][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = bboxes[inds][:, cid * 4:(cid + 1) * 4]
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], 0.3)
            cls_dets = cls_dets[keep.view(-1).long()]  # (keep, 5)
            class_ids = torch.ones(len(cls_dets), 1) * (cid - 1)
            cls_dets = torch.cat((
                cls_dets, class_ids.cuda() if use_cuda else class_ids), dim=1)
            detections.append(cls_dets)
    return torch.cat(detections, dim=0)


class TrainTester(ObjDetTrainTester):
    """Extends ObjDetTrainTester."""

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
        scores, _, rpn_loss, rcnn_bbox_loss, rois_label = \
            self._net_forward(batch, step)
        loss = self.criterion(scores, rois_label) + rcnn_bbox_loss
        if self.config.train_rpn:
            loss = loss + rpn_loss
        return loss

    def _net_outputs(self, batch, step):
        """Get network outputs for current batch."""
        scores, bboxes, _, _, labels = self._net_forward(batch, step)
        return scores, bboxes, labels


def train_test(config):
    """Train and test a net."""
    config.logger.debug(
        'Tackling %s for %d classes' % (config.task, config.num_classes))
    net = ObjectDetector(
        num_classes=config.num_classes,
        train_top=config.train_top,
        train_rpn=True)  # config.train_rpn)
    features = {'images', 'image_info', 'object_rcnn_rois'}
    train_tester = TrainTester(net, config, features)
    classifier_params = [
        param for name, param in net.named_parameters()
        if ('logit' in name or 'bbox' in name) and param.requires_grad]
    other_net_params = [
        param for name, param in net.named_parameters()
        if not ('logit' in name or 'bbox' in name) and param.requires_grad]
    optimizer = optim.Adam(
        [
            {'params': other_net_params, 'lr': 0.0002},
            {'params': classifier_params}
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
