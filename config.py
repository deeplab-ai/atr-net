# -*- coding: utf-8 -*-
"""Configuration parameters for each dataset and task."""

import json
import logging
from math import ceil
import os

from colorlog import ColoredFormatter
import numpy as np
import torch

# Directories
PATHS = {
    'analytics_path': 'analytics/',  # path to store various dataset analytics
    'faster_rcnn_path': 'faster_rcnn/',  # path of faster-rcnn functions
    'figures_path': 'figures/',  # path to store figures of loss curves
    'json_path': 'json_annos/',  # path of stored json annotations
    'loss_path': 'losses/',  # path to store loss history
    'models_path': 'models/',  # path to store trained models
    'results_path': 'results/'  # path to store test results
}

ORIG_ANNOS_PATHS = {  # paths of stored original dataset annotations
    'VG200': 'datasets/VG200/',
    'VG80K': 'datasets/VG80K/',
    'VGMSDN': 'datasets/VGMSDN/',
    'VGVTE': 'datasets/VGVTE/',
    'VRD': 'datasets/VRD/',
    'VrR-VG': 'datasets/VrR-VG/',
    'sVG': 'datasets/sVG/',
    'UnRel': 'datasets/UnRel/'
}

ORIG_IMAGES_PATH = {  # paths of stored original dataset images
    'VG200': 'VG/images/',
    'VG80K': 'VG/images/',
    'VGMSDN': 'VG/images/',
    'VGVTE': 'VG/images/',
    'VRD': 'VRD/images/',
    'VrR-VG': 'VG/images/',
    'sVG': 'VG/images/',
    'UnRel': 'UnRel/images/'
}

# Variables
USE_CUDA = torch.cuda.is_available()  # whether to use GPU


class Config():
    """
    A class to configure global or dataset/task-specific parameters.

    Inputs:
        - dataset: str, dataset codename, e.g. 'VRD', 'VG200' etc.
        - task: str, task codename, supported are:
            - 'preddet': Predicate Detection
            - 'predcls': Predicate Classification
            - 'sgcls': Scene Graph Classification
            - 'sggen': Scene Graph Generation
            - 'objcls': Object Classification
            - 'objdet': Object Detection
    Optional inputs:
        - filter_duplicate_rels: boolean, whether to filter relations
            annotated more than once (during training)
        - filter_multiple_preds: boolean, whether to sample a single
            predicate per object pair (during training)
        - annotations_per_batch: int, number of desired annotations
            per batch on average, in terms of relations or objects
            depending on the task
        - batch_size: int or None, batch size in images (if custom)
        - backbone: str, Faster-RCNN backbone network {'resnet', 'vgg'}
        - num_workers: int, workers employed by the data loader
        - apply_dynamic_lr: boolean, whether to adapt lr in each step
            in order to preserve lr / annotations per batch
        - use_early_stopping: boolean, whether to use a dynammic
            learning rate policy with early stopping
        - restore_on_plateau: boolean, whether to restore checkpoint
            on validation metric's plateaus (only effective in early
            stopping)
        - net_name: str, name of trained model, without dataset and task
        - phrase_recall: boolean, whether to evaluate phrase recall
    """

    def __init__(self, dataset, task,
                 filter_duplicate_rels=False, filter_multiple_preds=False,
                 annotations_per_batch=128, batch_size=None,
                 backbone='resnet', num_workers=2,
                 apply_dynamic_lr=False, use_early_stopping=True,
                 restore_on_plateau=True, net_name='', phrase_recall=False):
        """Initialize configuration instance."""
        assert dataset in ORIG_IMAGES_PATH
        self.dataset = dataset
        self.task = task
        self.filter_duplicate_rels = filter_duplicate_rels
        self.filter_multiple_preds = filter_multiple_preds
        self._annotations_per_batch = annotations_per_batch
        self._batch_size = batch_size
        self.backbone = backbone
        self.num_workers = num_workers
        self.apply_dynamic_lr = apply_dynamic_lr
        self.use_early_stopping = use_early_stopping
        self.restore_on_plateau = restore_on_plateau
        self.net_name = '_'.join([
            net_name,
            task if task not in {'sgcls', 'sggen', 'preddet'} else 'predcls',
            dataset])
        self.phrase_recall = phrase_recall
        self._set_dataset_classes(dataset)
        self._set_dataset_task_annos_per_img()
        self._set_dataset_word2vec(dataset)
        self._set_logger()

    def _set_dataset_classes(self, dataset):
        """Load dataset classes."""
        obj_json = PATHS['json_path'] + dataset + '_objects.json'
        if os.path.exists(PATHS['json_path']) and os.path.exists(obj_json):
            with open(obj_json) as fid:
                self.obj_classes = json.load(fid)
                self._num_obj_classes = len(self.obj_classes)
        pred_json = PATHS['json_path'] + dataset + '_predicates.json'
        if os.path.exists(PATHS['json_path']) and os.path.exists(pred_json):
            with open(pred_json) as fid:
                self.rel_classes = json.load(fid)
                self._num_rel_classes = len(self.rel_classes)

    def _set_dataset_word2vec(self, dataset):
        """Load dataset word2vec array."""
        w2v_json = PATHS['json_path'] + dataset + '_word2vec.json'
        if os.path.exists(PATHS['json_path']) and os.path.exists(w2v_json):
            with open(w2v_json) as fid:
                w2vec = json.load(fid)  # word2vec dictionary
                obj2vec = torch.from_numpy(np.array(w2vec['objects'])).float()
                p2vec = torch.from_numpy(np.array(w2vec['predicates'])).float()
            self.obj2vec = obj2vec.cuda() if self.use_cuda else obj2vec
            self.pred2vec = p2vec.cuda() if self.use_cuda else p2vec

    def _set_dataset_task_annos_per_img(self):
        """
        Different number of image-wise annotations per dataset-task.

        All fields except for 'objects' refer to predicate annotations:
            - If duplicates_filtered, clear relations annotated > 1 time
            - If predicates_filtered, sample a single predicate per pair
            - If pairs, use all possible pairs of objects
        """
        self._annos_per_img = {
            'VG200': {
                'relations': 6.98,
                'duplicates_filtered': 4.69,
                'predicates_filtered': 4.45,
                'objects': 10.87,
                'pairs': 146.3,
                'max_objects': 45,
            },
            'VG80K': {
                'relations': 21.96,
                'duplicates_filtered': 18.89,
                'predicates_filtered': 18.1,
                'objects': 23.48,
                'pairs': 696.85,
                'max_objects': 25
            },
            'VGMSDN': {
                'relations': 11.02,
                'duplicates_filtered': 9.13,
                'predicates_filtered': 8.79,
                'objects': 12.48,
                'pairs': 190.05,
                'max_objects': 83
            },
            'VGVTE': {
                'relations': 10.94,
                'duplicates_filtered': 9.28,
                'predicates_filtered': 9.03,
                'objects': 13.04,
                'pairs': 243.76,
                'max_objects': 110
            },
            'VRD': {
                'relations': 8.02,
                'duplicates_filtered': 7.89,
                'predicates_filtered': 7.13,
                'objects': 7,
                'pairs': 52.98,
                'max_objects': 21
            },
            'VrR-VG': {
                'relations': 3.45,
                'duplicates_filtered': 3.03,
                'predicates_filtered': 2.97,
                'objects': 4.79,
                'pairs': 34.63,
                'max_objects': 64
            },
            'sVG': {
                'relations': 10.89,
                'duplicates_filtered': 8.36,
                'predicates_filtered': 8.11,
                'objects': 11.39,
                'pairs': 195.95,
                'max_objects': 119
            },
            'UnRel': {
                'relations': 8.02,
                'duplicates_filtered': 7.89,
                'predicates_filtered': 7.13,
                'objects': 7,
                'pairs': 52.98
            }
        }

    def _set_logger(self):
        """Configure logger."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        stream = logging.StreamHandler()
        stream.setFormatter(ColoredFormatter(
            '%(log_color)s%(asctime)s%(reset)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(stream)

    @property
    def use_cuda(self):
        """Return whether to use CUDA or not."""
        return USE_CUDA and torch.cuda.is_available()

    @property
    def num_classes(self):
        """Return number of classes depending on task."""
        if self.task in {'objcls', 'objdet'}:
            return self._num_obj_classes
        return self._num_rel_classes

    @property
    def num_obj_classes(self):
        """Return number of object classes."""
        return self._num_obj_classes

    @property
    def num_rel_classes(self):
        """Return number of predicate classes."""
        return self._num_rel_classes

    @property
    def paths(self):
        """Return a dict of paths useful to train/test/inference."""
        return PATHS

    @property
    def orig_annos_path(self):
        """Return path of stored original dataset annotations."""
        return ORIG_ANNOS_PATHS[self.dataset]

    @property
    def orig_img_path(self):
        """Return path of stored dataset images."""
        return ORIG_IMAGES_PATH[self.dataset]

    @property
    def batch_size(self):
        """Return batch size in terms of images."""
        if self._batch_size is not None:
            return self._batch_size  # custom batch size defined
        if self.task == 'objdet':
            return 4
        annos_per_img = self._annos_per_img[self.dataset]
        if self.task in {'predcls', 'sgcls'}:
            annos_per_img = annos_per_img['pairs']
        elif self.task == 'objcls':
            annos_per_img = annos_per_img['objects']
        elif self.task == 'preddet' and self.filter_multiple_preds:
            annos_per_img = annos_per_img['predicates_filtered']
        elif self.task == 'preddet' and self.filter_duplicate_rels:
            annos_per_img = annos_per_img['duplicates_filtered']
        elif self.task in {'preddet', 'sggen'}:
            annos_per_img = annos_per_img['relations']
        batch_size = ceil(self._annotations_per_batch / annos_per_img)
        return max(batch_size, 1)

    @property
    def annotations_per_batch(self):
        """Return batch size in terms of annotations."""
        if self._batch_size is None or self.task in {'objdet', 'sggen'}:
            return self._annotations_per_batch
        annos_per_img = self._annos_per_img[self.dataset]
        if self.task in {'predcls', 'sgcls'}:
            annos_per_img = annos_per_img['pairs']
        elif self.task == 'objcls':
            annos_per_img = annos_per_img['objects']
        elif self.task == 'preddet' and self.filter_multiple_preds:
            annos_per_img = annos_per_img['predicates_filtered']
        elif self.task == 'preddet' and self.filter_duplicate_rels:
            annos_per_img = annos_per_img['duplicates_filtered']
        elif self.task == 'preddet':
            annos_per_img = annos_per_img['relations']
        return annos_per_img * self._batch_size

    @property
    def weight_decay(self):
        """Return weight decay for an optimizer."""
        return 5e-5 if 'VG' in self.dataset else 5e-4

    @property
    def train_rpn(self):
        """Return whether to retrain region proposal network."""
        return self.dataset not in {'VRD', 'UnRel'}

    @property
    def train_top(self):
        """Return whether to retrain the layer before the classifier."""
        return self.dataset not in {'VRD', 'UnRel'}

    @property
    def max_obj_dets_per_img(self):
        """Return number of maximum object detections per image."""
        return min(64, self._annos_per_img[self.dataset]['max_objects'])
