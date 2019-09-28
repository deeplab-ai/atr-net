# -*- coding: utf-8 -*-
"""Custom datasets and data loaders for Scene Graph Generation."""

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from src.tools.feature_extractors import get_image_blob, BaseFeatureExtractor
from src.tools.spatial_feature_extractors import (
    create_binary_masks, get_box_deltas, _compute_predicate_boxes)

FEATURE_EXTRACTOR = BaseFeatureExtractor()


class SGGDataset(Dataset):
    """Dataset utilities for Scene Graph Generation."""

    def __init__(self, annotations, config, features={}):
        """
        Initialize dataset.

        Inputs:
            - annotations: list of annotations per image
            - config: config class, see config.py
            - features: set of str, features to use in train/test
        """
        self._annotations = annotations
        self._config = config
        self._features = features
        self._set_init()
        self._set_methods()

    def __getitem__(self, index):
        """Get image's data (used by loader to later form a batch)."""
        anno = self._annotations[self._files[index]]
        return_json = {
            feature: self._methods[feature](anno)
            for feature in self._features if feature in self._methods
        }
        return_json['filenames'] = self._files[index]
        return return_json

    def __len__(self):
        """Override __len__ method, return dataset's size."""
        return len(self._files)

    def _set_init(self):
        """Set dataset variables."""
        self._dataset = self._config.dataset
        self._json_path = self._config.paths['json_path']
        self._orig_image_path = self._config.orig_img_path
        self._use_cuda = self._config.use_cuda
        self._annotations = {
            anno['filename']: anno for anno in self._annotations
        }
        self._files = list(self._annotations.keys())
        self._is_set = {
            'word2vec': False,
            'probabilities': False,
            'vocabulary': False
        }
        self._config.logger.debug(
            'Set up dataset of %d files' % len(self._files))

    def _set_methods(self):
        """Correspond a method to each feature type."""
        self._methods = {
            'bg_targets': self.get_bg_targets,
            'box_deltas': self.get_box_deltas,
            'boxes': self.get_boxes,
            'images': self.get_image,
            'image_info': self.get_image_info,
            'image_roi': self.get_image_roi,
            'labels': self.get_labels,
            'object_1hot_vectors': self.get_object_1hot_vectors,
            'object_embeddings': self.get_object_embeddings,
            'object_masks': self.get_object_masks,
            'object_rcnn_rois': self.get_object_rcnn_rois,
            'object_rois': self.get_object_rois,
            'object_targets': self.get_object_targets,
            'predicate_1hot_vectors': self.get_predicate_1hot_vectors,
            'predicate_embeddings': self.get_predicate_embeddings,
            'predicate_masks': self.get_predicate_masks,
            'predicate_probabilities': self.get_predicate_probabilities,
            'predicate_rcnn_rois': self.get_predicate_rcnn_rois,
            'predicate_rois': self.get_predicate_rois,
            'predicate_targets': self.get_predicate_targets,
            'relations': self.get_relations
        }

    def _set_probabilities(self, mode='predcls'):
        """Set predicate probability matrix for given dataset."""
        json_name = ''.join([
            self._json_path, self._dataset, '_', mode, '_probabilities.json'])
        with open(json_name) as fid:
            self.probabilities = np.array(json.load(fid))
        self._is_set['probabilities'] = True

    def _set_vocabulary(self):
        """Set vocabulary (objects, predicates) for given dataset."""
        with open(self._json_path + self._dataset + '_predicates.json') as fid:
            self.predicate_list = json.load(fid)
            self.predicate_1hot = np.eye(len(self.predicate_list))
        with open(self._json_path + self._dataset + '_objects.json') as fid:
            self.object_list = json.load(fid)
            self.object_1hot = np.eye(len(self.object_list))
        self._is_set['vocabulary'] = True

    def _set_word2vec(self):
        """Set word2vec matrices for the vocabulary of given dataset."""
        with open(self._json_path + self._dataset + '_word2vec.json') as fid:
            word2vec = json.load(fid)
            self.obj2vec = np.array(word2vec['objects'])
            self.pred2vec = np.array(word2vec['predicates'])
        self._is_set['word2vec'] = True

    @staticmethod
    def get_bg_targets(anno):
        """Return foreground/background targets for given image."""
        return np.array([
            0 if pred == '__background__' else 1
            for pred in anno['relations']['names']])

    @staticmethod
    def get_box_deltas(anno, box_type='lu_2016'):
        """Return box deltas for given image."""
        boxes = anno['objects']['boxes']
        rel_boxes = _compute_predicate_boxes(
            anno['objects']['boxes'], anno['relations']['subj_ids'],
            anno['relations']['obj_ids'], box_type)
        return get_box_deltas(
            boxes[anno['relations']['subj_ids']],
            boxes[anno['relations']['obj_ids']],
            rel_boxes, anno['height'], anno['width'])

    @staticmethod
    def get_boxes(anno):
        """Return (N, 2, 4) bounding boxes for given image."""
        boxes = anno['objects']['boxes']
        return np.concatenate((
            boxes[anno['relations']['subj_ids']][:, None, :],
            boxes[anno['relations']['obj_ids']][:, None, :]), axis=1)

    def get_image(self, anno):
        """Return an image blob (1, H, W, 3)."""
        return get_image_blob(
            self._orig_image_path + anno['filename'], anno['im_scale'])

    @staticmethod
    def get_image_info(anno):
        """Return height, width, scale of given image."""
        return np.array([[
            anno['height'], anno['width'], 1.0]]) * anno['im_scale']

    @staticmethod
    def get_image_roi(anno):
        """Return a whole given image as a RoI."""
        return np.array([[
            0.0, 0.0, 0.0, anno['width'], anno['height']]]) * anno['im_scale']

    @staticmethod
    def get_labels(anno):
        """Return label vector (subj, -1, obj) for given image."""
        object_ids = anno['objects']['ids']
        labels = -np.ones((len(anno['relations']['subj_ids']), 3))
        labels[:, 0] = object_ids[anno['relations']['subj_ids']]
        labels[:, 2] = object_ids[anno['relations']['obj_ids']]
        return labels

    def get_object_1hot_vectors(self, anno):
        """Return 1-hot vectors for the objects of given image."""
        if not self._is_set['vocabulary']:
            self._set_vocabulary()
        vecs = self.object_1hot[anno['objects']['ids']]
        if anno['objects']['scores'] is not None:
            vecs = vecs * anno['objects']['scores'][:, None]
        return vecs

    def get_object_embeddings(self, anno):
        """Return embeddings for objects of given image."""
        if not self._is_set['word2vec']:
            self._set_word2vec()
        return self.obj2vec[anno['objects']['ids']]

    @staticmethod
    def get_object_masks(anno):
        """Return mask features for objects of given image."""
        return create_binary_masks(
            anno['objects']['boxes'], anno['height'], anno['width'])

    @staticmethod
    def get_object_rcnn_rois(anno):
        """Return rois for objects of given image (for ObjDet)."""
        boxes = anno['objects']['boxes']
        rois = np.zeros((1, len(boxes), 5))
        rois[0, :, :4] = np.round(boxes[:, (2, 0, 3, 1)] * anno['im_scale'])
        rois[0, :, 4] = anno['objects']['ids'] + 1
        # rois = rois[:64]
        if rois.shape[1] < 64:
            padded_rois = np.zeros((1, 64, 5))
            padded_rois[0, :rois.shape[1], :rois.shape[2]] = rois
            return padded_rois
        return rois

    @staticmethod
    def get_object_rois(anno):
        """Return rois for objects of given image (for SGCls)."""
        boxes = anno['objects']['boxes']
        rois = np.zeros((1, len(boxes), 5))
        rois[0, :, 1:] = np.round(boxes[:, (2, 0, 3, 1)] * anno['im_scale'])
        rois[0, :, 0] = 0.0
        return rois

    @staticmethod
    def get_object_targets(anno):
        """Return object targets for given image."""
        return anno['objects']['ids']

    def get_predicate_1hot_vectors(self, anno):
        """Return 1-hot vectors for the predicates of given image."""
        if not self._is_set['vocabulary']:
            self._set_vocabulary()
        return self.predicate_1hot[anno['relations']['ids']]

    def get_predicate_embeddings(self, anno):
        """Return embeddings for predicates of given image."""
        if not self._is_set['word2vec']:
            self._set_word2vec()
        return self.pred2vec[anno['relations']['ids']]

    @staticmethod
    def get_predicate_masks(anno, box_type='lu_2016'):
        """Return mask features for predicates of given image."""
        rel_boxes = _compute_predicate_boxes(
            anno['objects']['boxes'], anno['relations']['subj_ids'],
            anno['relations']['obj_ids'], box_type)
        return create_binary_masks(rel_boxes, anno['height'], anno['width'])

    def get_predicate_probabilities(self, anno, mode='predcls'):
        """Return predicate probability vectors for given image."""
        if not self._is_set['probabilities']:
            self._set_probabilities(mode)
        object_ids = anno['objects']['ids']
        return self.probabilities[
            object_ids[anno['relations']['subj_ids']],
            object_ids[anno['relations']['obj_ids']]
        ]

    @staticmethod
    def get_predicate_rcnn_rois(anno, box_type='lu_2016'):
        """Return rois for predicates of given image (for ObjDet)."""
        boxes = _compute_predicate_boxes(
            anno['objects']['boxes'], anno['relations']['subj_ids'],
            anno['relations']['obj_ids'], box_type)
        rois = np.zeros((1, len(boxes), 5))
        rois[0, :, :4] = np.round(boxes[:, (2, 0, 3, 1)] * anno['im_scale'])
        rois[0, :, 4] = anno['relations']['ids'] + 1
        # rois = rois[:64]
        if rois.shape[1] < 64:
            padded_rois = np.zeros((1, 64, 5))
            padded_rois[0, :rois.shape[1], :rois.shape[2]] = rois
            return padded_rois
        return rois

    @staticmethod
    def get_predicate_rois(anno, box_type='lu_2016'):
        """Return rois for predicates of given image."""
        rel_boxes = _compute_predicate_boxes(
            anno['objects']['boxes'], anno['relations']['subj_ids'],
            anno['relations']['obj_ids'], box_type)
        rois = np.zeros((len(rel_boxes), 5))
        rois[:, 1:] = rel_boxes[:, (2, 0, 3, 1)]
        return rois * anno['im_scale']

    @staticmethod
    def get_predicate_targets(anno):
        """Return predicate targets for given image."""
        return anno['relations']['ids']

    @staticmethod
    def get_relations(anno):
        """Return an array of related object ids for given image."""
        return np.stack(
            (anno['relations']['subj_ids'], anno['relations']['obj_ids']),
            axis=1).tolist()


def sgg_collate_fn(batch_data, features):
    """Collate function for custom data loading."""
    return_batch = {'filenames': [item['filenames'] for item in batch_data]}
    tensor_features = {
        'box_deltas', 'images', 'image_info', 'image_roi',
        'object_1hot_vectors', 'object_embeddings', 'object_masks',
        'object_rcnn_rois', 'object_rois',
        'predicate_1hot_vectors', 'predicate_embeddings', 'predicate_masks',
        'predicate_probabilities', 'predicate_rcnn_rois', 'predicate_rois'
    }
    for feature in features:
        if feature in tensor_features:
            return_batch[feature] = [
                torch.from_numpy(item[feature]).float() for item in batch_data]
        elif 'targets' in feature:  # targets are long integers
            return_batch[feature] = [
                torch.from_numpy(item[feature]).long() for item in batch_data]
        else:  # list of numpy arrays
            return_batch[feature] = [item[feature] for item in batch_data]
    return return_batch


class SGGDataLoader(torch.utils.data.DataLoader):
    """Custom data loader for Scene Graph Generation."""

    def __init__(self, dataset, batch_size, shuffle=True, num_workers=2,
                 drop_last=False, collate_fn=sgg_collate_fn, use_cuda=False):
        """Initialize loader for given dataset and annotations."""
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, drop_last=drop_last,
                         collate_fn=collate_fn)
        self._use_cuda = use_cuda

    def get(self, feature, batch, step):
        """Get specific feature from a given batch."""
        not_tensors = {'boxes', 'filenames', 'labels', 'relations'}
        if feature == 'base_features':
            return self.get_base_features(batch, step)
        if feature in not_tensors or not self._use_cuda:
            return batch[feature][step]
        return batch[feature][step].cuda()

    @torch.no_grad()
    def _set_base_features(self, batch):
        """Get images of current batch."""
        im_blob = torch.stack(batch, dim=0)
        self._base_features = FEATURE_EXTRACTOR(
            im_blob.to(torch.device("cuda:0")) if self._use_cuda else im_blob)

    def get_base_features(self, batch, step):
        """Get ROI pooling features."""
        if step == 0:
            self._set_base_features(batch['images'])
        return self._base_features[step].unsqueeze(0)
