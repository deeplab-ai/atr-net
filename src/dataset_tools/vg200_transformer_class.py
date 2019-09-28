# -*- coding: utf-8 -*-
"""Transform annotations into a standard desired format."""

import json
import os
import math

import numpy as np
import h5py
from src.dataset_tools.dataset_transformer_class import DatasetTransformer


class VG200Transformer(DatasetTransformer):
    """Extends DatasetTransformer for VG200 annotations."""

    def __init__(self, config):
        """Initialize trnasformer."""
        super().__init__(config)

    def transform(self):
        """Run the transformation pipeline."""
        jsons = [
            self._preddet_json,
            self._predcls_json,
            self._predicate_json,
            self._object_json,
            self._word2vec_json,
            self._preddet_probability_json,
            self._predcls_probability_json
        ]
        if not all(os.path.exists(anno) for anno in jsons):
            annos = self.create_relationship_json()
            predicates, objects = self.save_predicates_objects(annos)
            if not os.path.exists(self._word2vec_json):
                self.save_word2vec_vectors(predicates, objects)
            annos = self.update_labels(annos, predicates, objects)
            if not os.path.exists(self._predcls_json):
                self.create_pred_cls_json(annos, predicates)
            if not os.path.exists(self._preddet_probability_json):
                with open(self._preddet_json) as fid:
                    annos = json.load(fid)
                self.compute_relationship_probabilities(
                    annos, predicates, objects, with_bg=False)
            if not os.path.exists(self._predcls_probability_json):
                with open(self._predcls_json) as fid:
                    annos = json.load(fid)
                self.compute_relationship_probabilities(
                    annos, predicates, objects, with_bg=True)

    def create_relationship_json(self):
        """
        Transform VG200 annotations.

        Inputs:
            - [
                {
                    'filename': name,
                    'split_id': split_id,
                    'relationships': {rel_id: pair},
                    'boxes': {obj_id: decoded box}
                }
            ]
        """
        self._load_dataset()
        return self._set_annos()

    def _load_dataset(self):
        # Load images' metadata
        with open(self._orig_annos_path + 'image_data.json') as fid:
            self._img_names = [
                img['url'].split('/')[-1]
                for img in json.load(fid)
                if img['image_id'] not in [1592, 1722, 4616, 4617]
            ]

        # Load object and predicate names
        with open(self._orig_annos_path + 'VG-SGG-dicts.json') as fid:
            dict_annos = json.load(fid)
        self._predicate_names = {
            int(key): val
            for key, val in dict_annos['idx_to_predicate'].items()
        }
        self._object_names = {
            int(key): val
            for key, val in dict_annos['idx_to_label'].items()
        }

    def _set_annos(self):
        annos = h5py.File(self._orig_annos_path + 'VG-SGG.h5', 'r')
        split_ids = np.array(annos['split'])
        first_test_index = np.nonzero(split_ids)[0][0]
        split_ids[first_test_index - 2000: first_test_index] = 1
        boxes = np.array(annos['boxes_512'])
        obj_labels = [
            int(label) for label in np.array(annos['labels']).flatten()]
        predicate_labels = [
            int(pred) for pred in np.array(annos['predicates']).flatten()]
        relationships = np.array(annos['relationships'])
        scales_heights_widths = {
            img_name: self._compute_im_scale(img_name)
            for img_name in self._img_names
        }
        annos = [
            {
                'filename': name,
                'split_id': int(split_id),
                'height': scales_heights_widths[name][1],
                'width': scales_heights_widths[name][2],
                'im_scale': scales_heights_widths[name][0],
                'objects': {
                    'names': [
                        self._object_names[obj_labels[obj]]
                        for obj in range(first_box, last_box + 1)],
                    'boxes': [
                        self._decode_box(
                            boxes[obj],
                            scales_heights_widths[name][1],
                            scales_heights_widths[name][2],
                            512)
                        for obj in range(first_box, last_box + 1)]
                },
                'relations': {
                    'names': [
                        self._predicate_names[predicate_labels[rel]]
                        for rel in range(first_rel, last_rel + 1)
                        if first_rel > -1],
                    'subj_ids': [
                        int(relationships[rel][0] - first_box)
                        for rel in range(first_rel, last_rel + 1)
                        if first_rel > -1],
                    'obj_ids': [
                        int(relationships[rel][1] - first_box)
                        for rel in range(first_rel, last_rel + 1)
                        if first_rel > -1]
                }
            }
            for name, split_id, first_rel, last_rel, first_box, last_box
            in zip(
                self._img_names, split_ids, annos['img_to_first_rel'][:],
                annos['img_to_last_rel'][:], annos['img_to_first_box'][:],
                annos['img_to_last_box'][:]
            )
            if first_box > -1
        ]
        return annos

    @staticmethod
    def _decode_box(box, orig_height, orig_width, im_long_size):
        """
        Convert encoded box back to original.

        Inputs:
            - box: array, [x_center, y_center, width, height]
            - orig_height: int, height of the original image
            - orig_width: int, width of the original image
            - im_long_size: int, rescaled length of longer lateral
        Returns:
            - decoded box: list, [y_min, y_max, x_min, x_max]
        """
        # Center-oriented to left-top-oriented
        box = box.tolist()
        box[0] -= box[2] / 2
        box[1] -= box[3] / 2

        # Re-scaling to original size
        scale = max(orig_height, orig_width) / im_long_size
        box[0] = max(math.floor(scale * box[0]), 0)
        box[1] = max(math.floor(scale * box[1]), 0)
        box[2] = max(math.ceil(scale * box[2]), 2)
        box[3] = max(math.ceil(scale * box[3]), 2)

        # Boxes at least 2x2 that fit in the image
        box[0] = min(box[0], orig_width - 2)
        box[1] = min(box[1], orig_height - 2)
        box[2] = min(box[2], orig_width - box[0])
        box[3] = min(box[3], orig_height - box[1])

        # Convert to [y_min, y_max, x_min, x_max]
        return [box[1], box[1] + box[3] - 1, box[0], box[0] + box[2] - 1]
