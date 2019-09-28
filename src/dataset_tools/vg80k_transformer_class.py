# -*- coding: utf-8 -*-
"""Transform annotations into a standard desired format."""

import os
import json

from src.dataset_tools.dataset_transformer_class import DatasetTransformer


class VG80KTransformer(DatasetTransformer):
    """Extands DatasetTransformer for VG80K."""

    def __init__(self, config):
        """Initialize VG80kTransformer."""
        super().__init__(config)

    def transform(self):
        """Run the transformation pipeline."""
        jsons = [
            self._preddet_json,
            self._predcls_json,
            self._predicate_json,
            self._object_json,
            self._word2vec_json
        ]
        if not all(os.path.exists(anno) for anno in jsons):
            annos = self.create_relationship_json()
            annos = self._transform_annotations(annos)
            predicates, objects = self.save_predicates_objects(annos)
            if not os.path.exists(self._word2vec_json):
                self.save_word2vec_vectors(predicates, objects)
            annos = self.update_labels(annos, predicates, objects)
            if not os.path.exists(self._predcls_json):
                self.create_pred_cls_json(annos, predicates)

    def create_relationship_json(self):
        """Transform VG80K annotations."""
        with open(self._orig_annos_path + 'train_clean.json') as fid:
            split_ids = {img: 0 for img in json.load(fid)}
        with open(self._orig_annos_path + 'val_clean.json') as fid:
            split_ids.update({img: 1 for img in json.load(fid)})
        with open(self._orig_annos_path + 'test_clean.json') as fid:
            split_ids.update({img: 2 for img in json.load(fid)})
        anno_json = 'relationships_clean_spo_joined_and_merged.json'
        with open(self._orig_annos_path + anno_json) as fid:
            annos = json.load(fid)
        self.noisy_labels = {
            'sasani', 'skyramp', 'linespeople', 'buruburu',
            'gunport', 'dirttrack', 'greencap', 'shrublike'
        }
        json_annos = self._merge_rel_annos(annos, split_ids)
        return json_annos

    def _merge_rel_annos(self, annos, split_ids):
        scales_heights_widths = {
            anno['image_id']:
                self._compute_im_scale(str(anno['image_id']) + '.jpg')
            for anno in annos
        }
        return [
            {
                'filename': str(anno['image_id']) + '.jpg',
                'split_id': int(split_ids[anno['image_id']]),
                'height': scales_heights_widths[anno['image_id']][1],
                'width': scales_heights_widths[anno['image_id']][2],
                'im_scale': scales_heights_widths[anno['image_id']][0],
                'relationships': [
                    {
                        'subject': rel['subject']['name'],
                        'subject_box': self._decode_box([
                            rel['subject'][item]
                            for item in ['x', 'y', 'w', 'h']]),
                        'predicate': str(rel['predicate']),
                        'object': rel['object']['name'],
                        'object_box': self._decode_box([
                            rel['object'][item]
                            for item in ['x', 'y', 'w', 'h']])
                    }
                    for rel in anno['relationships']
                    if self._decode_box([
                        rel['subject'][item] for item in ['x', 'y', 'w', 'h']])
                    and self._decode_box([
                        rel['object'][item] for item in ['x', 'y', 'w', 'h']])
                    and all(
                        word not in self.noisy_labels
                        for word in rel['subject']['name'].split())
                    and all(
                        word not in self.noisy_labels
                        for word in rel['object']['name'].split())
                    and all(
                        word not in self.noisy_labels
                        for word in rel['predicate'].split())
                ]
            }
            for anno in annos
            if scales_heights_widths[anno['image_id']][0] is not None
        ]

    @staticmethod
    def _decode_box(box):
        box = [
            int(box[1]), int(box[1]) + int(box[3]),
            int(box[0]), int(box[0]) + int(box[2])
        ]
        if box[0] >= box[1] or box[2] >= box[3]:
            return []
        return box
