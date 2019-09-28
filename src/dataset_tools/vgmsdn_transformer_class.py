# -*- coding: utf-8 -*-
"""Transform annotations into a standard desired format."""

import json

from src.dataset_tools.dataset_transformer_class import DatasetTransformer


class VGMSDNTransformer(DatasetTransformer):
    """Extands DatasetTransformer for VGMSDN."""

    def __init__(self, config):
        """Initialize VGMSDNTransformer."""
        super().__init__(config)

    def create_relationship_json(self):
        """Transform VGMSDN annotations."""
        with open(self._orig_annos_path + 'train.json') as fid:
            annos = json.load(fid)
        json_annos = self._merge_rel_annos(annos, 0)
        for anno in json_annos[-2000:]:
            anno['split_id'] = 1
        with open(self._orig_annos_path + 'test.json') as fid:
            annos = json.load(fid)
        json_annos += self._merge_rel_annos(annos, 2)
        noisy_labels = {
            'hang_on': 'hang on',
            'lay_on': 'lay on',
            'hang_from': 'hang from',
            'of_a': 'of a',
            'look_at': 'loon at',
            'in_a': 'in a',
            'walk_on': 'walk on',
            'on_side_of': 'on side of',
            'on_top_of': 'on top of',
            'on_front_of': 'on front of',
            'in_front_of': 'in front of',
            'stand_in': 'stand in',
            'sit_on': 'sit on',
            'inside_of': 'inside of',
            'stand_on': 'stand on',
            'be_in': 'be in',
            'on_a': 'on a',
            'attach_to': 'attach to',
            'next_to': 'next to',
            'wear_a': 'wear a',
            'have_a': 'have a',
            'sit_in': 'sit in',
            'be_on': 'be on'
        }
        for anno in json_annos:
            for rel in anno['relationships']:
                if rel['predicate'] in noisy_labels:
                    rel['predicate'] = noisy_labels[rel['predicate']]
        return json_annos

    def _merge_rel_annos(self, annos, split_id):
        scales_heights_widths = {
            anno['path']: self._compute_im_scale(anno['path'])
            for anno in annos
        }
        return [
            {
                'filename': anno['path'],
                'split_id': int(split_id),
                'height': scales_heights_widths[anno['path']][1],
                'width': scales_heights_widths[anno['path']][2],
                'im_scale': scales_heights_widths[anno['path']][0],
                'relationships': [
                    {
                        'subject': anno['objects'][rel['sub_id']]['class'],
                        'subject_box': self._decode_box(
                            anno['objects'][rel['sub_id']]['box']),
                        'predicate': str(rel['predicate']),
                        'object': anno['objects'][rel['obj_id']]['class'],
                        'object_box': self._decode_box(
                            anno['objects'][rel['obj_id']]['box']),
                    }
                    for rel in anno['relationships']
                    if self._decode_box(anno['objects'][rel['sub_id']]['box'])
                    and self._decode_box(anno['objects'][rel['obj_id']]['box'])
                ]
            }
            for anno in annos
            if scales_heights_widths[anno['path']][0] is not None
        ]

    @staticmethod
    def _decode_box(box):
        box = [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
        if box[0] >= box[1] or box[2] >= box[3]:
            return []
        return box
