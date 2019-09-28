# -*- coding: utf-8 -*-
"""A class to transform VRD matlab annotations into json format."""

import random

from scipy.io import loadmat

from src.dataset_tools.dataset_transformer_class import DatasetTransformer


class VRDTransformer(DatasetTransformer):
    """Transform matlab annotations to json."""

    def __init__(self, config):
        """Initialize VRDTransformer."""
        super().__init__(config)

    def create_relationship_json(self):
        """Transform VRD annotations."""
        annos = loadmat(self._orig_annos_path + 'annotation_train.mat')
        json_annos = self._merge_rel_annos(annos['annotation_train'][0], 0)
        for anno in json_annos[-350:]:
            anno['split_id'] = 1
        annos = loadmat(self._orig_annos_path + 'annotation_test.mat')
        json_annos += self._merge_rel_annos(annos['annotation_test'][0], 2)
        return json_annos

    def _merge_rel_annos(self, annos, split_id):
        scales_heights_widths = [
            self._compute_im_scale(anno[0]['filename'][0][0])
            for anno in annos
        ]
        return [
            {
                'filename': anno[0]['filename'][0][0],
                'split_id': split_id,
                'height': shw[1],
                'width': shw[2],
                'im_scale': shw[0],
                'relationships': [
                    {
                        'subject': rel[0]['phrase'][0][0][0][0],
                        'subject_box': rel[0]['subBox'][0][0].tolist(),
                        'predicate': rel[0]['phrase'][0][0][1][0],
                        'object': rel[0]['phrase'][0][0][2][0],
                        'object_box': rel[0]['objBox'][0][0].tolist()
                    }
                    for r, rel in enumerate(anno[0]['relationship'][0][0])
                ]
            }
            for anno, shw in zip(annos, scales_heights_widths)
            if self._handle_no_relationships(anno) and shw[0] is not None
        ]

    @staticmethod
    def _handle_no_relationships(anno):
        """Check if annotation 'anno' has a relationship part."""
        try:
            anno[0]['relationship']
            return True
        except:
            return False
