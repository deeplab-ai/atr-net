# -*- coding: utf-8 -*-
"""Transform annotations into a standard desired format."""

import h5py

from src.dataset_tools.dataset_transformer_class import DatasetTransformer


class VGVTETransformer(DatasetTransformer):
    """Extands DatasetTransformer for VGVTE."""

    def __init__(self, config):
        """Initialize VGVTETransformer."""
        super().__init__(config)

    def create_relationship_json(self):
        """Transform VGVTE annotations."""
        annos = h5py.File(self._orig_annos_path + 'vg1_2_meta.h5', 'r')
        self._predicates = {
            int(idx): str(name[()])
            for idx, name in dict(annos['meta']['pre']['idx2name']).items()
        }
        self._objects = {
            int(idx): str(name[()])
            for idx, name in dict(annos['meta']['cls']['idx2name']).items()
        }
        json_annos = self._merge_rel_annos(dict(annos['gt']['train']), 0)
        for anno in json_annos[-2000:]:
            anno['split_id'] = 1
        json_annos += self._merge_rel_annos(dict(annos['gt']['test']), 2)
        noisy_images = {  # contain bboxes > image
            '1829.jpg', '2391277.jpg', '150333.jpg',
            '3201.jpg', '713208.jpg', '1592325.jpg'
        }
        json_annos = [
            anno for anno in json_annos if anno['filename'] not in noisy_images
        ]
        return json_annos

    def _merge_rel_annos(self, annos, split_id):
        scales_heights_widths = {
            filename: self._compute_im_scale(filename + '.jpg')
            for filename in annos.keys()
        }
        return [
            {
                'filename': filename + '.jpg',
                'split_id': int(split_id),
                'height': scales_heights_widths[filename][1],
                'width': scales_heights_widths[filename][2],
                'im_scale': scales_heights_widths[filename][0],
                'relationships': [
                    {
                        'subject': self._objects[rel[0]].lower(),
                        'subject_box': self._decode_box(sub_box),
                        'predicate': self._predicates[rel[1]].lower(),
                        'object': self._objects[rel[2]].lower(),
                        'object_box': self._decode_box(obj_box)
                    }
                    for sub_box, rel, obj_box in zip(
                        relationships['sub_boxes'],
                        relationships['rlp_labels'],
                        relationships['obj_boxes'])
                ]
            }
            for filename, relationships in annos.items()
            if scales_heights_widths[filename][0] is not None
        ]

    @staticmethod
    def _decode_box(box):
        return [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
