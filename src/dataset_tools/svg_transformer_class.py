# -*- coding: utf-8 -*-
"""Transform annotations into a standard desired format."""

import json

from src.dataset_tools.dataset_transformer_class import DatasetTransformer


class SVGTransformer(DatasetTransformer):
    """Extands DatasetTransformer for sVG."""

    def __init__(self, config):
        """Initialize SVGTransformer."""
        super().__init__(config)

    def create_relationship_json(self):
        """Transform sVG annotations."""
        with open(self._orig_annos_path + 'svg_train.json') as fid:
            annos = json.load(fid)
        json_annos = self._merge_rel_annos(annos, 0)
        for anno in json_annos[-2000:]:
            anno['split_id'] = 1
        with open(self._orig_annos_path + 'svg_test.json') as fid:
            annos = json.load(fid)
        json_annos += self._merge_rel_annos(annos, 2)
        noisy_images = {  # contain bboxes > image
            '1191.jpg', '1360.jpg', '1159.jpg',
            '1018.jpg', '1327.jpg', '1280.jpg'
        }
        json_annos = [
            anno for anno in json_annos if anno['filename'] not in noisy_images
        ]
        return self._clear_obj_annos(json_annos)

    @staticmethod
    def _clear_obj_annos(annos):
        noisy_words = {
            "streetsign": "street sign",
            "theoutdoors": "outdoors",
            "licenseplate": "license plate",
            "stopsign": "stop sign",
            "toiletpaper": "toilet paper",
            "tennisracket": "tennis racket",
            "treetrunk": "tree trunk",
            "trafficlight": "traffic light",
            "bluesky": "blue sky",
            "firehydrant": "fire hydrant",
            "t-shirt": "t_shirt",
            "whiteclouds": "white clouds",
            "traincar": "train car",
            "tennisplayer": "tennis player",
            "skipole": "ski pole",
            "tenniscourt": "tennis court",
            "tennisball": "tennis ball",
            "baseballplayer": "baseball player"
        }
        for anno in annos:
            for rel in anno['relationships']:
                if rel['subject'] in noisy_words:
                    rel['subject'] = noisy_words[rel['subject']]
                if rel['object'] in noisy_words:
                    rel['object'] = noisy_words[rel['object']]
        return annos

    def _merge_rel_annos(self, annos, split_id):
        scales_heights_widths = [
            self._compute_im_scale(anno['url'].split('/')[-1])
            for anno in annos
        ]
        return [
            {
                'filename': anno['url'].split('/')[-1],
                'split_id': split_id,
                'height': shw[1],
                'width': shw[2],
                'im_scale': shw[0],
                'relationships': [
                    {
                        'subject': rel['phrase'][0],
                        'subject_box': [
                            rel['subject'][1], rel['subject'][3],
                            rel['subject'][0], rel['subject'][2]],
                        'predicate': rel['phrase'][1],
                        'object': rel['phrase'][2],
                        'object_box': [
                            rel['object'][1], rel['object'][3],
                            rel['object'][0], rel['object'][2]]
                    }
                    for rel in anno['relationships']
                ]
            }
            for anno, shw in zip(annos, scales_heights_widths)
            if shw[0] is not None
        ]
