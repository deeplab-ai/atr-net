# -*- coding: utf-8 -*-
"""Functions to load from and save to files."""

import os
import json
import random
from copy import deepcopy

import numpy as np

from config import PATHS


def load_annotations(mode, dataset, filter_duplicates=False,
                     filter_multiple=False):
    """
    Load annotations depending on mode.

    Inputs:
        - mode: str, e.g. 'preddet', 'predcls', 'sgcls', 'objdet' etc.
        - dataset: str, e.g. 'sVG', 'VG200', 'VRD' etc.
        - filter_duplicates: boolean, whether to filter relations
            annotated more than once
        - filter_multiple: boolean, whether to sample a single
            predicate per object pair
    Outputs:
        - annotations: list of dicts corresponding to images
        - zeroshot_annotations: list of dicts for unseen relationships
    """
    _mode = mode if mode in {'preddet', 'sggen'} else 'predcls'
    with open(PATHS['json_path'] + dataset + '_' + _mode + '.json') as fid:
        annotations = json.load(fid)
    orig_img_names = set(os.listdir(ORIG_ANNOS_PATHS[dataset]))
    annotations = [
        {
            'filename': anno['filename'],
            'split_id': anno['split_id'],
            'height': anno['height'],
            'width': anno['width'],
            'im_scale': anno['im_scale'],
            'objects': {
                'boxes': np.array(anno['objects']['boxes']),
                'ids': np.array(anno['objects']['ids']).astype(int),
                'names': np.array(anno['objects']['names']),
                'scores': (
                    np.array(anno['objects']['scores'])
                    if 'scores' in anno['objects'] else None)
            },
            'relations': {
                'ids': np.array(anno['relations']['ids']).astype(int),
                'names': np.array(anno['relations']['names']),
                'subj_ids': np.array(anno['relations']['subj_ids']),
                'obj_ids': np.array(anno['relations']['obj_ids'])
            }
        }
        for anno in annotations
        if anno['filename'] in orig_img_names
        and (any(anno['relations']['names']) or mode in {'objcls', 'objdet'})
        and any(anno['objects']['names'])
    ]
    if filter_duplicates:
        for anno in annotations:
            if anno['split_id'] == 0:
                anno['relations'] = _filter_duplicates(anno['relations'])
    if filter_multiple:
        for anno in annotations:
            if anno['split_id'] == 0:
                anno['relations'] = _filter_multiple(anno['relations'])
    seen = set(
        (anno['objects']['ids'][s_id], rel_id, anno['objects']['ids'][o_id])
        for anno in annotations if anno['split_id'] in (0, 1)
        for s_id, rel_id, o_id in zip(
            anno['relations']['subj_ids'],
            anno['relations']['ids'],
            anno['relations']['obj_ids']
        )
    )
    zeroshot_annotations = []
    for anno in annotations:
        if anno['split_id'] == 2 and any(anno['relations']['names'].tolist()):
            keep = [
                r for r, (sid, rid, oid) in enumerate(zip(
                    anno['objects']['ids'][anno['relations']['subj_ids']],
                    anno['relations']['ids'],
                    anno['objects']['ids'][anno['relations']['obj_ids']]
                ))
                if (sid, rid, oid) not in seen
            ]
            if keep:
                zeroshot_annotations.append({
                    'filename': anno['filename'],
                    'split_id': anno['split_id'],
                    'objects': anno['objects'],
                    'relations': {
                        'ids': anno['relations']['ids'][keep],
                        'names': anno['relations']['names'][keep],
                        'subj_ids': anno['relations']['subj_ids'][keep],
                        'obj_ids': anno['relations']['obj_ids'][keep],
                    }
                })
    return annotations, zeroshot_annotations


def _filter_duplicates(relations):
    """Filter relations (triplets & boxes) appearing more than once."""
    _, unique_inds = np.unique(np.stack(
        (relations['subj_ids'], relations['ids'], relations['obj_ids']), axis=1
    ), axis=0, return_index=True)
    relations['ids'] = relations['ids'][unique_inds]
    relations['subj_ids'] = relations['subj_ids'][unique_inds]
    relations['obj_ids'] = relations['obj_ids'][unique_inds]
    relations['names'] = relations['names'][unique_inds]
    return relations


def _filter_multiple(relations):
    """Filter multiple annotations for the same object pair."""
    _, unique_inds = np.unique(np.stack(
        (relations['subj_ids'], relations['obj_ids']), axis=1
    ), axis=0, return_index=True)
    relations['ids'] = relations['ids'][unique_inds]
    relations['subj_ids'] = relations['subj_ids'][unique_inds]
    relations['obj_ids'] = relations['obj_ids'][unique_inds]
    relations['names'] = relations['names'][unique_inds]
    return relations
