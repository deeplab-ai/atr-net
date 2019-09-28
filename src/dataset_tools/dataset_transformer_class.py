# -*- coding: utf-8 -*-
"""Transform datasets annotations into a standard desired format."""

import json
import os

import numpy as np
from PIL import Image
import torch

from model.utils.config import cfg, cfg_from_file


class DatasetTransformer():
    """
    A class to transform annotations of a given dataset.

    Datasets supported:
        - VG200
        - VGMSDN
        - VGVTE
        - VRD
        - VrR-VG
        - sVG
        - UnRel
    """

    def __init__(self, config):
        """Initialize transformer providing the dataset name."""
        self._dataset = config.dataset
        self._faster_rcnn_path = config.paths['faster_rcnn_path']
        self._orig_annos_path = config.orig_annos_path
        self._orig_images_path = config.orig_img_path
        base = config.paths['json_path'] + self._dataset
        self._preddet_json = base + '_preddet.json'
        self._predcls_json = base + '_predcls.json'
        self._predicate_json = base + '_predicates.json'
        self._object_json = base + '_objects.json'
        self._word2vec_json = base + '_word2vec.json'
        self._preddet_probability_json = base + '_preddet_probabilities.json'
        self._predcls_probability_json = base + '_predcls_probabilities.json'

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
            annos = self._transform_annotations(annos)
            predicates, objects = self.save_predicates_objects(annos)
            if not os.path.exists(self._word2vec_json):
                self.save_word2vec_vectors(predicates, objects)
            annos = self.update_labels(annos, predicates, objects)
            if not os.path.exists(self._predcls_json):
                self.create_pred_cls_json(annos, predicates)
            if self._dataset != 'VG80K':
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

    @staticmethod
    def create_relationship_json():
        """
        Transform relationship annotations.

        Returns a list of dicts:
        {
            'filename': filename (no path),
            'split_id': int, 0/1/2 for train/val/test,
            'height': int, image height in pixels,
            'width': int, image width in pixels,
            'im_scale': float, resize scale for Faster-RCNN,
            'relationships': [
                {
                    'subject': str, subject_name,
                    'subject_box': [y_min, y_max, x_min, x_max],
                    'predicate': str, predicate_name
                    'object': object_name,
                    'object_box': [y_min, y_max, x_min, x_max]
                }
            ]
        }
        """
        return []

    def create_pred_cls_json(self, annos, predicates):
        """
        Annotate all possible pairs, adding background classes.

        Saves a list of dicts:
        {
            'filename': filename (no path),
            'split_id': int, 0/1/2 for train/val/test,
            'height': int, image height in pixels,
            'width': int, image width in pixels,
            'im_scale': float, resize scale for Faster-RCNN,
            'objects': {
                'ids': list of int,
                'names': list of str,
                'boxes': list of 4-tuples
            },
            'relations': {
                'ids': list of int,
                'names': list of str,
                'subj_ids': list of int,
                'obj_ids': list of int
            }
        }
        """
        for anno in annos:
            pairs = {
                (s, o): []
                for s in range(len(anno['objects']['ids']))
                for o in range(len(anno['objects']['ids']))
                if s != o
            }
            for rel in range(len(anno['relations']['subj_ids'])):
                subj_id = anno['relations']['subj_ids'][rel]
                obj_id = anno['relations']['obj_ids'][rel]
                if (subj_id, obj_id) not in pairs:
                    pairs[(subj_id, obj_id)] = []
                pairs[(subj_id, obj_id)].append((
                    anno['relations']['names'][rel],
                    anno['relations']['ids'][rel]
                ))
            pairs = {
                rel_tuple:
                    rels if any(rels)
                    else [('__background__', len(predicates) - 1)]
                for rel_tuple, rels in pairs.items()}
            pairs = np.array([
                (subj_id, obj_id, name, pred_id)
                for (subj_id, obj_id), preds in pairs.items()
                for (name, pred_id) in preds
            ])
            if pairs.size:
                anno['relations'] = {
                    'subj_ids': pairs[:, 0].astype(int).tolist(),
                    'obj_ids': pairs[:, 1].astype(int).tolist(),
                    'names': pairs[:, 2].astype(str).tolist(),
                    'ids': pairs[:, 3].astype(int).tolist()
                }
        with open(self._predcls_json, 'w') as fid:
            json.dump(annos, fid)

    def save_predicates_objects(self, annos):
        """Save predicates and objects lists and embeddings."""
        predicates = sorted(list(set(
            name for anno in annos for name in anno['relations']['names']
            if name != '__background__'
        )))
        predicates.append('__background__')
        with open(self._predicate_json, 'w') as fid:
            json.dump(predicates, fid)
        objects = sorted(list(set(
            name for anno in annos for name in anno['objects']['names']
        )))
        with open(self._object_json, 'w') as fid:
            json.dump(objects, fid)
        return predicates, objects

    def save_word2vec_vectors(self, predicates, objects):
        """Build word2vec dictionary of dataset vocabulary."""
        voc = {word for name in predicates + objects for word in name.split()}
        with open('glove.42B.300d.txt') as fid:
            glove_w2v = {
                line.split()[0]: np.array(line.split()[1:]).astype(float)
                for line in fid.readlines() if line.split()[0] in voc
            }
        print(list(voc - glove_w2v.keys()))
        assert list(voc - glove_w2v.keys()) == ['__background__']
        pred_w2v = np.array([
            np.mean([glove_w2v[word] for word in name.split()], axis=0)
            if any(word in glove_w2v for word in name.split())
            else np.zeros(300)
            for name in predicates
        ])
        # Set background as the mean of other classes
        pred_w2v[-1] = np.mean(pred_w2v[:-1, :], axis=0)
        obj_w2v = np.array([
            np.mean([glove_w2v[word] for word in name.split()], axis=0)
            for name in objects
        ])
        with open(self._word2vec_json, 'w') as fid:
            json.dump({
                'predicates': pred_w2v.tolist(),
                'objects': obj_w2v.tolist()
            }, fid)

    def update_labels(self, annos, predicates, objects):
        """Update objects and predicates ids."""
        predicates = {pred: p for p, pred in enumerate(predicates)}
        objects = {obj: o for o, obj in enumerate(objects)}
        for anno in annos:
            anno['relations']['ids'] = [
                predicates[name] for name in anno['relations']['names']]
            anno['objects']['ids'] = [
                objects[name] for name in anno['objects']['names']]
        if not os.path.exists(self._preddet_json):
            with open(self._preddet_json, 'w') as fid:
                json.dump(annos, fid)
        return annos

    def _compute_im_scale(self, im_name):
        """Compute scaling for an image to match for Faster-RCNN."""
        cfg_from_file(self._faster_rcnn_path + 'res101_ls.yml')
        if not os.path.exists(self._orig_images_path + im_name):
            return None, None, None
        im_width, im_height = Image.open(self._orig_images_path + im_name).size
        im_size_min = min(im_height, im_width)
        im_size_max = max(im_height, im_width)
        im_scale = float(cfg.TEST.SCALES[0]) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        return im_scale, im_height, im_width

    def compute_relationship_probabilities(self, annos, predicates, objects,
                                           with_bg=True):
        """
        Compute probabilities P(Pred|<Sub,Obj>) from dataset.

        The Laplacian estimation is used:
            P(A|B) = (N(A,B)+1) / (N(B)+V_A),
        where:
            N(X) is the number of occurences of X and
            V_A is the number of different values that A can have

        Returns a (n_obj, n_obj, n_rel) array where P[i,j,k] = P(k|i,j)
        """
        prob_matrix = np.ones((len(objects), len(objects), len(predicates)))
        for anno in annos:
            if any(anno['relations']['names']):
                ids = np.array(anno['objects']['ids'])
                unique_triplets = np.unique(np.stack((
                    anno['relations']['subj_ids'],
                    anno['relations']['obj_ids'],
                    anno['relations']['ids']
                ), axis=1), axis=0)
                prob_matrix[
                    ids[unique_triplets[:, 0]],
                    ids[unique_triplets[:, 1]],
                    unique_triplets[:, 2]] += 1
        prob_matrix /= prob_matrix.sum(2)[:, :, None]
        if with_bg:
            with open(self._predcls_probability_json, 'w') as fid:
                json.dump(prob_matrix.tolist(), fid)
        else:
            with open(self._preddet_probability_json, 'w') as fid:
                json.dump(prob_matrix.tolist(), fid)

    @staticmethod
    def _transform_annotations(annos):
        """
        Transform relationship annotations.

        Returns a list of dicts:
            {
                'filename': filename (no path),
                'split_id': int, 0/1/2 for train/val/test,
                'height': int, image height in pixels,
                'width': int, image width in pixels,
                'im_scale': float, resize scale for Faster-RCNN,
                'objects': {
                    'names': list of str,
                    'boxes': list of 4-tuples
                },
                'relations': {
                    'names': list of str,
                    'subj_ids': list of int,
                    'obj_ids': list of int
                }
            }
        """
        objects = [
            {
                obj_tuple: o
                for o, obj_tuple in enumerate(sorted(list(
                    set(
                        (tuple(rel['subject_box']), rel['subject'])
                        for rel in anno['relationships']
                    ).union(set(
                        (tuple(rel['object_box']), rel['object'])
                        for rel in anno['relationships']
                    ))
                ), key=lambda t: (t[0][2] + t[0][3])))
            }
            for anno in annos
        ]
        inv_objects = [
            {v: k for k, v in obj_dict.items()} for obj_dict in objects]
        relationships = [
            {
                'names': [rel['predicate'] for rel in anno['relationships']],
                'subj_ids': [
                    obj_dict[(tuple(rel['subject_box']), rel['subject'])]
                    for rel in anno['relationships']],
                'obj_ids': [
                    obj_dict[(tuple(rel['object_box']), rel['object'])]
                    for rel in anno['relationships']]
            }
            for anno, obj_dict in zip(annos, objects)
        ]
        return [
            {
                'filename': anno['filename'],
                'split_id': anno['split_id'],
                'height': anno['height'],
                'width': anno['width'],
                'im_scale': anno['im_scale'],
                'objects': {
                    'names': [objs[o][1] for o in sorted(objs.keys())],
                    'boxes': [objs[o][0] for o in sorted(objs.keys())]
                },
                'relations': relations
            }
            for anno, relations, objs in zip(annos, relationships, inv_objects)
        ]
