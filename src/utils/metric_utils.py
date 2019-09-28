# -*- coding: utf-8 -*-
"""Functions to compute different metrics for evaluation."""

from collections import defaultdict

import numpy as np

from src.utils.file_utils import load_annotations


class RelationshipEvaluator():
    """A class providing methods to evaluate the VRD-SGGen problem."""

    def __init__(self, dataset):
        """Initialize evaluator setup for this dataset."""
        self._recall_types = np.array([20, 50, 100])  # R@20, 50, 100
        self._max_recall = self._recall_types[-1]
        self.reset()

        # Ground-truth labels and boxes
        annos, zeroshot_annos = load_annotations('preddet', dataset)
        self._annos = {
            'full': {
                anno['filename']: anno
                for anno in annos if anno['split_id'] == 2},
            'zeroshot': {anno['filename']: anno for anno in zeroshot_annos}
        }

    def reset(self):
        """Initialize recall_counters."""
        self._gt_positive_counter = {'full': [], 'zeroshot': []}
        self._true_positive_counter = {
            (rmode, cmode, dmode): []
            for rmode in ('micro', 'macro')
            for cmode in ('graph constraints', 'no constraints')
            for dmode in ('full', 'zeroshot')
        }

    def step(self, filename, scores, labels, boxes, phrase_recall):
        """
        Evaluate relationship or phrase recall.

        Inputs:
            - filename: str, name of the image to evaluate
            - scores: array (n_det, n_classes)
            - labels: array (n_det, 3): [subj_cls, -1, obj_cls]
            - boxes: array (n_det, 2, 4)
        """
        # Update true positive counter and get gt labels-bboxes
        gt_labels, gt_bboxes = {}, {}
        for dmode in ('full', 'zeroshot'):  # data mode
            if filename not in self._annos[dmode].keys():
                continue
            self._gt_positive_counter[dmode].append(
                len(self._annos[dmode][filename]['relations']['ids']))
            gt_labels[dmode], gt_bboxes[dmode] = self._get_gt(filename, dmode)

        # Compute the different recall types
        for rmode in ('micro', 'macro'):  # recall mode
            for cmode in ('graph constraints', 'no constraints'):  # constraint
                if filename in self._annos['full'].keys():
                    det_labels, det_bboxes = self._sort_detected(
                        scores, boxes, labels,
                        cmode == 'graph constraints', rmode == 'macro')
                for dmode in ('full', 'zeroshot'):  # data mode
                    if filename not in self._annos[dmode].keys():
                        continue
                    self._true_positive_counter[(rmode, cmode, dmode)].append(
                        relationship_recall(
                            self._recall_types, det_labels, det_bboxes,
                            gt_labels[dmode],
                            gt_bboxes[dmode],
                            macro_recall=rmode == 'macro',
                            phrase_recall=phrase_recall))

    def print_stats(self, task):
        """Print recall statistics for given task."""
        for rmode in ('micro', 'macro'):
            for cmode in ('graph constraints', 'no constraints'):
                for dmode in ('full', 'zeroshot'):
                    print(
                        '%sRecall@20-50-100 %s %s with %s:'
                        % (rmode, task, dmode, cmode),
                        self._compute_recall(rmode, cmode, dmode)
                    )

    def _compute_recall(self, rmode, cmode, dmode):
        """Compute micro or macro recall."""
        if rmode == 'micro':
            return (  # sum over tp / sum over gt
                100 * np.sum(
                    self._true_positive_counter[(rmode, cmode, dmode)],
                    axis=0)
                / np.sum(self._gt_positive_counter[dmode]))
        return (  # mean over (tp_i / gt_i) for each image i
            100 * np.mean(
                np.array(self._true_positive_counter[(rmode, cmode, dmode)])
                / np.array(self._gt_positive_counter[dmode])[:, None],
                axis=0))

    def _get_gt(self, filename, dmode):
        """
        Return ground truth labels and bounding boxes.

        - gt_labels: array (n_gt, 3), (subj, pred, obj)
        - gt_bboxes: array (n_t, 2, 4) (subj.-obj. boxes)
        """
        anno = self._annos[dmode][filename]
        gt_labels = np.stack((
            anno['objects']['ids'][anno['relations']['subj_ids']],
            anno['relations']['ids'],
            anno['objects']['ids'][anno['relations']['obj_ids']]
        ), axis=1)
        gt_bboxes = np.stack((
            anno['objects']['boxes'][anno['relations']['subj_ids']],
            anno['objects']['boxes'][anno['relations']['obj_ids']]
        ), axis=1)
        return gt_labels, gt_bboxes

    def _sort_detected(self, scores, boxes, labels,
                       graph_constraints=True, macro_recall=False):
        """
        Merge detected scores, labels and boxes to desired format.

        Inputs:
            - scores: array (n_det, n_classes)
            - boxes: array (n_det, 2, 4)
            - labels: array (n_det, 3): [subj_cls, -1, obj_cls]
            - graph_constraints: bool, when False, evaluate multilabel
            - macro_recall: bool, when True, clear duplicate detections
        Returns:
            - det_labels: array (N, 3), [subj_id, pred_id, obj_id]
            - det_bboxes: array (N, 2, 4), [subj_box, obj_box]
        """
        if macro_recall:  # clear duplicate detections
            _, unique_dets = np.unique(
                np.concatenate((labels, boxes.reshape(-1, 8), scores), axis=1),
                axis=0, return_index=True)
            scores = scores[unique_dets]
            boxes = boxes[unique_dets]
            labels = labels[unique_dets]

        scores = scores[:, :-1]  # clear background scores
        # Sort scores of each pair
        classes = np.argsort(scores)[:, ::-1]
        scores = np.sort(scores)[:, ::-1]
        if graph_constraints:  # only one prediction per pair
            classes = classes[:, :1]
            scores = scores[:, :1]

        # Sort across image and keep top-100 predictions
        top_detections_indices = np.unravel_index(
            np.argsort(scores, axis=None)[::-1][:self._max_recall],
            scores.shape)
        det_labels = labels[top_detections_indices[0]]
        det_labels[:, 1] = classes[top_detections_indices]
        det_bboxes = boxes[top_detections_indices[0]]
        return det_labels, det_bboxes

    def last_image_stats(self, rmode, cmode, dmode):
        """Get true and total positives and recall for last image."""
        # Get tp for R@100, so as to minimize threshold errors
        true_pos = self._true_positive_counter[(rmode, cmode, dmode)][-1][-1]
        total_pos = self._gt_positive_counter[dmode][-1]
        return (true_pos, total_pos, true_pos / total_pos)


class ObjectClsEvaluator():
    """A class providing methods to evaluate the ObjCls problem."""

    def __init__(self, dataset):
        """Initialize evaluator setup for this dataset."""
        self.reset()

        # Ground-truth labels and boxes
        annos, _ = load_annotations('preddet', dataset)
        self._annos = {
            anno['filename']: anno for anno in annos if anno['split_id'] == 2}

    def reset(self):
        """Initialize counters."""
        self._gt_positive_counter = []
        self._true_positive_counter = {'top-1': [], 'top-5': []}

    def step(self, filename, scores):
        """
        Evaluate accuracy for a given image.

        Inputs:
            - filename: str, name of the image to evaluate
            - scores: array (n_det, n_classes)
        """
        # Update true positive counter and get gt labels-bboxes
        if filename in self._annos.keys():
            self._gt_positive_counter.append(
                len(self._annos[filename]['objects']['ids']))
            gt_labels = self._annos[filename]['objects']['ids']

        # Compute the different recall types
        if filename in self._annos.keys():
            det_classes = np.argsort(scores)[:, ::-1]
            keep_top_1 = det_classes[:, 0] == gt_labels
            keep_top_5 = (det_classes[:, :5] == gt_labels[:, None]).any(1)
            self._true_positive_counter['top-1'].append(
                len(det_classes[keep_top_1]))
            self._true_positive_counter['top-5'].append(
                len(det_classes[keep_top_5]))

    def print_stats(self):
        """Print accuracy statistics."""
        for rmode in ('micro', 'macro'):
            for tmode in ('top-1', 'top-5'):
                print(
                    '%sAccuracy %s:'
                    % (rmode, tmode),
                    self._compute_acc(rmode, tmode)
                )

    def _compute_acc(self, rmode, tmode):
        """Compute micro or macro accuracy."""
        if rmode == 'micro':
            return (  # sum over tp / sum over gt
                100 * np.sum(
                    self._true_positive_counter[tmode],
                    axis=0)
                / np.sum(self._gt_positive_counter))
        return (  # mean over (tp_i / gt_i) for each image i
            100 * np.mean(
                np.array(self._true_positive_counter[tmode])
                / np.array(self._gt_positive_counter),
                axis=0))


class ObjectDetEvaluator():
    """A class providing methods to evaluate the ObjCls problem."""

    def __init__(self, dataset):
        """Initialize evaluator setup for this dataset."""
        self.reset()

        # Ground-truth labels and boxes
        annos, _ = load_annotations('preddet', dataset)
        self._annos = {
            anno['filename']: anno for anno in annos if anno['split_id'] == 2}

    def reset(self):
        """Initialize positive counters."""
        self._gt_positives = defaultdict(int)
        self._true_positives = defaultdict(list)
        self._scores = defaultdict(list)

    def step(self, filename, scores, boxes, labels):
        """
        Evaluate the detections of a given image.

        Inputs:
            - filename: str, name of the image to evaluate
            - scores: array (n_det,)
            - boxes: array (n_det, 4)
            - labels: array (ndet,)
        """
        # Sort detections based on their scores
        score_sort = scores.argsort()[::-1]
        labels = labels[score_sort]
        boxes = boxes[score_sort]
        scores = scores[score_sort]

        # Get gt annotations and update gt counter
        if filename in self._annos.keys():
            gt_boxes = self._annos[filename]['objects']['boxes']
            gt_classes = self._annos[filename]['objects']['ids']
            for cid in gt_classes:
                self._gt_positives[cid] += 1

        # Compute the different recall types
        if filename in self._annos.keys():
            tps = detection_precision(labels, boxes, gt_classes, gt_boxes)
            for cid, score, tp_value in zip(labels, scores, tps):
                self._true_positives[cid].append(tp_value)
                self._scores[cid].append(score)

    def print_stats(self):
        """Print mAP statistics."""
        print('Mean Average Precision:', self._compute_map())

    def _compute_map(self):
        """Compute mean average precision."""
        for name in self._gt_positives:
            if name not in self._true_positives:
                self._true_positives[name] = [0]
                self._scores[name] = [0]
            score_sort = np.argsort(self._scores[name])[::-1]
            self._true_positives[name] = np.array(
                self._true_positives[name]
            )[score_sort]
        aps = [
            voc_ap(
                np.cumsum(self._true_positives[name])
                / self._gt_positives[name],
                np.cumsum(self._true_positives[name])
                / np.cumsum(np.ones_like(self._true_positives[name]))
            )
            for name in sorted(self._gt_positives.keys())
        ]
        return np.mean(aps) * 100


def compute_area(bbox):
    """Compute area of box 'bbox' ([y_min, y_max, x_min, x_max])."""
    return max(0, bbox[3] - bbox[2] + 1) * max(0, bbox[1] - bbox[0] + 1)


def compute_overlap(det_bboxes, gt_bboxes):
    """
    Compute overlap of detected and ground truth boxes.

    Inputs:
        - det_bboxes: array (n, 4), n x [y_min, y_max, x_min, x_max]
            The detected bounding boxes for subject and object
        - gt_bboxes: array (n, 4), n x [y_min, y_max, x_min, x_max]
            The ground truth bounding boxes for subject and object
        n is 2 in case of relationship recall, 1 in case of phrases
    Returns:
        - overlap: non-negative float <= 1
    """
    overlaps = []
    for det_bbox, gt_bbox in zip(det_bboxes, gt_bboxes):
        intersection_bbox = [
            max(det_bbox[0], gt_bbox[0]),
            min(det_bbox[1], gt_bbox[1]),
            max(det_bbox[2], gt_bbox[2]),
            min(det_bbox[3], gt_bbox[3])
        ]
        intersection_area = compute_area(intersection_bbox)
        union_area = (compute_area(det_bbox)
                      + compute_area(gt_bbox)
                      - intersection_area)
        overlaps.append(intersection_area / union_area)
    return min(overlaps)


def create_phrase_boxes(bboxes):
    """Create predicate boxes given the subj. and obj. boxes."""
    return np.array([
        [[
            min(bbox[0][0], bbox[1][0]),
            max(bbox[0][1], bbox[1][1]),
            min(bbox[0][2], bbox[1][2]),
            max(bbox[0][3], bbox[1][3])
        ]]
        for bbox in bboxes  # (N, 2, 4)
    ])


def intersect_2d(arr_1, arr_2):
    """
    Return a boolean array of row matches.

    Given two arrays [m1, n] and [m2, n], return a [m1, m2] array where
    each entry is True if those rows match.
    """
    assert arr_1.shape[1] == arr_2.shape[1]
    return (arr_1[..., None] == arr_2.T[None, ...]).all(1)


def relationship_recall(chkpnts, det_labels, det_bboxes, gt_labels,
                        gt_bboxes, macro_recall=False, phrase_recall=False):
    """
    Evaluate relationship recall, with top n_re predictions per image.

    Inputs:
        - chkpnts: array, thresholds of predictions to keep
        - det_labels: array (Ndet, 3) of detected labels,
            where Ndet is the number of predictions in this image and
            each row: subj_tag, pred_tag, obj_tag]
        - det_bboxes: array (Ndet, 2, 4) of detected boxes,
            where Ndet is the number of predictions in this image and
            each 2x4 array: [
                [y_min_subj, y_max_subj, x_min_subj, x_max_subj]
                [y_min_obj, y_max_obj, x_min_obj, x_max_obj]
            ]
        - gt_labels: array (N, 3) of ground-truth labels,
            where N is the number of ground-truth in this image and
            each row: subj_tag, pred_tag, obj_tag]
        - gt_bboxes: array (N, 2, 4) of ground-truth boxes,
            where N is the number of ground-truth in this image and
            each 2x4 array: [
                [y_min_subj, y_max_subj, x_min_subj, x_max_subj]
                [y_min_obj, y_max_obj, x_min_obj, x_max_obj]
            ]
        - macro_recall: bool, whether to evaluate macro recall
        - phrase_recall: bool, whether to evaluate phrase recall
    Returns:
        - detected positives per top-N threshold
    """
    if phrase_recall:
        det_bboxes = create_phrase_boxes(det_bboxes)
        gt_bboxes = create_phrase_boxes(gt_bboxes)
    relationships_found = np.zeros_like(chkpnts, dtype=np.float)

    # Check only detections that match any of the ground-truth
    possible_matches = intersect_2d(det_labels, gt_labels)
    check_inds = possible_matches.any(1)
    for ind, bbox in zip(np.where(check_inds)[0], det_bboxes[check_inds]):
        overlaps = np.array([
            compute_overlap(bbox, gt_box) if match else 0
            for gt_box, match in zip(gt_bboxes, possible_matches[ind])
        ])
        if macro_recall:
            overlaps = np.where(overlaps >= 0.5)[0]
            possible_matches[:, overlaps] = False
            relationships_found[chkpnts > ind] += len(overlaps)
        elif (overlaps >= 0.5).any():  # micro-recall
            possible_matches[:, np.argmax(overlaps)] = False
            relationships_found[chkpnts > ind] += 1
    return relationships_found  # (R@20, R@50, R@100)


def detection_precision(det_labels, det_bboxes, gt_labels, gt_bboxes,
                        min_overlap=0.5):
    """
    Evaluate precision, detecting true positives.

    Inputs:
        - det_labels: array (Ndet,) of detected labels,
            where Ndet is the number of predictions in this image
        - det_bboxes: array (Ndet, 4) of detected boxes,
            where Ndet is the number of predictions in this image and
            each 1x4 array: [y_min, y_max, x_min, x_max]
        - gt_labels: array (N,) of ground-truth labels
        - gt_bboxes: array (N, 4) of ground-truth boxes
        - min_overlap: float, overlap threshold to consider detection
    Returns:
        - a binary list with 1 indicating a true positive
    """
    # Check only detections that match any of the ground-truth
    possible_matches = det_labels[..., None] == gt_labels.T[None, ...]
    check_inds = possible_matches.any(1)
    true_positives = np.copy(check_inds) * 1
    for ind, bbox in zip(np.where(check_inds)[0], det_bboxes[check_inds]):
        overlaps = np.array([
            compute_overlap([bbox], [gt_box]) if match else 0
            for gt_box, match in zip(gt_bboxes, possible_matches[ind])
        ])
        if (overlaps >= min_overlap).any():
            possible_matches[:, np.argmax(overlaps)] = False
        else:
            true_positives[ind] = 0
    return true_positives.tolist()


def voc_ap(recall, precision):
    """Code to compute Average Precision as given by PASCAL VOC 2012."""
    rec = np.zeros(len(recall) + 2)
    rec[1:-1] = recall
    rec[-1] = 1.0
    prec = np.zeros(len(precision) + 2)
    prec[1:-1] = precision
    # Make the precision monotonically decreasing
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    # Return the area under the curve (numerical integration)
    return np.sum((rec[1:] - rec[:-1]) * prec[1:])
