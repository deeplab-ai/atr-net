# -*- coding: utf-8 -*-
"""Functions to extract spatial features for objects/relationships."""

import numpy as np


def create_binary_masks(boxes, img_height, img_width, mask_size=32):
    """Create binary masks that are non-zero inside boxes."""
    masks = np.zeros((len(boxes), 1, mask_size, mask_size))
    h_ratio = float(mask_size) / img_height  # height ratio
    w_ratio = float(mask_size) / img_width  # width ratio
    y_min = np.maximum(0, np.floor(boxes[:, 0] * h_ratio)).astype(int)
    y_max = np.minimum(
        mask_size - 1, np.ceil(boxes[:, 1] * h_ratio)
    ).astype(int) + 1
    x_min = np.maximum(0, np.floor(boxes[:, 2] * w_ratio)).astype(int)
    x_max = np.minimum(
        mask_size - 1, np.ceil(boxes[:, 3] * w_ratio)
    ).astype(int) + 1
    for ind, _ in enumerate(masks):
        masks[ind, 0, y_min[ind]:y_max[ind], x_min[ind]:x_max[ind]] = 1.0
    return masks


def get_box_deltas(subj_boxes, obj_boxes, pred_boxes, height, width):
    """
    Another spatial feature.

    (D(S,O), D(S,P), D(O,P)), with D(S,O)=
    (xs-xo)/ws, (ys-yo)/hs, log(ws/wo),log(hs/ho), (xo-xs)/wo, (yo-ys)/ho
    """
    x_subj = (subj_boxes[:, 2] + subj_boxes[:, 3]) / 2
    y_subj = (subj_boxes[:, 0] + subj_boxes[:, 1]) / 2
    x_pred = (pred_boxes[:, 2] + pred_boxes[:, 3]) / 2
    y_pred = (pred_boxes[:, 0] + pred_boxes[:, 1]) / 2
    x_obj = (obj_boxes[:, 2] + obj_boxes[:, 3]) / 2
    y_obj = (obj_boxes[:, 0] + obj_boxes[:, 1]) / 2
    w_subj = subj_boxes[:, 3] - subj_boxes[:, 2]
    h_subj = subj_boxes[:, 1] - subj_boxes[:, 0]
    w_pred = pred_boxes[:, 3] - pred_boxes[:, 2]
    h_pred = pred_boxes[:, 1] - pred_boxes[:, 0]
    w_obj = obj_boxes[:, 3] - obj_boxes[:, 2]
    h_obj = obj_boxes[:, 1] - obj_boxes[:, 0]
    return np.stack((
        (x_subj - x_obj) / w_subj, (y_subj - y_obj) / h_subj,
        np.log(w_subj / w_obj), np.log(h_subj / h_obj),
        (x_obj - x_subj) / w_obj, (y_obj - y_subj) / h_obj,
        (x_subj - x_pred) / w_subj, (y_subj - y_pred) / h_subj,
        np.log(w_subj / w_pred), np.log(h_subj / h_pred),
        (x_pred - x_subj) / w_pred, (y_pred - y_subj) / h_pred,
        (x_obj - x_pred) / w_obj, (y_obj - y_pred) / h_obj,
        np.log(w_obj / w_pred), np.log(h_obj / h_pred),
        (x_pred - x_obj) / w_pred, (y_pred - y_obj) / h_pred,
        subj_boxes[:, 0] / height, subj_boxes[:, 1] / height,
        subj_boxes[:, 2] / width, subj_boxes[:, 3] / width,
        obj_boxes[:, 0] / height, obj_boxes[:, 1] / height,
        obj_boxes[:, 2] / width, obj_boxes[:, 3] / width,
        pred_boxes[:, 0] / height, pred_boxes[:, 1] / height,
        pred_boxes[:, 2] / width, pred_boxes[:, 3] / width,
        w_subj * h_subj / (height * width),
        w_obj * h_obj / (height * width),
        w_pred / w_subj, h_pred / h_subj,
        w_pred / w_obj, h_pred / h_obj,
        w_obj / w_subj, h_obj / h_subj
    ), axis=1)


def _compute_predicate_boxes(boxes, subj_ids, obj_ids, box_type):
    subj_boxes = boxes[subj_ids]
    obj_boxes = boxes[obj_ids]
    return np.stack([
        np.minimum(subj_boxes[:, 0], obj_boxes[:, 0]),
        np.maximum(subj_boxes[:, 1], obj_boxes[:, 1]),
        np.minimum(subj_boxes[:, 2], obj_boxes[:, 2]),
        np.maximum(subj_boxes[:, 3], obj_boxes[:, 3])
    ], axis=1)
