# -*- coding: utf-8 -*-
"""Model training/testing pipeline on a specific dataset and task."""

import argparse

import _init_paths
from config import Config
from src.models import (
    atr_net,
    object_classifier, object_detector
)

MODELS = {
    'atr_net': atr_net,
    'object_classifier': object_classifier,
    'object_detector': object_detector
}


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset codename (e.g. VG200)',
        type=str, default='VG200'
    )
    parser.add_argument(
        '--task', dest='task', help='Task, check main.py for supported tasks',
        type=str, default='preddet'
    )
    parser.add_argument(
        '--filter_duplicate_rels', dest='filter_duplicate_rels',
        help='Whether to filter relations annotated more than once',
        action='store_true'
    )
    parser.add_argument(
        '--filter_multiple_preds', dest='filter_multiple_preds',
        help='Whether to sample a single predicate per object pair',
        action='store_true'
    )
    parser.add_argument(
        '--annotations_per_batch', dest='annotations_per_batch',
        help='Batch size in terms of annotations (e.g. relationships)',
        type=int, default=128
    )
    parser.add_argument(
        '--batch_size', dest='batch_size',
        help='Batch size in terms of images',
        type=int
    )
    parser.add_argument(
        '--backbone', dest='backbone', help='Faster-RCNN backbone network',
        type=str, default='resnet'
    )
    parser.add_argument(
        '--num_workers', dest='num_workers',
        help='Number of workers employed by data loader',
        type=int, default=2
    )
    parser.add_argument(
        '--apply_dynamic_lr', dest='apply_dynamic_lr',
        help='Adapt learning rate so that lr / batch size = const',
        action='store_true'
    )
    parser.add_argument(
        '--not_use_early_stopping', dest='not_use_early_stopping',
        help='Plateau with early stopping learning rate policy',
        action='store_true'
    )
    parser.add_argument(
        '--restore_on_plateau', dest='restore_on_plateau',
        help='Adapt learning rate so that lr / batch size = const',
        action='store_false'
    )
    parser.add_argument(
        '--model', dest='model', help='Model to train (see main.py)',
        type=str, default='embeddings_net'
    )
    parser.add_argument(
        '--net_name', dest='net_name', help='Name of trained model',
        type=str, default=''
    )
    parser.add_argument(
        '--phrase_recall', dest='phrase_recall',
        help='Whether to evaluate phrase recall',
        action='store_true'
    )
    return parser.parse_args()


def main():
    """Train and test a network pipeline."""
    args = parse_args()
    model = MODELS[args.model]
    cfg = Config(
        dataset=args.dataset,
        task=args.task,
        filter_duplicate_rels=args.filter_duplicate_rels,
        filter_multiple_preds=args.filter_multiple_preds,
        annotations_per_batch=args.annotations_per_batch,
        batch_size=args.batch_size,
        backbone=args.backbone,
        num_workers=args.num_workers,
        apply_dynamic_lr=args.apply_dynamic_lr,
        use_early_stopping=not args.not_use_early_stopping,
        restore_on_plateau=args.restore_on_plateau,
        net_name=args.net_name if args.net_name else args.model,
        phrase_recall=args.phrase_recall
    )
    model.train_test(cfg)

if __name__ == "__main__":
    main()
