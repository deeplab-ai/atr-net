# -*- coding: utf-8 -*-
"""Functions for training and testing a network."""

import json
import os
from time import time

from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.utils.data_loading_utils import (
    SGGDataset, SGGDataLoader, sgg_collate_fn)
from src.utils.file_utils import load_annotations
from src.utils.metric_utils import (
    RelationshipEvaluator, ObjectClsEvaluator, ObjectDetEvaluator)


class SGGTrainTester():
    """
    Train and test utilities for Scene Graph Generation.

    Inputs upon initialization:
        - net: PyTorch nn.Module, the network to train/test
        - config: class Config, see config.py
        - features: set of str, features to load for net
    """

    def __init__(self, net, config, features):
        """Initiliaze train/test instance."""
        self.net = net
        self.config = config
        self.features = features
        self._set_from_config(config)

    def _set_from_config(self, config):
        """Load config variables."""
        self._annotations_per_batch = config.annotations_per_batch
        self._apply_dynamic_lr = config.apply_dynamic_lr
        self._backbone = config.backbone
        self._batch_size = config.batch_size
        self._dataset = config.dataset
        self._figures_path = config.paths['figures_path']
        self._filter_duplicate_rels = config.filter_duplicate_rels
        self._filter_multiple_preds = config.filter_multiple_preds
        self._json_path = config.paths['json_path']
        self._loss_path = config.paths['loss_path']
        self._models_path = config.paths['models_path']
        self._net_name = config.net_name
        self._num_workers = config.num_workers
        self._phrase_recall = config.phrase_recall
        self._restore_on_plateau = config.restore_on_plateau
        self._results_path = config.paths['results_path']
        self._task = config.task
        self._use_cuda = config.use_cuda
        self._use_early_stopping = config.use_early_stopping
        self.logger = config.logger

    def train(self, optimizer, criterion=None, scheduler=None, epochs=1):
        """Train a neural network if it does not already exist."""
        self.logger.info("Performing training for " + self._net_name)
        self.optimizer = optimizer
        if criterion is not None:
            self.criterion = criterion.cuda() if self._use_cuda else criterion
        self.scheduler = scheduler

        # Check if the model is already trained
        model_path_name = self._models_path + self._net_name + '.pt'
        if self._check_for_existent_model(model_path_name):
            return self.net

        # Check for existent checkpoint
        checkpoints, loss_history = self._check_for_checkpoint(model_path_name)
        epochs = list(range(epochs))[max(checkpoints):]

        # Settings and loading
        self.net.train()
        if self._use_cuda:
            self.net.cuda()
        self._set_data_loaders()
        self.data_loader = self._data_loaders['train']
        self.logger.debug("Batch size is " + str(self._batch_size))

        # Main training procedure
        for epoch in epochs:
            loss_history, keep_training = self._train_epoch(
                epoch, model_path_name, loss_history
            )
            if not keep_training:
                self.logger.info('Model converged, exit training')
                break

        # Training is complete, save model and plot loss curves
        self._save_model(model_path_name)
        self.logger.info('Finished Training')
        if any(loss_history):
            self.plot_loss(loss_history)
        return self.net

    def _train_epoch(self, epoch, model_path_name, loss_history):
        """Train the network for one epoch."""
        epoch_start = time()
        keep_training = True

        # Adjust learning rate
        if self.scheduler is not None and not self._use_early_stopping:
            self.scheduler.step()
        for param_group in self.optimizer.param_groups:
            param_group['base_lr'] = param_group['lr']
        curr_lr = max(p['lr'] for p in self.optimizer.param_groups)
        self.logger.debug("Learning rate is now " + str(curr_lr))

        # Main epoch pipeline
        accum_loss = 0
        for batch in tqdm(self.data_loader):
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward + Backward + Optimize on batch data
            loss = self._compute_train_loss(batch)
            loss.backward()
            self.optimizer.step()
            accum_loss += loss.item()

        # After each epoch: reset lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['base_lr']
        val_loss = self._compute_validation_loss().item()
        if self._use_early_stopping:
            ret_epoch, keep_training = self.scheduler.step(val_loss)
            if ret_epoch < epoch:
                if self._restore_on_plateau or not keep_training:
                    self._load_model(model_path_name, epoch=ret_epoch,
                                     restore={'net', 'optimizer'})
                    self.scheduler.reduce_lr()

        # Print training statistics
        accum_loss /= len(self.data_loader)
        loss_history.append((accum_loss, val_loss, curr_lr))
        self.logger.info(
            '[Epoch %d] loss: %.3f, validation loss: %.3f, run time: %.2fs'
            % (epoch, accum_loss, val_loss, time() - epoch_start)
        )
        self._save_model(model_path_name, epoch=epoch)
        with open(self._loss_path + self._net_name + '.json', 'w') as fid:
            json.dump(loss_history, fid)
        return loss_history, keep_training

    @torch.no_grad()
    def test(self):
        """Test a neural network."""
        # Settings and loading
        self.logger.info(
            "Test %s on %s on %s" % (self._net_name, self._task, self._dataset)
        )
        self.net.eval()
        if self._use_cuda:
            self.net.cuda()
        self.features = self.features.union({'boxes', 'labels'})
        self._set_data_loaders()
        self.data_loader = self._data_loaders['test']
        rel_eval = RelationshipEvaluator(self._dataset)

        # Forward pass on test set
        results = {}
        for batch in tqdm(self.data_loader):
            for step in range(len(batch['filenames'])):
                boxes = batch['boxes'][step]
                pred_scores, subj_scores, obj_scores = \
                    self._net_outputs(batch, step)
                scores = pred_scores.cpu().numpy()
                labels = batch['labels'][step]
                if self._task not in {'preddet', 'predcls'}:
                    subj_scores = subj_scores.cpu().numpy()
                    obj_scores = obj_scores.cpu().numpy()
                    scores = (
                        np.max(subj_scores, axis=1)[:, None]
                        * scores
                        * np.max(obj_scores, axis=1)[:, None])
                    labels[:, 0] = np.argmax(subj_scores, axis=1)
                    labels[:, 2] = np.argmax(obj_scores, axis=1)
                filename = batch['filenames'][step]
                rel_eval.step(filename, scores, labels, boxes,
                              self._phrase_recall)
                results.update({
                    filename: {
                        'boxes': boxes.tolist(),
                        'labels': labels.tolist(),
                        'scores': scores.tolist()
                    }
                })
        # Print metrics and save results
        rel_eval.print_stats(self._task)
        with open(self._results_path + self._net_name + '.json', 'w') as fid:
            json.dump(results, fid)

    def plot_loss(self, loss_history):
        """
        Plot training and validation loss.

        loss_history is a list of 3-element tuples, like
        (train_loss, val_loss, lr)
        """
        train_loss = [loss for loss, _, _ in loss_history]
        validation_loss = [val_loss for _, val_loss, _ in loss_history]
        lrs = [lr for _, _, lr in loss_history]
        _, axs = plt.subplots()
        axs.plot(train_loss)
        axs.plot(validation_loss, 'orange')
        axs.plot(lrs, 'green')
        plt.title(self._net_name + ' Loss Curves')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train Loss', 'Val. Loss', 'lr'], loc='upper left')
        plt.savefig(
            self._figures_path + self._net_name + '_Loss.jpg',
            bbox_inches='tight'
        )
        plt.close()

    def _check_for_existent_model(self, model_path_name):
        """Check if a trained model is existent."""
        if os.path.exists(model_path_name):
            self._load_model(model_path_name)
            self.logger.debug("Found existing trained model.")
            return True
        return False

    def _check_for_checkpoint(self, model_path_name):
        """Check if an intermediate checkpoint exists."""
        epochs_to_resume = [
            int(name[(len(self._net_name) + 6): -3]) + 1
            for name in os.listdir(self._models_path)
            if name.startswith(self._net_name + '_epoch')
            and name[(len(self._net_name) + 6): -3].isdigit()
        ] + [0]
        loss_history = []
        if any(epochs_to_resume):
            self._load_model(model_path_name, epoch=max(epochs_to_resume) - 1)
            self.logger.debug(
                'Found checkpoint for epoch: %d' % (max(epochs_to_resume) - 1))
            self.net.train()
            with open(self._loss_path + self._net_name + '.json') as fid:
                loss_history = json.load(fid)
            loss_history = loss_history[:max(epochs_to_resume)]
        return epochs_to_resume, loss_history

    @staticmethod
    def _compute_loss(batch, step):
        """Compute loss for current batch and step."""
        return 0

    def _compute_train_loss(self, batch):
        """Compute train loss."""
        accum_loss = [
            self._compute_loss(batch, step)
            for step in range(len(batch['filenames']))
        ]
        if self._apply_dynamic_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = (
                    param_group['base_lr']
                    * sum(len(s) for s in accum_loss)
                    / self._annotations_per_batch)
        return (
            sum(torch.sum(loss) for loss in accum_loss)
            / sum(len(loss) for loss in accum_loss)
        )

    @torch.no_grad()
    def _compute_validation_loss(self):
        """Compute validation loss."""
        self.net.eval()
        self.data_loader = self._data_loaders['val']
        accum_loss = [
            self._compute_loss(batch, step)
            for batch in self.data_loader
            for step in range(len(batch['filenames']))
        ]
        self.net.train()
        self.data_loader = self._data_loaders['train']
        return (
            sum(torch.sum(loss) for loss in accum_loss)
            / sum(len(loss) for loss in accum_loss)
        )

    def _load_model(self, model_path_name, epoch=None,
                    restore={'net', 'optimizer', 'scheduler'}):
        """Load a checkpoint, possibly referring to specific epoch."""
        if epoch is not None:
            checkpoint = torch.load(
                model_path_name[:-3] + '_epoch' + str(epoch) + '.pt')
        else:
            checkpoint = torch.load(model_path_name)
        if 'net' in restore:
            self.net.load_state_dict(checkpoint['model_state_dict'])
        if self._use_cuda:
            self.net.cuda()
        else:
            self.net.cpu()
        if 'optimizer' in restore:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler' in restore:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def _save_model(self, model_path_name, epoch=None):
        """Save a checkpoint, possibly referring to specific epoch."""
        checkpoint = {
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        if epoch is not None:
            torch.save(
                checkpoint,
                model_path_name[:-3] + '_epoch' + str(epoch) + '.pt')
        else:
            torch.save(checkpoint, model_path_name)

    def _net_outputs(self, batch, step):
        """Get network outputs for current batch and step."""
        return self.net()

    def _set_data_loaders(self):
        mode_ids = {'train': 0, 'val': 1, 'test': 2}
        annotations = np.array(load_annotations(
            self._task, self._dataset,
            self._filter_duplicate_rels, self._filter_multiple_preds)[0])
        split_ids = np.array([anno['split_id'] for anno in annotations])
        if self._dataset != 'VRD':
            datasets = {
                split: SGGDataset(
                    annotations[split_ids == split_id].tolist(),
                    self.config, self.features)
                for split, split_id in mode_ids.items()
            }
        else:
            datasets = {
                'train': SGGDataset(
                    annotations[split_ids == 0].tolist()
                    + annotations[split_ids == 1].tolist(),
                    self.config, self.features),
                'val': SGGDataset(
                    annotations[split_ids == 2].tolist(),
                    self.config, self.features),
                'test': SGGDataset(
                    annotations[split_ids == 2].tolist(),
                    self.config, self.features)
            }
        self._data_loaders = {
            split: SGGDataLoader(
                datasets[split], batch_size=self._batch_size,
                shuffle=split == 'train', num_workers=self._num_workers,
                drop_last=split in {'train', 'val'},
                collate_fn=lambda data: sgg_collate_fn(data, self.features),
                use_cuda=self._use_cuda)
            for split in mode_ids
        }


class ObjClsTrainTester(SGGTrainTester):
    """Train and test utilities for Object Classification."""

    def __init__(self, net, config, features):
        """Initiliaze train/test instance."""
        super().__init__(net, config, features)

    @torch.no_grad()
    def test(self):
        """Test a neural network."""
        # Settings and loading
        self.logger.info(
            "Test %s on %s on %s" % (self._net_name, self._task, self._dataset)
        )
        self.net.eval()
        if self._use_cuda:
            self.net.cuda()
        self._set_data_loaders()
        self.data_loader = self._data_loaders['test']
        obj_eval = ObjectClsEvaluator(self._dataset)

        # Forward pass on test set, epoch=0
        results = {}
        for batch in tqdm(self.data_loader):
            for step in range(len(batch['filenames'])):
                scores = self._net_outputs(batch, step).cpu().numpy()
                filename = batch['filenames'][step]
                obj_eval.step(filename, scores)
                results.update({filename: {'scores': scores.tolist()}})
        # Print metrics and save results
        obj_eval.print_stats()
        with open(self._results_path + self._net_name + '.json', 'w') as fid:
            json.dump(results, fid)


class ObjDetTrainTester(SGGTrainTester):
    """Train and test utilities for Object Detection."""

    def __init__(self, net, config, features):
        """Initiliaze train/test instance."""
        super().__init__(net, config, features)

    @torch.no_grad()
    def test(self):
        """Test a neural network."""
        # Settings and loading
        self.logger.info(
            "Test %s on %s on %s" % (self._net_name, self._task, self._dataset)
        )
        self.net.eval()
        if self._use_cuda:
            self.net.cuda()
        self._set_data_loaders()
        self.data_loader = self._data_loaders['test']
        obj_eval = ObjectDetEvaluator(self._dataset)
        anno_constructor = SGGenAnnosConstructor(self._dataset, self.config)
        max_per_img = self.config.max_obj_dets_per_img

        # Forward pass on test set, epoch=0
        results = {}
        for batch in tqdm(self.data_loader):
            for step in range(len(batch['filenames'])):
                scores, bboxes, labels = self._net_outputs(batch, step)
                scores = scores.cpu().numpy()
                bboxes = bboxes.cpu().numpy()
                labels = labels.cpu().numpy()
                score_sort = scores.argsort()[::-1][:max_per_img]
                labels = labels[score_sort]
                bboxes = bboxes[score_sort]
                scores = scores[score_sort]
                # boxes: (x1, y1, x2, y2) back to (y1, y2, x1, x2)
                bboxes = bboxes[:, (1, 3, 0, 2)]
                filename = batch['filenames'][step]
                obj_eval.step(filename, scores, bboxes, labels)
                anno_constructor.step(filename, scores, bboxes, labels)
                results.update({
                    filename: {
                        'scores': scores.tolist(),
                        'boxes': bboxes.tolist(),
                        'labels': labels.tolist()}})
        # Print metrics and save results
        obj_eval.print_stats()
        anno_constructor.save()
        with open(self._results_path + self._net_name + '.json', 'w') as fid:
            json.dump(results, fid)


class SGGenAnnosConstructor():
    """Create SGGen annotations with detected boxes."""

    def __init__(self, dataset, config):
        """Load dataset and keep test annotations."""
        self._dataset = dataset
        self._num_classes = config.num_rel_classes
        self._cls_names = np.array(config.obj_classes)
        self._json_path = config.paths['json_path']
        with open(self._json_path + dataset + '_preddet.json') as fid:
            annos = json.load(fid)
        self._annos = {anno['filename']: anno for anno in annos}

    def step(self, filename, scores, bboxes, labels):
        """Save a new image annotation."""
        if filename in self._annos:
            anno = dict(self._annos[filename])
            anno['objects'] = {
                'ids': labels.tolist(),
                'boxes': bboxes.tolist(),
                'names': self._cls_names[labels.astype(int)].tolist(),
                'scores': scores.tolist()
            }
            subj_ids, obj_ids = self._create_all_pairs(len(scores))
            anno['relations'] = {
                'ids': [self._num_classes - 1] * len(subj_ids),
                'names': ['__background__'] * len(subj_ids),
                'subj_ids': subj_ids.tolist(),
                'obj_ids': obj_ids.tolist()
            }
            self._annos[filename] = dict(anno)

    def save(self):
        """Save annotations to file."""
        with open(self._json_path + self._dataset + '_sggen.json', 'w') as fid:
            json.dump(list(self._annos.values()), fid)

    @staticmethod
    def _create_all_pairs(num_objects):
        """Create all possible combinations of objects."""
        obj_inds = np.arange(num_objects)
        return np.where(obj_inds[:, None] != obj_inds.T[None, :])
