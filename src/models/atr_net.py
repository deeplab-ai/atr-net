# -*- coding: utf-8 -*-
"""
Attention-Translation-Relation Network.

Model of our ICCVW 2019 paper:
"Attention-Translation-Relation Network for Scalable Scene Graph
Generation",
Authors:
Gkanatsios N., Pitsikalis V., Koutras P., Maragos P..

Code by: N. Gkanatsios
"""

from time import time

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

from config import USE_CUDA
from src.utils.train_test_utils import SGGTrainTester
from src.tools.early_stopping_scheduler import EarlyStopping
from src.models.object_classifier import ObjectClassifier


class ATRNet(nn.Module):
    """ATR-Net main."""

    def __init__(self, num_classes, train_top=False, attention='multi_head',
                 use_language=True, use_spatial=True):
        """Initialize model."""
        super().__init__()
        self.p_branch = PredicateBranch(
            num_classes, attention,
            use_language=use_language, use_spatial=use_spatial)
        self.os_branch = ObjectSubjectBranch(
            num_classes, attention,
            use_language=use_language, use_spatial=use_spatial)
        self.fc_fusion = nn.Sequential(
            nn.Linear(2 * num_classes, 100), nn.ReLU(),
            nn.Linear(100, num_classes)
        )
        self.fc_bin_fusion = nn.Linear(2 * 2, 2)
        self.feat_extractor = ObjectClassifier(num_classes, train_top, False)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.mode = 'train'

    def forward(self, subj_feats, pred_feats, obj_feats, deltas,
                subj_masks, obj_masks, subj_embs, obj_embs):
        """Forward pass."""
        masks = torch.cat((subj_masks, obj_masks), dim=1)
        pred_scores, bin_pred_scores = self.p_branch(
            pred_feats, deltas, masks, subj_embs, obj_embs)
        os_scores, bin_os_scores = self.os_branch(
            subj_feats, obj_feats, subj_embs, obj_embs, masks, deltas)
        scores = self.fc_fusion(torch.cat((pred_scores, os_scores), dim=1))
        bin_scores = self.fc_bin_fusion(torch.cat(
            (bin_pred_scores, bin_os_scores), dim=1))
        if self.mode == 'test':  # scores across pairs are compared in R_70
            scores = self.softmax(scores)
            pred_scores = self.softmax(pred_scores)
            os_scores = self.softmax(os_scores)
            bin_scores = self.softmax(bin_scores)
            bin_pred_scores = self.softmax(bin_pred_scores)
            bin_os_scores = self.softmax(bin_os_scores)
        return (
            scores, pred_scores, os_scores,
            bin_scores, bin_pred_scores, bin_os_scores
        )

    def feat_forward(self, base_features, im_info, rois):
        """Forward of feature extractor."""
        _, top_features, _, _, _ = self.feat_extractor(
            base_features, im_info, rois)
        return top_features


class PredicateBranch(nn.Module):
    """
    Predicate Branch.

    attention: multihead, singlehead, None
    """

    def __init__(self, num_classes, attention='multi_head',
                 use_language=True, use_spatial=True):
        """Initialize model."""
        super().__init__()
        self._attention_type = attention
        _use_cuda = USE_CUDA and torch.cuda.is_available()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(2048, 256, 1), nn.ReLU())
        self.conv_1b = nn.Sequential(
            nn.Conv2d(2048, 256, 1), nn.ReLU())
        self.attention_layer = AttentionLayer(
            use_language and attention is not None,
            use_spatial and attention is not None)
        self.pooling_weights = AttentionalWeights(
            num_classes, feature_dim=256, attention_type=attention)
        self.binary_pooling_weights = AttentionalWeights(
            2, feature_dim=256, attention_type=attention)
        self.attentional_pooling = AttentionalPoolingLayer()
        self.conv_2 = nn.Sequential(nn.Conv1d(256, 128, 1), nn.ReLU())
        self.conv_2b = nn.Sequential(nn.Conv1d(256, 128, 1), nn.ReLU())
        if attention == 'multi_head':
            self.classifier_weights = AttentionalWeights(
                num_classes, feature_dim=128, attention_type=attention)
            _bias = (
                torch.rand(1, num_classes).cuda() if _use_cuda
                else torch.rand(1, num_classes))
            self.bias = nn.Parameter(_bias)
            self.binary_classifier_weights = AttentionalWeights(
                2, feature_dim=128, attention_type=attention)
            _binary_bias = (
                torch.rand(1, 2).cuda() if _use_cuda else torch.rand(1, 2))
            self.binary_bias = nn.Parameter(_binary_bias)
        else:
            self.classifier_weights = nn.Linear(128, num_classes)
            self.binary_classifier_weights = nn.Linear(128, 2)

    def forward(self, pred_feats, deltas, masks, subj_embs, obj_embs):
        """Forward pass."""
        attention = self.attention_layer(subj_embs, obj_embs, deltas, masks)
        conv_pred_feats = self.conv_1(pred_feats)
        bin_conv_pred_feats = self.conv_1b(pred_feats)
        if self._attention_type is not None:
            pred_feats = self.attentional_pooling(
                conv_pred_feats,
                self.pooling_weights(attention)
            )
            bin_pred_feats = self.attentional_pooling(
                bin_conv_pred_feats,
                self.binary_pooling_weights(attention)
            )
        else:
            pred_feats = conv_pred_feats.mean(3).mean(2).unsqueeze(-1)
            bin_pred_feats = bin_conv_pred_feats.mean(3).mean(2).unsqueeze(-1)
        pred_feats = self.conv_2(pred_feats)
        bin_pred_feats = self.conv_2b(bin_pred_feats)
        if self._attention_type == 'multi_head':
            return (
                torch.sum(
                    pred_feats * self.classifier_weights(attention),
                    dim=1)
                + self.bias,
                torch.sum(
                    bin_pred_feats * self.binary_classifier_weights(attention),
                    dim=1)
                + self.binary_bias
            )
        return (
            self.classifier_weights(pred_feats.view(-1, 128)),
            self.binary_classifier_weights(bin_pred_feats.view(-1, 128))
        )


class ObjectSubjectBranch(nn.Module):
    """Object-Subject Branch."""

    def __init__(self, num_classes, attention='multi_head',
                 use_language=True, use_spatial=True):
        """Initialize model."""
        super().__init__()
        self._attention_type = attention
        _use_cuda = USE_CUDA and torch.cuda.is_available()
        self.fc_subj = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.fc_obj = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.fc_subj_b = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.fc_obj_b = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.attention_layer = AttentionLayer(
            use_language and attention is not None,
            use_spatial and attention is not None)
        if attention == 'multi_head':
            self.classifier_weights = AttentionalWeights(
                num_classes, feature_dim=128, attention_type=attention)
            _bias = (
                torch.rand(1, num_classes).cuda() if _use_cuda
                else torch.rand(1, num_classes))
            self.bias = nn.Parameter(_bias)
            self.binary_classifier_weights = AttentionalWeights(
                2, feature_dim=128, attention_type=attention)
            _binary_bias = (
                torch.rand(1, 2).cuda() if _use_cuda else torch.rand(1, 2))
            self.binary_bias = nn.Parameter(_binary_bias)
        else:
            self.classifier_weights = nn.Linear(128, num_classes)
            self.binary_classifier_weights = nn.Linear(128, 2)

    def forward(self, subj_feats, obj_feats, subj_embs, obj_embs, masks,
                deltas):
        """Forward pass, return output scores."""
        attention = self.attention_layer(subj_embs, obj_embs, deltas, masks)
        os_feats = self.fc_obj(obj_feats) - self.fc_subj(subj_feats)
        os_feats = os_feats.unsqueeze(-1)
        bin_os_feats = self.fc_obj_b(obj_feats) - self.fc_subj_b(subj_feats)
        bin_os_feats = bin_os_feats.unsqueeze(-1)
        if self._attention_type == 'multi_head':
            return (
                torch.sum(
                    os_feats * self.classifier_weights(attention),
                    dim=1)
                + self.bias,
                torch.sum(
                    os_feats * self.binary_classifier_weights(attention),
                    dim=1)
                + self.binary_bias
            )
        return (
            self.classifier_weights(os_feats.view(-1, 128)),
            self.binary_classifier_weights(bin_os_feats.view(-1, 128))
        )


class AttentionalWeights(nn.Module):
    """Compute weights based on spatio-linguistic attention."""

    def __init__(self, num_classes, feature_dim, attention_type):
        """Initialize model."""
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feature_dim
        self.attention_type = attention_type
        if attention_type == 'multi_head':
            self.att_fc = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, feature_dim * num_classes), nn.ReLU())
        elif attention_type == 'single_head':
            self.att_fc = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, feature_dim), nn.ReLU())

    def forward(self, attention):
        """Forward pass."""
        if self.attention_type is None:
            return None
        if self.attention_type == 'single_head':
            return self.att_fc(attention).view(-1, self.feat_dim, 1)
        return self.att_fc(attention).view(-1, self.feat_dim, self.num_classes)


class AttentionLayer(nn.Module):
    """Drive attention using language and/or spatial features."""

    def __init__(self, use_language=True, use_spatial=True):
        """Initialize model."""
        super().__init__()
        self._use_language = use_language
        self._use_spatial = use_spatial
        self.fc_subject = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        self.fc_object = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        self.fc_lang = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU())
        self.mask_net = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, 8), nn.ReLU())
        self.fc_delta = nn.Sequential(
            nn.Linear(38, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU())
        self.fc_spatial = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU())

    def forward(self, subj_embs, obj_embs, deltas, masks):
        """Forward pass."""
        lang_attention, spatial_attention = None, None
        if self._use_language:
            lang_attention = self.fc_lang(torch.cat((
                self.fc_subject(subj_embs),
                self.fc_object(obj_embs)
            ), dim=1))
        if self._use_spatial:
            spatial_attention = self.fc_spatial(torch.cat((
                self.mask_net(masks).view(masks.shape[0], -1),
                self.fc_delta(deltas)
            ), dim=1))
        if self._use_language or self._use_spatial:
            attention = 0
            if self._use_language:
                attention = attention + lang_attention
            if self._use_spatial:
                attention = attention + spatial_attention
            return attention
        return None


class AttentionalPoolingLayer(nn.Module):
    """Attentional Pooling layer."""

    def __init__(self):
        """Initialize model."""
        super().__init__()
        self.register_buffer('const', torch.FloatTensor([0.0001]))
        self.softplus = nn.Softplus()

    def forward(self, features, weights):
        """
        Forward pass.

        Inputs:
            - features: tensor (batch_size, 256, 4, 4), the feature map
            - weights: tensor (batch, 256, num_classes),
                per-class attention weights
        """
        features = features.unsqueeze(-1)
        att_num = (  # (bs, 4, 4, num_classes)
            self.softplus(
                (features * weights.unsqueeze(2).unsqueeze(2)).sum(1))
            + self.const)
        att_denom = att_num.sum(2).sum(1)  # (bs, num_classes)
        attention_map = (  # (bs, 4, 4, num_classes)
            att_num
            / att_denom.unsqueeze(1).unsqueeze(2))
        return (attention_map.unsqueeze(1) * features).sum(3).sum(2)


class TrainTester(SGGTrainTester):
    """Extends SGGTrainTester."""

    def __init__(self, net, config, features):
        """Initialize instance."""
        super().__init__(net, config, features)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self._obj2vec = config.obj2vec
        if self._task == 'sgcls':
            self.obj_classifier = ObjectClassifier(
                config.num_obj_classes, False, False)
            checkpoint = torch.load(
                self._models_path
                + 'object_classifier_objcls_' + self._dataset + '.pt')
            self.obj_classifier.load_state_dict(checkpoint['model_state_dict'])
            if self._use_cuda:
                self.obj_classifier.cuda()
            else:
                self.obj_classifier.cpu()

    def _net_forward(self, batch, step):
        relations = self.data_loader.get('relations', batch, step)
        base_features = self.data_loader.get('base_features', batch, step)
        if self.net.mode == 'train' or self._task != 'sgcls':
            obj_vecs = self.data_loader.get('object_1hot_vectors', batch, step)
        elif self._task == 'sgcls':
            _, _, obj_vecs, _, _ = self.obj_classifier(
                base_features,
                self.data_loader.get('image_info', batch, step),
                self.data_loader.get('object_rcnn_rois', batch, step))
        obj_features = self.net.feat_forward(
            base_features,
            self.data_loader.get('image_info', batch, step),
            self.data_loader.get('object_rcnn_rois', batch, step))
        obj_features = obj_features.mean(3).mean(2)
        embeddings = self._obj2vec[obj_vecs.argmax(1)]
        deltas = self.data_loader.get('box_deltas', batch, step)
        masks = self.data_loader.get('object_masks', batch, step)
        pred_rois = self.data_loader.get('predicate_rcnn_rois', batch, step)
        if self.net.mode == 'train':
            deltas = deltas[:44 ** 2]
            pred_rois = pred_rois[:, :44 ** 2, :]
            relations = relations[:44 ** 2]
        rel_batch_size = 128
        scores, subject_scores, object_scores = [], [], []
        p_scores, os_scores = [], []
        bin_scores, bin_p_scores, bin_os_scores = [], [], []
        for btch in range(1 + (len(relations) - 1) // rel_batch_size):
            rels = relations[btch * rel_batch_size:(btch + 1) * rel_batch_size]
            pred_features = self.net.feat_forward(
                base_features,
                self.data_loader.get('image_info', batch, step),
                pred_rois[
                    :, btch * rel_batch_size:(btch + 1) * rel_batch_size, :])
            (
                b_scores, b_p_scores, b_os_scores,
                b_bin_scores, b_bin_p_scores, b_bin_os_scores
            ) = self.net(
                torch.stack([obj_features[r[0]] for r in rels], dim=0),
                pred_features,
                torch.stack([obj_features[r[1]] for r in rels], dim=0),
                deltas[btch * rel_batch_size:(btch + 1) * rel_batch_size],
                torch.stack([masks[r[0]] for r in rels], dim=0),
                torch.stack([masks[r[1]] for r in rels], dim=0),
                torch.stack([embeddings[r[0]] for r in rels], dim=0),
                torch.stack([embeddings[r[1]] for r in rels], dim=0)
            )
            scores.append(b_scores)
            p_scores.append(b_p_scores)
            os_scores.append(b_os_scores)
            bin_scores.append(b_bin_scores)
            bin_p_scores.append(b_bin_p_scores)
            bin_os_scores.append(b_bin_os_scores)
            subject_scores.append(
                torch.stack([obj_vecs[r[0]] for r in rels], dim=0))
            object_scores.append(
                torch.stack([obj_vecs[r[1]] for r in rels], dim=0))
        return (
            torch.cat(scores, dim=0),
            torch.cat(subject_scores, dim=0), torch.cat(object_scores, dim=0),
            torch.cat(p_scores, dim=0), torch.cat(os_scores, dim=0),
            torch.cat(bin_scores, dim=0),
            torch.cat(bin_p_scores, dim=0), torch.cat(bin_os_scores, dim=0)
        )

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        (
            scores, _, _, p_scores, os_scores,
            bin_scores, bin_p_scores, bin_os_scores
        ) = self._net_forward(batch, step)
        targets = self.data_loader.get('predicate_targets', batch, step)[:1936]
        bg_targets = self.data_loader.get('bg_targets', batch, step)[:44 ** 2]
        if len(bg_targets[bg_targets == 1]) >= 1:
            scale = len(bg_targets) / len(bg_targets[bg_targets == 1])
        else:
            scale = 1
        loss = (
            self.criterion(scores, targets) * scale
            + self.criterion(p_scores, targets) * scale
            + self.criterion(os_scores, targets) * scale
            + self.cross_entropy(bin_scores, bg_targets)
            + self.cross_entropy(bin_p_scores, bg_targets)
            + self.cross_entropy(bin_os_scores, bg_targets))
        return loss

    def _net_outputs(self, batch, step):
        """Get network outputs for current batch."""
        probs, subj_probs, obj_probs, _, _, bin_probs, _, _ = \
            self._net_forward(batch, step)
        if self._task == 'preddet':
            bin_probs = self.data_loader.get('bg_targets', batch, step).float()
            return probs * bin_probs.unsqueeze(-1), subj_probs, obj_probs
        return probs * bin_probs[:, 1].unsqueeze(-1), subj_probs, obj_probs


def train_test(config):
    """Train and test a net."""
    config.logger.debug(
        'Tackling %s for %d classes' % (config.task, config.num_classes))
    net_params = config_net_params(config)
    print(net_params)
    net = ATRNet(
        num_classes=config.num_classes,
        train_top=config.train_top,
        attention=net_params['attention'],
        use_language=net_params['use_language'],
        use_spatial=net_params['use_spatial'])
    features = {
        'bg_targets',
        'box_deltas',
        'images',
        'image_info',
        'object_1hot_vectors',
        'object_masks',
        'object_rcnn_rois',
        'predicate_rcnn_rois',
        'predicate_targets',
        'relations'
    }
    train_tester = TrainTester(net, config, features)
    top_net_params = [
        param for name, param in net.named_parameters()
        if 'top_net' in name and param.requires_grad]
    other_net_params = [
        param for name, param in net.named_parameters()
        if 'top_net' not in name and param.requires_grad]
    optimizer = optim.Adam(
        [
            {'params': top_net_params, 'lr': 0.0002},
            {'params': other_net_params}
        ], lr=0.002, weight_decay=config.weight_decay)
    if config.use_early_stopping:
        scheduler = EarlyStopping(
            optimizer, factor=0.3, patience=1, max_decays=3)
    else:
        scheduler = MultiStepLR(optimizer, [3, 6], gamma=0.3)
    t_start = time()
    train_tester.train(
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(
            ignore_index=config.num_classes - 1, reduction='none'),
        scheduler=scheduler,
        epochs=30 if config.use_early_stopping else 7)
    config.logger.info('Training time: ' + str(time() - t_start))
    train_tester.net.mode = 'test'
    t_start = time()
    train_tester.test()
    config.logger.info('Test time: ' + str(time() - t_start))


def config_net_params(config):
    """Configure net parameters."""
    net_params = {
        'attention': 'multi_head',
        'use_language': True,
        'use_spatial': True
    }
    if 'single_head' in config.net_name:
        net_params['attention'] = 'single_head'
    if 'no_att' in config.net_name:
        net_params['attention'] = None
    if 'no_lang' in config.net_name:
        net_params['use_language'] = False
    if 'no_spat' in config.net_name:
        net_params['use_spatial'] = False
    return net_params
