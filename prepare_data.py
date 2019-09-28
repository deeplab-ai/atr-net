# -*- coding: utf-8 -*-
"""Functions to transform annotations and create train/test dataset."""

import os

from config import Config
from src.dataset_tools.vg200_transformer_class import VG200Transformer
from src.dataset_tools.vg80k_transformer_class import VG80KTransformer
from src.dataset_tools.vgmsdn_transformer_class import VGMSDNTransformer
from src.dataset_tools.vgvte_transformer_class import VGVTETransformer
from src.dataset_tools.vrd_transformer_class import VRDTransformer
from src.dataset_tools.vrr_vg_transformer_class import VrRVGTransformer
from src.dataset_tools.svg_transformer_class import SVGTransformer

TRANSFORMERS = {
    'VG200': VG200Transformer(Config('VG200', '')),
    'VG80K': VG80KTransformer(Config('VG80K', '')),
    'VGMSDN': VGMSDNTransformer(Config('VGMSDN', '')),
    'VGVTE': VGVTETransformer(Config('VGVTE', '')),
    'VRD': VRDTransformer(Config('VRD', '')),
    'VrR-VG': VrRVGTransformer(Config('VrR-VG', '')),
    'sVG': SVGTransformer(Config('sVG', ''))
}


def main(datasets):
    """Run the data preprocessing and creation pipeline."""
    config = Config('VRD', '')
    if not os.path.exists(config.paths['figures_path']):
        os.mkdir(config.paths['figures_path'])
    if not os.path.exists(config.paths['json_path']):
        os.mkdir(config.paths['json_path'])
    if not os.path.exists(config.paths['loss_path']):
        os.mkdir(config.paths['loss_path'])
    if not os.path.exists(config.paths['models_path']):
        os.mkdir(config.paths['models_path'])
    if not os.path.exists(config.paths['results_path']):
        os.mkdir(config.paths['results_path'])
    for dataset in datasets:
        print('Creating annotations for ' + dataset)
        TRANSFORMERS[dataset].transform()
    print('Done.')

if __name__ == "__main__":
    main(['VG80K', 'sVG', 'VrR-VG', 'VGVTE', 'VGMSDN', 'VG200', 'VRD'])
