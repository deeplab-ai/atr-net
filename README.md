# Attention-Translation-Relation Network for Scalable Scene Graph Generation

Code for our ICCV Scene Graph Representation and Learning Workshop paper, 2019.

## Requirements
Tested with Python 3.5 and 3.6.
Tested versions of important requirements:
* numpy==1.16.2
* torch==1.0.1
* opencv-python==4.0.0.21

Other packages you may need:
* colorlog==4.0.2
* h5py==2.9.0
* matplotlib==3.0.3
* xmltodict==0.12.0

## Setup
1. Clone the repository
```
git clone https://github.com/deeplab-ai/atr-net.git
cd atr-net
```
2. Setup Faster-RCNN
```
./scripts/setup_faster_rcnn.sh
```
3. Download images (edit ./scripts/download_images.sh to select datasets to download)
```
./scripts/download_images.sh
```
4. Download annotations (edit ./scripts/download_data.sh to select datasets)
```
./scripts/download_data.sh
```
5. Download GloVe
```
./scripts/download_glove_vectors.sh
```
6. Transform annotations and create project folders (edit prepare_data.py to select datasets)
```
python3 prepare_data.py
```

## Train/test a model
```
python3 main.py --dataset=DATASET --task=TASK --model=MODEL
```
See main.py for other input arguments.
Also see config.py for choices on dataset and task arguments.
Example:
```
python3 main.py --dataset=VG200 --task=predcls --model=atr_net
```

## For further questions
Open an issue!
