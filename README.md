# VisCOLL
Code and data for the paper "Visually grounded continual learning of compositional phrases, EMNLP 2020". [Paper](https://arxiv.org/abs/2005.00785)

Checkout out [Project website](https://inklab.usc.edu/viscoll-project/) for **data explorers** and the **leaderboard**!

## Installation

```
conda create -n viscoll-env python==3.7.5
conda activate viscoll-env
pip install -r requirements.txt
```

## Overview

VisCOLL proposes a problem setup and studies algorithms for continual learning and compositionality over visual-linguistic data. In VisCOLL, the model visits a stream of examples with an evolving data distribution over time and learn to perform masked phrases prediction. We create COCO-shift and Flickr-shift (based on COCO-captions and Flickr30k-entities) for study. 

<img src="https://inklab.usc.edu/viscoll-project/assets/img/overview2.png">


## Training and evaluation

This repo include code for running and evaluating ER/MIR/AGEM continual learning algorithms on VisCOLL datasets (COCO-shift and Flickr-shift), with VLBERT or LXMERT models.

For example, to train a VLBERT model with a memory of 10,000 examples on coco using Experience Replay (ER) algorithm, run:

```
python train_mlm.py --name debug --config configs/mlmcaptioning/er.yaml --seed 0 --cfg MLMCAPTION.BASE=vlbert OUTPUT_DIR=runs/
```

You may check `walkthough.ipynb` for a detailed walkthrough of training, inference, and evaluation.

## Data

We release the constructed data streams and scripts for visualization in the `datasets` folder.

## COCO shift

File name format: task_buffer_real_split_{1}_split_{2}_novel_comps_{3}_task_partition_any_objects.pkl

1. The official data split where examples are drawn from
2. How the dataset is applied during training. We separate out 5,000 examples from original train split as the validation examples. The official val split is used as the test split
3. Whether the data is the novel-composition split of 24 held-out concept pairs

The pickled file is python list where each item is a dict:
```
{'annotation': {'image_id': 374041, 'id': 31929, 'caption': 'Someone sitting on an elephant standing by a fence.  '}, 'task': 'elephant', 'mask': (3, 5)}
```

The `mask` is the text span that should be masked for prediction.  `image_id` can be used for locating the image in the dataset.

### Image features for COCO

If you use the provided data stream above (i.e. do not create the non-stationary data stream itself, which requires additional extra steps
such as phrase extraction), the only extra files required are,

1. Extracted image features for COCO under `datasets/coco-features`. We use code from [this repo](https://github.com/airsplay/py-bottom-up-attention.git) to extract
features. We will upload our extracted features soon;

2. A json dictionary mapping of image id to image features in the file above. Included in this repo.

### Measuring compositional generalization

We use data from [this repo](https://github.com/mitjanikolaus/compositional-image-captioning) to perform evaluation of compositional generalization. Please put `compositional-image-captioning/data/dataset_splits/`
under `datasets/novel_comps` and `compositional-image-captioning/data/occurrences/` under `datasets/occurrences`.

## Flickr shift
We use official split for Flickr30k Entities.
```
{'tokens': ['a', 'motorcyclist', '"', 'pops', 'a', 'wheelie', '"', 'in', 'a', 'grassy', 'field', 'framed', 'by', 'rolling', 'hills', '.'], 'task_id': 518, 'image_id': '3353962769', 'instance_id': 8714, 'phrase_offset': (0, 2)}
```
The `phrase offset` is the text span that should be masked for prediction. `image_id` can be used for locating the image in the dataset.

### Image features for Flickr

Similar to COCO, unless you want to create the datastream yourself, the only files required now are,

1. Extracted image features for Flickr under `datasets/flickr-features`. Simiarly, we use code from [this repo](https://github.com/airsplay/py-bottom-up-attention.git) to extract
features. This file will also be uploaded soon.

2. A json dictionary mapping of image id to image features in the file above. Included in this repo.

## Visualizing the stream
We provide `visual_stream.ipynb` to visualize task distributions over time in Flickr-shift dataset.

## Citation
```
@inproceedings{Jin2020VisuallyGC,
    title={Visually Grounded Continual Learning of Compositional Phrases},
    author={Xisen Jin and Junyi Du and Arka Sadhu and R. Nevatia and X. Ren},
    booktitle={EMNLP},
    year={2020}
}
```
