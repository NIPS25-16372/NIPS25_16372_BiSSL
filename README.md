# BiSSL: Enhancing the Alignment Between Self-Supervised Pre-Training and Downstream Fine-Tuning via Bilevel Optimization
This repository contains the pytorch-implementation accompanying the paper on BiSSL (submitted to NeurIPS 2025).  The codebase supports both classification and object detection tasks along the SimCLR and BYOL pretext tasks, and is verified for single-node multi-GPU setups using Python 3.10.12, PyTorch 2.1.2, Torchvision 0.16.2, Timm 1.0.15 and Ray 2.9.2.
![](figs/bissl_pipeline.png)

---

## Table of Contents
- [Pretrained Models](#pretrained-models)
- [Training Pipeline Overview](#training-pipeline-overview)
  - [Self-Supervised Pretraining](#1-self-supervised-pretraining)
  - [BiSSL Training](#2-bissl-training)
  - [Fine-Tuning](#3-fine-tuning)

---

## Pretrained Models
Pretrained model weights and configuration files used in the paper can be downloaded [here](https://drive.google.com/drive/folders/120GUKlcpeh3rhKq9W_-6lSHCKWhQx7gB?usp=share_link). By default, models and corresponding config files are stored under the `models/` directory. Use `--model-root` to specify an alternative location.

## Training Pipeline Overview
General config: `runs/config_general.py`

### 1. Self-Supervised Pretraining
Run scripts: 
- `runs/pretext/simclr/run.py`
- `runs/pretext/byol/run.py`

To tun SimCLR with 4 GPUs:
```
torchrun --nproc-per-node 4 runs/pretext/simclr/run.py --root 'PATH_TO_ROOT'
```

To similarly run BYOL:
```
torchrun --nproc-per-node 4 runs/pretext/byol/run.py --root 'PATH_TO_ROOT'
```
#### Config files
(*To see configurable parameters for a run, simply run the script with the sole argument `-h`. E.g. `python runs/pretext/byol/run.py -h`*)

Default config files:
- Pretext General: `runs/pretext/config.py`
- SimCLR Specific: `runs/pretext/simclr/config.py`
- BYOL Specific: `runs/pretext/byol/config.py`

---

### 2. BiSSL Training
Run scripts: 
- `runs/bissl/classification/*`
- `runs/bissl/object_detection/*`

The current codebase supports BiSSL with both SimCLR and BYOL as pretext tasks and can be used for classification or detection.

#### Classification (Example: SimCLR and Oxford-IIIT Pets)
```
torchrun --nproc-per-node 4 runs/bissl/classification/simclr/run.py \
  --root PATH_TO_ROOT \
  --pretrained_model_backbone 'Pretext_simclr_arch-resnet50_backbone_id-hb63rtyl.pth'\
  --pretrained_model_head 'Pretext_simclr_arch-resnet50_head_id-hb63rtyl.pth' \
  --pretrained_model_config 'Pretext_simclr_arch-resnet50_config_id-hb63rtyl.json' \
  --d-dataset 'pets' \
  --d-lr 0.03 \
  --d-wd 0.001
```

#### Object Detection (Example: SimCLR and VOC07+12)

```
torchrun --nproc-per-node 4 runs/bissl/object_detection/simclr/run.py \
  --root PATH_TO_ROOT \
  --pretrained_model_backbone 'Pretext_simclr_arch-resnet50_backbone_id-hb63rtyl.pth' \
  --pretrained_model_head 'Pretext_simclr_arch-resnet50_head_id-hb63rtyl.pth' \
  --pretrained_model_config 'Pretext_simclr_arch-resnet50_config_id-hb63rtyl.json' \
  --d-dataset 'voc07+12detection'
```

#### Config Files
(*To see configurable parameters for a run, simply run the script with the sole argument `-h`. E.g. `python runs/bissl/classification/simclr/run.py -h`*)

Default config files:
- General: `runs/bissl/config.py`
- *Classification-specific:*
  - General: `runs/bissl/classification/config.py`
  - SimCLR: `runs/bissl/classification/simclr/config.py`
  - BYOL: `runs/bissl/classification/byol/config.py`
- *Object Detection-specific*
  - General: `runs/bissl/object_detection/config.py`
  - SimCLR: `runs/bissl/object_detection/simclr/config.py`
  - BYOL: `runs/bissl/object_detection/byol/config.py`

---

### 3. Fine-Tuning
Run scripts: 
- `runs/fine_tune/classification/*`
- `runs/fine_tune/object_detection/*`

The current codebase supports fine-tuning for classification and object detection tasks respectively.


#### Classification (Example: Oxford-IIIT Pets)
##### Post Pretext (HPO)
Hyper-parameter optimization (HPO) with a random grid search over 100 combinations of learning rates and weight decays (as specified in the paper) used for fine-tuning a self-supervised pre-trained backbone via SimCLR on the pets dataset:
```
torchrun --nproc-per-node 4 runs/fine_tune/classification/resnet/post_pretext_ft/run.py \
  --root PATH_TO_ROOT \
  --pretrained_model_backbone 'Pretext_simclr_arch-resnet50_backbone_id-hb63rtyl.pth' \
  --pretrained_model_config 'Pretext_simclr_arch-resnet50_config_id-hb63rtyl.json' \
  --dset 'pets' \
  --num-runs 100 \
  --use-hpo 1
```

##### Post BiSSL (HPO)
To conduct a similar run, but with a backbone obtained using BiSSL instead, run:
Post BiSSL
```
torchrun --nproc-per-node 4 runs/fine_tune/classification/resnet/post_bissl_ft/run.py \
  --root PATH_TO_ROOT \
  --pretrained_model_backbone 'BiSSL_simclr_classification_pets_arch-resnet50_lower_backbone_id-izqrdr53.pth' \
  --pretrained_model_config 'BiSSL_simclr_classification_pets_arch-resnet50_config_id-izqrdr53.json' \
  --num-runs 100 \
  --use-hpo 1
```

##### Post BiSSL (Fixed HPs)
To conduct 10 fine-tunings with different random seeds (post BiSSL) on the pets dataset with a fixed learning rate of 0.03 and weight decay of 0.001, run:
```
torchrun --nproc-per-node 4 runs/fine_tune/classification/resnet/post_bissl_ft/run.py \
  --root PATH_TO_ROOT \
  --pretrained_model_backbone 'BiSSL_simclr_classification_pets_arch-resnet50_lower_backbone_id-izqrdr53.pth' \
  --pretrained_model_config 'BiSSL_simclr_classification_pets_arch-resnet50_config_id-izqrdr53.json' \
  --num-runs 10 \
  --use-hpo 0 \
  --lr 0.03 \
  --wd 0.001
```

##### Object Detection (Example: VOC07+12)
The code runs similarly to fine-tuning for classificaiton:
##### Post Pretext (HPO)
```
torchrun --nproc-per-node 4 runs/fine_tune/object_detection/resnet/post_pretext_ft/run.py \
  --root PATH_TO_ROOT \
  --pretrained_model_backbone 'Pretext_simclr_arch-resnet50_backbone_id-hb63rtyl.pth' \
  --pretrained_model_config 'Pretext_simclr_arch-resnet50_config_id-hb63rtyl.json' \
  --dset 'voc07+12detection' \
  --num-runs 100 \
  --use-hpo 1
```

#### Config Files
(*To see configurable parameters for a run, simply run the script with the sole argument `-h`. E.g. `python runs/fine_tune/classification/resnet/post_bissl_ft/run.py -h`*)

Default config files:
- General: `runs/fine_tune/config.py`
- *Classification-specific:*
  - General: `runs/fine_tune/classification/config.py`
  - Post Pretext: `runs/fine_tune/classification/resnet/post_pretext_ft/config.py`
  - Post BiSSL: `runs/fine_tune/classification/resnet/post_bissl_ft/config.py`
- *Object Detection-specific:*
  - General: `runs/fine_tune/object_detection/config.py`
  - Post Pretext: `runs/fine_tune/object_detection/resnet/post_pretext_ft/config.py`
  - Post BiSSL: `runs/fine_tune/object_detection/resnet/post_bissl_ft/config.py`




