# BiSSL: Enhancing the Alignment Between Self-Supervised Pre-Training and Downstream Fine-Tuning via Bilevel Optimization
This repository contains the pytorch-implementation accompanying the paper on BiSSL. The codebase supports downstream classification tasks and the SimCLR and BYOL pretext tasks, and is verified for single-node multi-GPU setups using Python 3.10.12, PyTorch 2.1.2, Torchvision 0.16.2, Timm 1.0.15 and Ray 2.9.2.
![](figs/bissl_pipeline.png)

---

## Table of Contents
- [Datasets and Pretrained Models](#datasets-and-pretrained-models)
- [Training Pipeline Overview](#training-pipeline-overview)
  - [Self-Supervised Pretraining](#1-self-supervised-pretraining)
  - [BiSSL Training](#2-bissl-training)
  - [Fine-Tuning](#3-fine-tuning)

---

## Datasets and Pretrained Models
By default, datasets are stored under the `data/` directory. Use `--dataset-root` to specify an alternative location. The dataset splits made in conjunction with this codebase are located in `data/tv_splits/`.

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
- `runs/bissl/classification/simclr/run.py`
- `runs/bissl/classification/byol/run.py`

The current codebase supports BiSSL with both SimCLR and BYOL as pretext tasks and can be used for downstream classification tasks.

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

#### Config Files
(*To see configurable parameters for a run, simply run the script with the sole argument `-h`. E.g. `python runs/bissl/classification/simclr/run.py -h`*)

Default config files:
- General: `runs/bissl/config.py`
- *Classification-specific:*
  - General: `runs/bissl/classification/config.py`
  - SimCLR: `runs/bissl/classification/simclr/config.py`
  - BYOL: `runs/bissl/classification/byol/config.py`

---

### 3. Fine-Tuning
Run scripts: 
- `runs/fine_tune/classification/resnet/post_pretext_ft/run.py`
- `runs/fine_tune/classification/resnet/post_bissl_ft/run.py`


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
  --pretrained_model_backbone 'BiSSL_simclr_classification_pets_arch-resnet50_lower_backbone_id-2gn6d5az.pth' \
  --pretrained_model_config 'BiSSL_simclr_classification_pets_arch-resnet50_config_id-2gn6d5az.json' \
  --num-runs 100 \
  --use-hpo 1
```

##### Post BiSSL (Fixed HPs)
To conduct 10 fine-tunings with different random seeds (post BiSSL) on the pets dataset with a fixed learning rate of 0.03 and weight decay of 0.001, run:
```
torchrun --nproc-per-node 4 runs/fine_tune/classification/resnet/post_bissl_ft/run.py \
  --root PATH_TO_ROOT \
  --pretrained_model_backbone 'BiSSL_simclr_classification_pets_arch-resnet50_lower_backbone_id-2gn6d5az.pth' \
  --pretrained_model_config 'BiSSL_simclr_classification_pets_arch-resnet50_config_id-2gn6d5az.json' \
  --num-runs 10 \
  --use-hpo 0 \
  --lr 0.03 \
  --wd 0.001
```

#### Config Files
(*To see configurable parameters for a run, simply run the script with the sole argument `-h`. E.g. `python runs/fine_tune/classification/resnet/post_bissl_ft/run.py -h`*)

Default config files:
- General: `runs/fine_tune/config.py`
- *Classification-specific:*
  - General: `runs/fine_tune/classification/config.py`
  - Post Pretext: `runs/fine_tune/classification/resnet/post_pretext_ft/config.py`
  - Post BiSSL: `runs/fine_tune/classification/resnet/post_bissl_ft/config.py`




