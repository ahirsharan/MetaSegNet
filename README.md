
## Differentiable Meta-learning Model for Few-shot Semantic Segmentation (MetaSegNet)

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/pytorch-0.4.0-%237732a8)](https://github.com/ahirsharan/MetaSegNet.git)

<!-- TOC -->

- [Meta Transfer Learning for Few Shot Semantic Segmentation using U-Net](#Meta-Transfer-Learning-for-Few-Shot-Semantic-Segmentation-using-U-Net)
  - [Requirements](#requirements)
  - [Code structure](#code-structure)
  - [Running Experiments](#Running-Experiments)
  - [Hyperparameters and Options](#Hyperparameters-and-Options)

<!-- /TOC -->

## Requirements
PyTorch and Torchvision needs to be installed before running the scripts, together with `PIL` for data-preprocessing and `tqdm` for showing the training progress.

To run this repository, kindly install python 3.5 and PyTorch 0.4.0 with Anaconda.

You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install PyTorch and torchvision on it:

```bash
conda create --name segfew python=3.5
conda activate segfew
conda install pytorch=0.4.0 
conda install torchvision -c pytorch
```

Clone this repository:

```bash
git clone https://github.com/ahirsharan/MetaSegNet.git
```

## Code Structure
The code structure is based on [MTL-template](https://github.com/yaoyao-liu/meta-transfer-learning) and [Pytorch-Segmentation](https://github.com/yassouali/pytorch_segmentation). 

```
.
├── Datasets
    |
    ├── COCOAug     
    ├── Pascal5Aug
    ├── FSS1000   
    |  
├── MetaSegNet
    |
    ├── FewShotPreprocessing.py     # utility to organise the Few-shot data into train and novel
    ├── cocogen.py                  # utility to organise the Few-shot data into train and novel after generating masks
    ├── augment.py                  # For generic data Augmentation 
    |
    |  
    ├── dataloader              
    |   ├── dataset_loader.py       # data loader for pre datasets
    |   └── samplers.py             # samplers for meta task dataset(Few-Shot) 
    |
    |
    ├── models                      
    |   ├── mtl.py                  # meta-transfer class
    |   └── metasegnet.py           # Resnet-9 class
    |
    ├── trainer                     
    |   ├── meta.py                  # meta-train trainer class
    |   
    |
    ├── utils                       
    |   ├── gpu_tools.py            # GPU tool functions
    |   ├── metrics.py              # Metrics functions
    |   ├── losses.py               # Loss functions
    |   ├── lovasz_losses.py        # Lovasz Loss function
    |   └── misc.py                 # miscellaneous tool functions
    |
    ├── main.py                     # the python file with main function and parameter settings
    └── run_meta.py                 # the script to run meta-train and meta-test phases
```
## Running Experiments

Run meta-train and meta-test phase:
```bash
python run_meta.py
```
The test predictions and logs(models) will be stored in the same root directory under resultsx and logsx where x can be changed in trainer/meta.py

## Hyperparameters and Options
Hyperparameters and options in `main.py`.

- `model_type` The network architecture
- `dataset` Meta dataset (change!!)
- `phase` train or test
- `seed` Manual seed for PyTorch, "0" means using random seed
- `gpu` GPU id (change!!)
- `dataset_dir` Directory for the images (change!!)
- `max_epoch` Epoch number for meta-train phase (change!!)
- `num_batch` The number for different tasks used for meta-train (change!!)
- `way` Way number, how many classes in a task(Background excluded) (change!!)
- `train_query` Shots: The number of training samples for each class in a task (change!!)
- `test_query` The number of test samples for each class in a task
- `meta_lr` Learning rate for embedding model
- `base_lr` Learning rate for the inner loop
- `update_step` The number of updates for the inner loop
- `step_size` The number of epochs to reduce the meta learning rates
- `gamma` Gamma for the meta-train learning rate decay
- `init_weights` The pretained weights for meta-train phase
- `meta_label` Additional label for meta-train
