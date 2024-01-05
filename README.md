# CSSNet
Official code for [CSSNet: Cascaded Spatial Shift Network for Multi-organ Segmentation](https://www.sciencedirect.com/science/article/pii/S0010482524000398). Our paper has been accepted by Computers in Biology and Medicine!


Our project is based on [TransUNet](https://github.com/Beckschen/TransUNet) and [AS-MLP](https://github.com/svip-lab/AS-MLP). Thanks for their great work!

## Installation

1. First install the pytorch.
```Shell
conda create -n mlp python=3.8 -y
conda activate mlp
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

2. Following the [TransUNet](https://github.com/Beckschen/TransUNet) to install the dependency.
3. If you need the preprocessed data, please send an email to kunyangzhou@seu.edu.cn.

## Training and Testing

1. Following the [TransUNet](https://github.com/Beckschen/TransUNet) to prepare the data.

2. Run the following code to train the model.
```Shell
python train.py --dataset Synapse
```

3. You can use the following code to test the model.
```Shell
python test.py --dataset Synapse
```

## Core code

1. You can find the Cascaded-MLP code in [as_mlp.py](networks/as_mlp.py).

2. The network code is in [Seg.py](networks/Seg.py).

## Citation
If you think our work is helpful, please cite our paper
```latex
@article{2024cssnet,
author = {Yeqin Shao, Kunyang Zhou, and Lichi Zhang},
title = {CSSNet: Cascaded spatial shift network for multi-organ segmentation},
journal = {Computers in Biology and Medicine},
year = {2024}
}
```
