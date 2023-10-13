# CSSNet
Official code for CSSNet: Cascaded Spatial Shift Network for Multi-organ Segmentation.

Our code is based on [TransUNet](https://github.com/Beckschen/TransUNet) and [AS-MLP](https://github.com/svip-lab/AS-MLP). Thanks for their great work!

## Installation

1. First install the pytorch.
```Shell
conda create -n mlp python=3.8 -y
conda activate mlp
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

2. Following the [TransUNet](https://github.com/Beckschen/TransUNet) to install the dependency.

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