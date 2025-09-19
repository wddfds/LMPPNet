# LMPPNet

This is the code for the paper xxxx

## Environment 
Python-3.9， CUDA-11.8， Pytorch-2.0.1

### Install
1. Create a new conda environment

```
conda create -n lmppnet python=3.9
conda activate parunet
```

2. Install dependencies
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Dataset
Prepare the dataset path as follows:
```
┬─ ITS
│   ├─ train
│   │   ├─ hazy
│   │   │   ├─ 00001.png
│   │   │   ├─ 00002.png
│   │   │   └─ ...
│   │   ├─ clear
│   │   │   ├─ 00001.png
│   │   │   ├─ 00002.png
│   │   │   └─ ...
│   └─ test
│       ├─ hazy
│       │   └─ ...
│       └─ clear
│           └─ ...
└─ OTS
    ├─ train
    │   └─ ...
    └─ test
        └─ ...
```

## Test
Run the following script to test the trained model:
```sh
python test.py  --dataset [dataset_path] --model_dir [weight path] --save_image[whether save the predict images]
```
Example
```sh
python test.py  --dataset ./dataset/ITS --model_dir ./pretrained/ITS.pth --save_image True
```

## Train
You can modify the training hyperparameters for each experiment and run the following script to train a new model:
```sh
python train.py  --project [project name] --dataset [dataset_path] 
```
Example
```sh
python train.py  --project new_project --dataset ./dataset/ITS
```

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={},
  volume={},
  pages={}
}
```
