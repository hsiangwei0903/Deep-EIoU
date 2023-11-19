# Deep-EIoU

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/iterative-scale-up-expansioniou-and-deep/multi-object-tracking-on-sportsmot)](https://paperswithcode.com/sota/multi-object-tracking-on-sportsmot?p=iterative-scale-up-expansioniou-and-deep)

This is the official code for paper "Iterative Scale-Up ExpansionIoU and Deep Features Association for Multi-Object Tracking in Sports (2024 WACV RWS Workshop)". [Arxiv](https://arxiv.org/abs/2306.13074)

## Setup Instructions

* Clone this repo, and we'll call the directory that you cloned as {Deep-EIoU Root}
* Install dependencies.
```
conda create -n DeepEIoU python=3.7
conda activate DeepEIoU

# Install pytorch with the proper cuda version to suit your machine
# We are using torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 with cuda==11.6

cd Deep-EIoU/reid
pip install -r requirements.txt
pip install cython_bbox
python setup.py develop
cd ..
```

## Reproduce on SportsMOT dataset

### 1. Data preparation for reproduce on SportsMOT dataset

To reproduce on the SportsMOT dataset, you need to download the detection and embedding files from [drive](https://drive.google.com/drive/folders/14gh9e5nQhqHsw77EfxZaUyn9NgPP0-Tq?usp=sharing)

Please download these files and put them in the corresponding folder.

```
{Deep-EIoU Root}
   |——————Deep-EIoU
   └——————detection
   |        └——————v_-9kabh1K8UA_c008.npy
   |        └——————v_-9kabh1K8UA_c009.npy
   |        └——————...
   └——————embedding
            └——————v_-9kabh1K8UA_c008.npy
            └——————v_-9kabh1K8UA_c009.npy
            └——————...
```

### 2. Run tracking on SportsMOT dataset
Run the following commands, you should see the tracking result for each sequences in the interpolation folder.
Please directly zip the tracking results and submit to the [SportsMOT evaluation server](https://codalab.lisn.upsaclay.fr/competitions/12424#participate).

```
python tools/sport_track.py --root_path <Deep-EIoU Root>
python tools/sport_interpolation.py --root_path <Deep-EIoU Root>
```

## Demo on custom dataset

### 1. Model preparation for demo on custom dataset
To demo on your custom dataset, download the detector and ReID model from [drive](https://drive.google.com/drive/folders/1wItcb0yeGaxOS08_G9yRWBTnpVf0vZ2w) and put them in the corresponding folder.

```
{Deep-EIoU Root}
   └——————Deep-EIoU
            └——————checkpoints
                └——————best_ckpt.pth.tar (YOLOX Detector)
                └——————sports_model.pth.tar-60 (OSNet ReID Model)
```

### 2. Demo on custom dataset
Demo on our provided video
```
python tools/demo.py
```
Demo on your custom video
```
python tools/demo.py --path <your video path>
```

## Citation
If you find our work useful, please kindly cite our paper, thank you.
```
@article{huang2023iterative,
  title={Iterative Scale-Up ExpansionIoU and Deep Features Association for Multi-Object Tracking in Sports},
  author={Huang, Hsiang-Wei and Yang, Cheng-Yen and Hwang, Jenq-Neng and Huang, Chung-I},
  journal={arXiv preprint arXiv:2306.13074},
  year={2023}
}
```

## Acknowledgements
The code is based on [ByteTrack](https://github.com/ifzhang/ByteTrack), [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) and [BoT-SORT](https://github.com/NirAharon/BoT-SORT), thanks for their wonderful work!

## Contact
Hsiang-Wei Huang (hwhuang@uw.edu)
