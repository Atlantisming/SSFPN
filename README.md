## SSFPN: Selective Spatial Feature Pyramid Network for Remote Sensing Object Detection([later](<>))

By Ziming Xu, Bin Ren, Kunhua Zhang

This project is based on [mmrotate1.x](https://github.com/open-mmlab/mmrotate/tree/1.x).

## Abstract

It is great important that feature fusion module generate multi-scale feature maps after backbone in object detection especially in Oriented small object detection.
Although the FPN series of models have achieved promising results in general object detection, their application in remote sensing object detection remains under-explored.
Following the prior knowledge that small oriented object need contextual information to be detected and various type of object need different long-range context.
In this paper, we take these two reason into account and propose an intuitive and simple fusion module named Selective Spatial Feature Pyramid Network (SSFPN).
SSFPN dynamically adjusts the large spatial receptive field across multi-scale feature maps, providing enhanced modeling of the contextual variations among different objects in remote sensing scenarios.

## Environment

```
sys.platform: linux
Python: 3.8.18 (default, Sep 11 2023, 13:40:15) [GCC 11.2.0]
CUDA available: True
MUSA available: False
numpy_random_seed: 2147483648
GPU 0,1,2,3,4,5,6,7: NVIDIA GeForce RTX 3090
CUDA_HOME: /usr/local/cuda
NVCC: Not Available
GCC: n/a
PyTorch: 1.13.1
PyTorch compiling details: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201402
  - Intel(R) oneAPI Math Kernel Library Version 2023.1-Product Build 20230303 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.6.0 (Git Hash 52b5f107dd9cf10910aaa19cb47f3abf9b349815)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_37,code=compute_37
  - CuDNN 8.5
  - Magma 2.6.1
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/opt/rh/devtoolset-9/root/usr/bin/c++, CXX_FLAGS= -fabi-version=11 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wunused-local-typedefs -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.13.1, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF,

TorchVision: 0.14.1
OpenCV: 4.9.0
mmcv: 2.0.1
MMEngine: 0.10.3
mmdet: 3.0.0rc6
MMRotate: 1.0.0rc1+8b30525
```

## Install

```shell
conda create --name mmrotate1 python=3.8 -y
conda activate mmrotate1
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet>3.0.0rc4,<3.1.0"
git clone https://github.com/Atlantisming/SSFPN.git
cd SSFPN
pip install -v -e .
```

## Dataset

```
SSFPN
├── data
│   ├── DOTA
│   │   ├── train
│   │   ├── val
│   │   ├── test
```

See more details for preparing datasets at [mmrotate1.x tools](https://github.com/open-mmlab/mmrotate/tree/1.x).

## Training

Single gpu for train:

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train.py projects/LSKNet/configs/lsk_s_sspafpn_1x_dota_le90.py
```

Multiple gpus for train:

```shell
bash ./tools/dist_train.sh projects/LSKNet/configs/lsk_s_sspafpn_1x_dota_le90.py 2
```

Train in pycharm: If you want to train in pycharm, you can run it in train.py.

see more details at [mmdetection](https://github.com/open-mmlab/mmdetection).

## Testing

```shell
CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/LSKNet/configs/lsk_s_sspafpn_1x_dota_le90.py <CHECKPOINT_FILE>
```

## Results on DOTAv1.0

|                           Model                            |  mAP  | Angle | lr schd | Batch Size |                         Configs                         |         Download         |
| :--------------------------------------------------------: | :---: | :---: | :-----: | :--------: | :-----------------------------------------------------: | :----------------------: |
| [RTMDet-l](https://arxiv.org/abs/2212.07784) (1024,1024,-) | 81.33 |   -   | 3x-ema  |     8      |                            -                            |            -             |
|             Oriented_RCNN_R50 (1024,1024,200)              | 81.37 | le90  |   1x    |    2\*8    |   [later](./configs/lsknet/lsk_t_fpn_1x_dota_le90.py)   | [model](<>) \| [log](<>) |
|                  LSKNet_S (1024,1024,200)                  | 81.64 | le90  |   1x    |    1\*8    |   [later](./configs/lsknet/lsk_s_fpn_1x_dota_le90.py)   | [model](<>) \| [log](<>) |
|                 LSKNet_S\\ (1024,1024,200)                 | 81.85 | le90  |   1x    |    1\*8    | [later](./configs/lsknet/lsk_s_ema_fpn_1x_dota_le90.py) | [model](<>) \| [log](<>) |

## Citations

If you find SSFPN useful in your research, please consider citing:

```
@inproceedings{zhou2022mmrotate,
  title   = {MMRotate: A Rotated Object Detection Benchmark using PyTorch},
  author  = {Zhou, Yue and Yang, Xue and Zhang, Gefan and Wang, Jiabao and Liu, Yanyi and
             Hou, Liping and Jiang, Xue and Liu, Xingzhao and Yan, Junchi and Lyu, Chengqi and
             Zhang, Wenwei and Chen, Kai},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages = {7331–7334},
  numpages = {4},
  year={2022}
}

later
```
