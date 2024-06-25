# Median-ensemble Adversarial Training (MEAT)

This repository contains the code of ["Self-Ensemble Adversarial Training for Improved Robustness"](https://ieeexplore.ieee.org/abstract/document/10446117) published at ICASSP 2024. 

## Prerequisites
* Python (3.10)
* Pytorch (1.13)
* CUDA
* Torchvision (0.14)
* Torchattacks (3.3)

## How to train

Train WideResNet-34-10 on CIFAR-10:
```
$ python3 meat.py --arch 'WRN'
```
