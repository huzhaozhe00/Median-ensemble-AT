# Median-ensemble Adversarial Training (MEAT)

This repository contains the code of ["Median-Ensemble Adversarial Training for Improving Robustness and Generalization"](https://ieeexplore.ieee.org/abstract/document/10446117) published at ICASSP 2024. 

## Requirements
The development environment is:
* Python (3.10)
* Pytorch (1.13)
* CUDA
* Torchvision (0.14)
* Torchattacks (3.3)

## Training

To train a ResNet-18 on CIFAR-10 using PGD-10:
```
python meat.py
```

To train a WideResNet-34-10 on CIFAR-10 using PGD-10:
```
python meat.py --arch 'WRN'
```
