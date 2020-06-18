# Code for [LSD-C: Linearly Separable Deep Clusters](https://arxiv.org/abs/2006.10039)

by [Sylvestre-Alvise Rebuffi*](http://www.robots.ox.ac.uk/~srebuffi/), [Sebastien Ehrhardt*](), [Kai Han*](http://www.hankai.org), [Andrea Vedaldi](http://www.robots.ox.ac.uk/~vedaldi/), [Andrew Zisserman](http://www.robots.ox.ac.uk/~az/)


## Dependencies

All dependencies are included in `environment.yml`. To install the environment, please run:

```shell
conda env create -f environment.yml
```

Then, you can activate the installed environment by running:

```
conda activate lsd_c
```

## Downloading the pretrained RotNet on CIFAR 10

The pretrained initialization network can be downloaded with the following link:

https://www.dropbox.com/s/4c6jw6caz3tsoe0/RotNet_cifar10.pt?dl=0

## Running our clustering method on CIFAR 10

For the kNN labeling method, please run:

```shell
python cifar10_clustering.py 
```
For the cosine labeling method, please run:

```shell
python cifar10_clustering.py --similarity_type cosine --hyperparam 0.9
```
For the SNE labeling method, please run:

```shell
python cifar10_clustering.py --similarity_type SNE --hyperparam 0.01
```

## Citation
If this work is helpful for your research, please cite our paper.
```
@article{rebuffi2020lsdc,
author    = {Sylvestre-Alvise Rebuffi and Sebastien Ehrhardt and Kai Han and Andrea Vedaldi and Andrew Zisserman},
title     = {LSD-C: Linearly Separable Deep Clusters},
journal = {arXiv},
year      = {2020}
}
```

## Acknowledgments
This work is supported by the [EPSRC Programme Grant Seebibyte EP/M013774/1](http://seebibyte.org/), [Mathworks/DTA DFR02620](), and [ERC IDIU-638009](https://cordis.europa.eu/project/rcn/196773/factsheet/en).
