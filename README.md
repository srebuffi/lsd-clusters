# Code for "LSD-C: Linearly Separable Deep Clusters"

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

The network can be downloaded with the following link:

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
