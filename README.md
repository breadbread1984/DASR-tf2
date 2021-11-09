# Degradation Aware Super Resolution

## Introduction

this project implements degradation aware super resolution (DASR) algorithm.

## prepare dataset

download DIV2K and Flickr2K and merge their HR images into one directory.

## how to train

train with command

```shell
python3 train.py --scale=(2|3|4) --dataset_path=<path/to/HR directory> --batch_size=<batch size>
```

a pretrained checkpoint can be download [here](https://pan.baidu.com/s/1eD69OpuppNDroZcjj2lKgA), passcode is **r7vq**

## how to save model

save the trained model with the command

```shell
python3 save_model.py --scale=(2|3|4)
```

## how to test saved model

test the model with the command.
one among image, video and dataset_path must be provided.

```shell
python3 test.py --scale=(2|3|4) [--image <test image>] [--video <test video>] [--dataset_path=<path/to/HR directory>]
```
