# Degradation Aware Super Resolution

## Introduction

this project implements degradation aware super resolution (DASR) algorithm.

## prepare dataset

download DIV2K and Flickr2K and merge their HR images into one directory.

## how to train

train with command

```shell
python3 train.py --dataset_path=<path/to/HR directory> --batch_size=<batch size>
```

## how to save model

save the trained model with the command

```shell
python3 train.py --save_model
```

## how to test saved model

test the model with the command

```shell
python3 test.py
```
