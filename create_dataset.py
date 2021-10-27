#!/usr/bin/python3

from os import listdir;
from os.path import join, splitext;
import numpy as np;
import cv2;
import tensorflow as tf;

class Dataset(object):
  def __init__(self, dataset_dir, scale):
    dir_hr = dataset_dir
    self.hr_list = list();
    self.scale = scale;
    for f in listdir(dir_hr):
      if splitext(f)[1] == '.png':
        self.hr_list.append(join(dir_hr, f));
  def get_generator(self, is_train = True, lr_patch_size = 48):
    def gen():
      for f in self.hr_list:
        # load hr image
        img = cv2.imread(f)[...,::-1];
        # augment
        if is_train == True:
          if np.random.uniform() < 0.5:
            img = cv2.flip(img, 1); # hflip
          if np.random.uniform() < 0.5:
            img = cv2.flip(img, 0); # vflip
          if np.random.uniform() < 0.5:
            img = np.transpose(img, (1,0,2)); # w x h x c
        h, w = img.shape[:2];
        if is_train == True:
          # get two patches from this image
          hr_patch_size = round(self.scale * lr_patch_size);
          inputs = list();
          outputs = None;
          for i in range(2):
            x = np.random.randint(low = 0, high = w - hr_patch_size, size = ());
            y = np.random.randint(low = 0, high = h - hr_patch_size, size = ());
            patch = img[y:y+hr_patch_size,x:x+hr_patch_size,:];
            if i == 0: outputs = patch;
            inputs.append(cv2.resize(patch, (lr_patch_size, lr_patch_size), interpolation = cv2.INTER_CUBIC));
          inputs = tuple(inputs);
        else:
          # return the whole image
          h = h - (h % self.scale);
          w = w - (w % self.scale);
          img = img[:h, :w, :];
          outputs = img;
          inputs = cv2.resize(img, (w//self.scale, h//self.scale), interpolation = cv2.INTER_CUBIC);
        yield inputs, outputs;
    return gen;
  def get_parse_function(self, is_train = True):
    def train_parse_function(inputs, outputs):
      lr_patch1, lr_patch2 = inputs;
      hr_patch1 = outputs;
      mean = tf.reshape(tf.constant([114.444 , 111.4605, 103.02  ], dtype = tf.float32), (1,1,3));
      lr_patch1 = tf.cast(lr_patch1, dtype = tf.float32) - mean;
      lr_patch2 = tf.cast(lr_patch2, dtype = tf.float32) - mean;
      hr_patch1 = tf.cast(hr_patch1, dtype = tf.float32) - mean;
      return (lr_patch1, lr_patch2), {'sr': hr_patch1, 'moco': 0};
    def test_parse_function(inputs, outputs):
      lr = inputs;
      hr = outputs;
      lr = tf.image.resize(lr, (192,192), method = 'bicubic');
      hr = tf.image.resize(hr, (192 * self.scale, 192 * self.scale), method = 'bicubic');
      mean = tf.reshape(tf.constant([114.444 , 111.4605, 103.02  ], dtype = tf.float32), (1,1,3));
      lr = tf.cast(lr, dtype = tf.float32) - mean;
      hr = tf.cast(hr, dtype = tf.float32) - mean;
      return lr, {'sr': hr, 'moco': 0};
    return train_parse_function if is_train == True else test_parse_function;
  def load_dataset(self, is_train = True, lr_patch_size = 48):
    return tf.data.Dataset.from_generator(self.get_generator(is_train, lr_patch_size),
                                          ((tf.float32, tf.float32), tf.float32) if is_train == True else (tf.float32, tf.float32),
                                          ((tf.TensorShape([lr_patch_size,lr_patch_size,3]), tf.TensorShape([lr_patch_size,lr_patch_size,3])), tf.TensorShape([lr_patch_size * self.scale, lr_patch_size * self.scale,3])) if is_train == True else (tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3]))
                                         ).map(self.get_parse_function(is_train));

if __name__ == "__main__":
  trainset = iter(Dataset('dataset', scale = 2).load_dataset(is_train = True));
  testset = iter(Dataset('dataset', scale = 2).load_dataset(is_train = False));
  for i in range(5):
    samples, labels = next(trainset);
    lr1, lr2 = samples;
    lr1 = (lr1.numpy() + np.reshape([114.444 , 111.4605, 103.02  ], (1,1,3))).astype(np.uint8);
    lr2 = (lr2.numpy() + np.reshape([114.444 , 111.4605, 103.02  ], (1,1,3))).astype(np.uint8);
    hr1 = (labels['sr'].numpy() + np.reshape([114.444 , 111.4605, 103.02  ], (1,1,3))).astype(np.uint8);
    cv2.imshow('lr1', lr1);
    cv2.imshow('lr2', lr2);
    cv2.imshow('hr1', hr1);
    cv2.waitKey();
  for i in range(5):
    samples, labels = next(testset);
    lr = samples;
    lr = (lr.numpy() + np.reshape([114.444 , 111.4605, 103.02  ], (1,1,3))).astype(np.uint8);
    hr = (labels['sr'].numpy() + np.reshape([114.444 , 111.4605, 103.02  ], (1,1,3))).astype(np.uint8);
    cv2.imshow('lr', lr);
    cv2.imshow('hr', hr);
    cv2.waitKey();
