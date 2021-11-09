#!/usr/bin/python3

from os.path import join;
from absl import app, flags;
import numpy as np;
import cv2;
import tensorflow as tf;
from create_dataset import Dataset;
from models import BlindSuperResolution;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_string('dataset_path', default = None, help = 'where the HR images locate');
  flags.DEFINE_string('video', default = None, help = 'test video');
  flags.DEFINE_string('image', default = None, help = 'test image');
  flags.DEFINE_enum('scale', default = '2', enum_values = ['2', '3', '4'], help = 'train DASR on which scale of DIV2K');

def main(unused_argv):
  dasr = tf.keras.models.load_model(join('models', 'dasr_x%s.h5' % FLAGS.scale));
  if FLAGS.image is not None:
    img = cv2.imread(FLAGS.image)[...,::-1]; # convert to RGB
    if img is None:
      print('invalid image!');
      exit();
    lr = np.expand_dims(tf.cast(img), axis = 0) - np.reshape([114.444 , 111.4605, 103.02  ], (1,1,1,3));
    sr = dasr(lr);
    sr = np.squeeze(sr.numpy() + np.reshape([114.444 , 111.4605, 103.02  ], (1,1,1,3)), axis = 0).astype(np.uint8)[...,::-1];
    cv2.imshow('sr', sr);
    cv2.waitKey();
  elif FLAGS.video is not None:
    video = cv2.VideoCapture(FLAGS.video);
    if video.isOpened() == False:
      print('invalid video!');
      exit();
    fr = video.get(cv2.CAP_PROP_FPS);
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT));
    writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fr, (width * int(FLAGS.scale), height * int(FLAGS.scale)));
    retval = True;
    while retval:
      retval, img = video.read();
      if retval == False: break;
      img = img[...,::-1]; # convert to rgb
      lr = np.expand_dims(tf.cast(img, dtype = tf.float32), axis = 0) - np.reshape([114.444 , 111.4605, 103.02  ], (1,1,1,3));
      sr, loss = dasr([lr, lr]);
      sr = np.squeeze(sr.numpy() + np.reshape([114.444 , 111.4605, 103.02  ], (1,1,1,3)), axis = 0).astype(np.uint8)[...,::-1]; # convert to bgr
      writer.write(sr);
    video.release();
    writer.release();
  elif FLAGS.dataset_path is not None:
    testset = Dataset(FLAGS.dataset_path, scale = int(FLAGS.scale)).load_dataset(mode = 'test').batch(1);
    for lr, hr in testset:
      sr, loss = dasr([lr, lr]);
      sr = tf.cast(sr + tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,1,3)), dtype = tf.uint8);
      sr = tf.squeeze(sr, axis = 0).numpy()[...,::-1]; # convert to bgr
      cv2.imshow('sr', sr);
      cv2.waitKey();
  else:
    raise Exception('one among image, video and dataset_path must be given!');
    exit();

if __name__ == "__main__":
  add_options();
  app.run(main);
