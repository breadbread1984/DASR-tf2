#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
from absl import app, flags;
import numpy as np;
import cv2;
import tensorflow as tf;
from create_dataset import Dataset;
from models import BlindSuperResolution;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_integer('batch_size', default = 32, help = 'batch size');
  flags.DEFINE_string('checkpoint', default = 'checkpoints', help = 'path to checkpoint directory');
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate');
  flags.DEFINE_integer('eval_steps', default = 100, help = 'how many iterations for each evaluation');
  flags.DEFINE_integer('checkpoint_steps', default = 1000, help = 'how many iterations for each checkpoint');
  flags.DEFINE_enum('scale', default = '2', enum_values = ['2', '3', '4'], help = 'train DASR on which scale of DIV2K');
  flags.DEFINE_string('dataset_path', default = None, help = 'where the HR images locate');

class SummaryCallback(tf.keras.callbacks.Callback):
  def __init__(self, dasr, eval_freq = 1000):
    self.dasr = dasr;
    self.eval_freq = eval_freq;
    testset = Dataset(FLAGS.dataset_path, scale = int(FLAGS.scale)).load_dataset(mode = 'test').batch(1).repeat(-1);
    self.iter = iter(testset);
    self.log = tf.summary.create_file_writer(FLAGS.checkpoint);
  def on_batch_end(self, batch, logs = None):
    if batch % self.eval_freq == 0:
      lr, hr = next(self.iter);
      pred_hr, loss = self.dasr([lr,lr]);
      pred_hr = tf.cast(pred_hr + tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,1,3)), dtype = tf.uint8);
      gt_hr = tf.cast(hr['sr'] + tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,1,3)), dtype = tf.uint8);
      with self.log.as_default():
        for key, value in logs.items():
          tf.summary.scalar(key, value, step = self.dasr.optimizer.iterations);
        tf.summary.image('ground truth', gt_hr, step = self.dasr.optimizer.iterations);
        tf.summary.image('predict', pred_hr, step = self.dasr.optimizer.iterations);

def main(unused_argv):
  # 1) train moco only
  # 1.1) create model and compile
  dasr = BlindSuperResolution(scale = int(FLAGS.scale), enable_train = True);
  moco = dasr.get_layer('moco');
  moco_opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 60, decay_rate = 0.9));
  moco.compile(optimizer = moco_opt,
               loss = {'output_2': tf.keras.losses.BinaryCrossentropy(from_logits = True)});
  # 1.2) create dataset
  moco_trainset = Dataset(FLAGS.dataset_path, scale = int(FLAGS.scale)).load_dataset(mode = 'moco').shuffle(10 * FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  # 1.3) fit
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = FLAGS.checkpoint),
    tf.keras.callbacks.ModelCheckpoint(filepath = join(FLAGS.checkpoint, 'moco_ckpt'), save_freq = FLAGS.checkpoint_steps),
  ];
  moco.fit(moco_trainset, callbacks = callbacks, epochs = 100);
  moco.save_weights('moco_weights.h5');
  # 2) train whole network
  # 2.1) create model and compile
  dasr_opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps = 125, decay_rate = 0.5));
  dasr.compile(optimizer = dasr_opt,
               loss = {'sr': tf.keras.losses.MeanAbsoluteError(), 'moco': tf.keras.losses.BinaryCrossentropy(from_logits = True)},
               metrics = {'sr': tf.keras.metrics.MeanAbsoluteError()});
  # 2.2) create dataset
  dasr_trainset = Dataset(FLAGS.dataset_path, scale = int(FLAGS.scale)).load_dataset(mode = 'train').shuffle(10 * FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  # 2.3) fit
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = FLAGS.checkpoint),
    tf.keras.callbacks.ModelCheckpoint(filepath = join(FLAGS.checkpoint, 'dasr_ckpt'), save_freq = FLAGS.checkpoint_steps),
    SummaryCallback(dasr, FLAGS.eval_steps),
  ];
  dasr.fit(dasr_trainset, callbacks = callbacks, epochs = 500);
  dasr.save_weights('dasr_weights.h5');

if __name__ == "__main__":
  add_options();
  app.run(main);
