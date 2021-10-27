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
  flags.DEFINE_integer('batch_size', default = 16, help = 'batch size');
  flags.DEFINE_string('checkpoint', default = 'checkpoints', help = 'path to checkpoint directory');
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate');
  flags.DEFINE_integer('epochs', default = 560, help = 'epochs');
  flags.DEFINE_bool('save_model', default = False, help = 'whether to save model');
  flags.DEFINE_integer('eval_steps', default = 100, help = 'how many iterations for each evaluation');
  flags.DEFINE_integer('checkpoint_steps', default = 100, help = 'how many iterations for each checkpoint');
  flags.DEFINE_enum('scale', default = '2', enum_values = ['2', '3', '4'], help = 'train DASR on which scale of DIV2K');
  flags.DEFINE_string('dataset_path', default = None, help = 'where the HR images locate');

class SummaryCallback(tf.keras.callbacks.Callback):
  def __init__(self, dasr, eval_freq = 1000):
    self.dasr = dasr;
    self.eval_freq = eval_freq;
    testset = Dataset(FLAGS.dataset_path, scale = int(FLAGS.scale)).load_dataset(is_train = False).batch(1).repeat(-1);
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
  # 1) create model
  if exists(join(FLAGS.checkpoint, 'ckpt')):
    dasr = tf.keras.models.load_model(join(FLAGS.checkpoint, 'ckpt'), custom_objects = {'tf': tf}, compile = True);
    optimizer = dasr.optimizer;
  else:
    dasr = BlindSuperResolution(scale = int(FLAGS.scale), enable_train = True);
    optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(FLAGS.lr, decay_steps = 100));
    dasr.compile(optimizer = optimizer,
                 loss = {'sr': tf.keras.losses.MeanAbsoluteError(), 'moco': tf.keras.losses.BinaryCrossentropy(from_logits = True)},
                 metrics = {'sr': tf.keras.metrics.MeanAbsoluteError()});
  if FLAGS.save_model:
    if not exists('models'): mkdir('models');
    dasr.save_weights(join('models', 'dasr_weights.h5'));
    exit();
  # 2) create dataset
  trainset = Dataset(FLAGS.dataset_path, scale = int(FLAGS.scale)).load_dataset(is_train = True).shuffle(10 * FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = Dataset(FLAGS.dataset_path, scale = int(FLAGS.scale)).load_dataset(is_train = False).shuffle(10 * FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  # 3) optimizer
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = FLAGS.checkpoint),
    tf.keras.callbacks.ModelCheckpoint(filepath = join(FLAGS.checkpoint, 'ckpt'), save_freq = FLAGS.checkpoint_steps),
    SummaryCallback(dasr, FLAGS.eval_steps),
  ];
  dasr.fit(trainset, epochs = FLAGS.epochs, validation_data = testset, callbacks = callbacks);
  dasr.save_weights('dasr_weights.h5');

if __name__ == "__main__":
  add_options();
  app.run(main);
