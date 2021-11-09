#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
from absl import app, flags;
import tensorflow as tf;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_string('checkpoint', default = 'checkpoints', help = 'checkpoint directory');
  flags.DEFINE_enum('scale', default = '2', enum_values = {'2', '3', '4'}, help = 'scale of resolution');

def main(unused_argv):
  dasr = tf.keras.models.load_model(join(FLAGS.checkpoint, 'dasr_ckpt'), custom_objects = {'tf': tf}, compile = True);
  if not exists('models'): mkdir('models');
  dasr.save_weights(join('models', 'dasr_x%s_weights.h5' % FLAGS.scale));

if __name__ == "__main__":
  add_options();
  app.run(main);

