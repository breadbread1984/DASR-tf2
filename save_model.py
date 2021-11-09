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
  # 1) load from checkpoints
  dasr = tf.keras.models.load_model(join(FLAGS.checkpoint, 'dasr_ckpt'), custom_objects = {'tf': tf}, compile = True);
  # 2) extract trained layers and recompile a new model
  moco = dasr.get_layer('moco');
  sr = dasr.get_layer('sr');
  encoder = moco.encoder_q;
  inputs = tf.keras.Input((None, None, 3));
  da_embedding, _ = encoder(inputs);
  results = sr([inputs, da_embedding]);
  dasr = tf.keras.Model(inputs = inputs, outputs = results);
  # 3) save model
  if not exists('models'): mkdir('models');
  dasr.save(join('models', 'dasr_x%s.h5' % FLAGS.scale));

if __name__ == "__main__":
  add_options();
  app.run(main);

