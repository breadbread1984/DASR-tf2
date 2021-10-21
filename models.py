#!/usr/bin/python3

import tensorflow as tf;

def DegradationEncoder():
  inputs = tf.keras.Input((None, None, 3)); # inputs.shape = (batch, height, width, 3)
  results = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same')(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU(alpha = 0.1)(results);
  results = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU(alpha = 0.1)(results);
  results = tf.keras.layers.Conv2D(128, kernel_size = (3,3), strides = (2,2), padding = 'same')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU(alpha = 0.1)(results);
  results = tf.keras.layers.Conv2D(128, kernel_size = (3,3), padding = 'same')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU(alpha = 0.1)(results);
  results = tf.keras.layers.Conv2D(256, kernel_size = (3,3), strides = (2,2), padding = 'same')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU(alpha = 0.1)(results);
  results = tf.keras.layers.Conv2D(256, kernel_size = (3,3), padding = 'same')(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.LeakyReLU(alpha = 0.1)(results);
  results = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis = (1,2)))(results); # results.shape = (batch, 256)
  results = tf.keras.layers.Dense(units = 256)(results);
  results = tf.keras.layers.LeakyReLU(alpha = 0.1)(results);
  results = tf.keras.layers.Dense(units = 256)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":
  de = DegradationEncoder();
  import numpy as np;
  inputs = np.random.normal(size = (4, 224,224,3));
  outputs = de(inputs);
  print(outputs.shape);
