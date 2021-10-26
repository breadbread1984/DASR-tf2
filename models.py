#!/usr/bin/python3

from collections import deque;
import tensorflow as tf;

def Encoder():
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
  features = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis = (1,2)))(results); # results.shape = (batch, 256)
  results = tf.keras.layers.Dense(units = 256)(features);
  results = tf.keras.layers.LeakyReLU(alpha = 0.1)(results);
  results = tf.keras.layers.Dense(units = 256)(results);
  return tf.keras.Model(inputs = inputs, outputs = (features, results));

class MOCO(tf.keras.Model):
  def __init__(k = 32 * 256, m = 0.999, t = 0.07, enable_train = True, **kwargs):
    super(MOCO, self).__init__(**kwargs);
    self.k = k;
    self.m = m;
    self.t = t;
    self.enable_train = enable_train;
    self.encoder_q = Encoder();
    if self.enable_train == False: self.encoder_q.trainable = False;
    self.encoder_k = Encoder();
    if self.enable_train == False: self.encoder_k.trainable = False;
    # copy weights from q to k
    self.encoder_k.set_weights(self.encoder_q.get_weights());
    self.queue = deque();
    for i in range(self.k):
      self.queue.append(tf.math.l2_normalize(tf.random.normal(shape = (self.encoder_q.outputs[0].shape[-1],), dtype = tf.float32), axis = 0));
  def call(self, inputs):
    # inputs.shape = (batch, height, width, channels)
    if self.enable_train:
      features, q = self.encoder_q(inputs); # q.shape = (batch, 256)
      q = tf.math.l2_normalize(q, axis = -1);
      # update key encoder with query encoder
      for i in range(len(self.encoder_q.trainable_variables)):
        self.encoder_k.trainable_variables[i] = self.m * self.encoder_k.trainable_variables[i] + (1 - self.m) * self.encoder_q.trainable_variables[i];
      _, k = self.encoder_k(inputs); # k.shape = (batch, 256)
      k = tf.math.l2_normalize(k, axis = -1); # k.shape = (batch, 256)
      l_pos = tf.math.reduce_sum(q * k, axis = -1, keepdims = True); # l_pos.shape = (batch, 1)
      l_neg = tf.linalg.matmul(q, tf.stack(self.queue, axis = -1)); # l_neg.shape = (batch, k)
      logits = tf.concate([l_pos, l_neg], axis = -1) / self.t; # logits.shape = (batch, 1+k)
      labels = tf.zeros((inputs.shape[0],)); # labels.shape = (batch,)
      # update queue
      for i in range(inputs.shape[0]):
        self.queue.popleft();
        self.queue.append(k[i]);
      assert len(self.queue) == self.k;
      return features, logits, labels;
    else:
      features, _ = self.encoder_q(inputs); # features.shape = (batch, 256)
      return features;

if __name__ == "__main__":
  de = DegradationEncoder();
  import numpy as np;
  inputs = np.random.normal(size = (4, 224,224,3));
  outputs = de(inputs);
  print(outputs.shape);
