#!/usr/bin/python3

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
    self.encoder_k = Encoder();
    self.queue = [tf.math.l2_normalize(tf.random.normal(shape = (self.encoder_q.outputs[0].shape[-1],), dtype = tf.float32)) for i in range(self.k)];
    # copy weights from q to k
    self.encoder_k.set_weights(self.encoder_q.get_weights());
  def call(self, inputs):
    if self.enable_train:
      img_q, img_k = inputs;
      # img_q.shape = (batch, height, width, channels)
      # img_k.shape = (batch, height, width, channels)
      features, q = self.encoder_q(img_q); # q.shape = (batch, 256)
      q = tf.math.l2_normalize(q, axis = -1);
      # update key encoder with query encoder
      for i in range(len(self.encoder_q.trainable_variables)):
        self.encoder_k.trainable_variables[i] = self.m * self.encoder_k.trainable_variables[i] + (1 - self.m) * self.encoder_q.trainable_variables[i];
      _, k = self.encoder_k(img_k); # k.shape = (batch, 256)
      k = tf.math.l2_normalize(k, axis = -1); # k.shape = (batch, 256)
      l_pos = tf.math.reduce_sum(q * k, axis = -1, keepdims = True); # l_pos.shape = (batch, 1)
      l_neg = tf.linalg.matmul(q, tf.stop_gradient(tf.stack(self.queue, axis = -1))); # l_neg.shape = (batch, k)
      logits = tf.concate([l_pos, l_neg], axis = -1); # logits.shape = (batch, 1+k)
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)(tf.zeros((img_q.shape[0],)), logits / self.t);
      # update queue
      del self.queue[:img_q.shape[0]];
      self.queue.extend([tf.squeeze(e, axis = 0) for e in tf.split(k, img_q.shape[0], axis = 0)]);
      assert len(self.queue) == self.k;
      return features, loss;
    else:
      features, _ = self.encoder_q(inputs); # features.shape = (batch, 256)
      return features;

def DASR():
  inputs = tf.keras.Input((None, None, 256));
  results = tf.keras.layers.Dense(units = 64)(inputs);
  results = tf.keras.layers.LeakyReLU(0.1)(results);

def BlindSR(enable_train = True):
  query = tf.keras.Input((None, None, 3));
  if enable_train:
    key = tf.keras.Input((None, None, 3));
    features, loss = MOCO(enable_train = enable_train)(query, key);
  else:
    features = MOCO(enable_train = enable_train)(query);
  

if __name__ == "__main__":
  de = DegradationEncoder();
  import numpy as np;
  inputs = np.random.normal(size = (4, 224,224,3));
  outputs = de(inputs);
  print(outputs.shape);
