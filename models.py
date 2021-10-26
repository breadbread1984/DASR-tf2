#!/usr/bin/python3

from math import log2;
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

class Queue(tf.keras.layers.Layer):
  def __init__(self, k, **kwargs):
    self.k = k;
    super(Queue, self).__init__(**kwargs);
  def build(self, input_shape):
    # input_shape = (batch, dim)
    self.queue = self.add_weight(shape = (input_shape[-1], self.k), dtype = tf.float32, initializer = tf.keras.initializers.RandomNormal(mean = 0, stddev = 1), trainable = False, name = 'queue');
    self.queue.assign(tf.math.l2_normalize(self.queue, axis = 0));
  def call(self, inputs):
    # inputs.shape = (batch, dim)
    retval = tf.identity(self.queue);
    self.queue.assign(tf.concat([self.queue[:,tf.shape(inputs)[0]:], tf.transpose(inputs)], axis = -1));
    return retval;

class MOCO(tf.keras.Model):
  def __init__(self, k = 32 * 256, m = 0.999, t = 0.07, enable_train = True, **kwargs):
    super(MOCO, self).__init__(**kwargs);
    self.k = k;
    self.m = m;
    self.t = t;
    self.enable_train = enable_train;
    self.encoder_q = Encoder();
    self.encoder_k = Encoder();
    self.queue = Queue(k); # self.queue.shape = (256, k)
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
      l_neg = tf.linalg.matmul(q, self.queue(k)); # l_neg.shape = (batch, k)
      logits = tf.concat([l_pos, l_neg], axis = -1); # logits.shape = (batch, 1+k)
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)(tf.zeros((tf.shape(img_q)[0],)), logits / self.t);
      return features, loss;
    else:
      features, _ = self.encoder_q(inputs); # features.shape = (batch, 256)
      return features;

def DA_Conv():
  image = tf.keras.Input((None, None, 64));
  de = tf.keras.Input((64,));
  # 1) branch 1 (image feature convolution with degradation embedding)
  img_results = tf.keras.layers.Lambda(lambda x: tf.reshape(tf.transpose(x, (1,2,3,0)), (1, tf.shape(x)[1], tf.shape(x)[2], -1)))(image); # img_results.shape = (1, height, width, 64 * batch)
  de_results = tf.keras.layers.Dense(units = 64, use_bias = False)(de);
  de_results = tf.keras.layers.LeakyReLU(0.1)(de_results);
  de_results = tf.keras.layers.Dense(units = 3 * 3 * 64, use_bias = False)(de_results); # de_results.shape = (batch, 3 * 3 * 64)
  de_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1, 0)))(de_results); # de_results.shape = (3 * 3 * 64, batch)
  kernel = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (3,3,-1,1)))(de_results); # kernel.shape = (3,3, 64 * batch,1)
  branch1_results = tf.keras.layers.Lambda(lambda x: tf.nn.depthwise_conv2d(x[0], x[1], strides = [1,1,1,1], padding = 'SAME'))([img_results, kernel]); # results.shape = (1, height, width, 64 * batch)
  branch1_results = tf.keras.layers.ReLU()(branch1_results);
  branch1_results = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.reshape(x, (tf.shape(x)[1], tf.shape(x)[2], 64, -1)), (3,0,1,2)))(branch1_results); # results.shape = (batch, height, width, 64)
  branch1_results = tf.keras.layers.Conv2D(64, kernel_size = (1,1), padding = 'same')(branch1_results); # results.shape = (batch, height, width, 64)
  # 2) branch 2 (image feature attention with degradation embedding)
  branch2_results = tf.keras.layers.Dense(64 // 8, use_bias = False)(de); # branch2_results.shape = (batch, 8)
  branch2_results = tf.keras.layers.LeakyReLU(0.1)(branch2_results);
  branch2_results = tf.keras.layers.Dense(64, use_bias = False, activation = tf.keras.activations.sigmoid)(branch2_results); # branch2_results.shape = (batch, 64)
  branch2_results = tf.keras.layers.Reshape((1,1,64))(branch2_results); # branch2_results.shape = (batch, 1, 1, 64)
  branch2_results = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([image, branch2_results]); # branch2_results.shape = (batch, height, width, 64)
  # 3) output
  results = tf.keras.layers.Add()([branch1_results, branch2_results]);
  return tf.keras.Model(inputs = (image, de), outputs = results);

def DABlock():
  image = tf.keras.Input((None, None, 64));
  de = tf.keras.Input((64,));
  results = DA_Conv()([image, de]); # results.shape = (batch, height, width, 64);
  results = tf.keras.layers.LeakyReLU(0.1)(results);
  results = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same')(results);
  results = tf.keras.layers.LeakyReLU(0.1)(results);
  results = DA_Conv()([results, de]); # results.shape = (batch, height, width, 64)
  results = tf.keras.layers.LeakyReLU(0.1)(results);
  results = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same')(results);
  results = tf.keras.layers.Add()([results, image]);
  return tf.keras.Model(inputs = (image, de), outputs = results);

def DAGroup():
  image = tf.keras.Input((None, None, 64));
  de = tf.keras.Input((64,));
  results = image;
  for i in range(5):
    results = DABlock()([results, de]);
  results = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same')(results);
  results = tf.keras.layers.Add()([results, image]);
  return tf.keras.Model(inputs = (image, de), outputs = results);

def Upsampler(scale = 2):
  inputs = tf.keras.Input((None, None, 64));
  results = inputs;
  if log2(scale) == round(log2(scale), 0): # scale is 2^n
    for i in range(int(log2(scale))):
      results = tf.keras.layers.Conv2D(4 * 64, kernel_size = (3,3), padding = 'same')(results); # results.shape = (batch, height, width, channels * 4)
      results = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(results); # results.shape = (batch, height * 2, width * 2, channels)
  elif scale == 3:
    results = tf.keras.layers.Conv2D(9 * 64, kernel_size = (3,3), padding = 'same')(results); # results.shape = (batch, height, width, channels * 9)
    results = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 3))(results); # results.shape = (batch, height * 3, width * 3, channels)
  else:
    raise Exception('unsupported scale!');
  return tf.keras.Model(inputs = inputs, outputs = results);

def DASuperResolution(scale = 2):
  assert log2(scale) == round(log2(scale), 0);
  image = tf.keras.Input((None, None, 3)); # image with mean intensity reduced
  # head
  img_results = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same')(image);
  # compress
  de = tf.keras.Input((256,)); # degradation embedding
  de_results = tf.keras.layers.Dense(units = 64)(de);
  de_results = tf.keras.layers.LeakyReLU(0.1)(de_results);
  # body
  skip = img_results;
  for i in range(5):
    img_results = DAGroup()([img_results, de_results]);
  img_results = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same')(img_results);
  img_results = tf.keras.layers.Add()([img_results, skip]);
  # tail
  img_results = Upsampler(scale)(img_results);
  img_results = tf.keras.layers.Conv2D(3, kernel_size = (3,3), padding = 'same')(img_results);
  return tf.keras.Model(inputs = (image, de), outputs = img_results);

def BlindSuperResolution(scale, enable_train = True):
  inputs = tf.keras.Input((None, None, 3));
  if enable_train:
    key = tf.keras.Input((None, None, 3));
    da_embedding, loss = MOCO(enable_train = enable_train, name = 'moco')([inputs, key]);
  else:
    da_embedding = MOCO(enable_train = enable_train, name = 'moco')(inputs);
  results = DASuperResolution(scale = scale)([inputs, da_embedding]);
  return tf.keras.Model(inputs = (inputs, key) if enable_train else inputs, outputs = (results, loss) if enable_train == True else results);

if __name__ == "__main__":
  import numpy as np;
  inputs = np.random.normal(size = (4, 224,224,3));
  key = np.random.normal(size = (4, 224,224,3));
  bsr = BlindSuperResolution(2);
  with tf.GradientTape() as tape:
    sr, loss = bsr([inputs, key]);
  grads = tape.gradient(loss, bsr.get_layer('moco').encoder_k.trainable_variables);
  print(grads);
  print(sr.shape, loss.shape);
