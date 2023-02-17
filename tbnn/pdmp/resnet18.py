"""
ResNet-18
Reference:
[1] K. He et al. Deep Residual Learning for Image Recognition. CVPR, 2016
[2] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers:
Surpassing human-level performance on imagenet classification. In
ICCV, 2015.
"""

import numpy as np
from tensorflow.keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow_addons.layers import FilterResponseNormalization, TLU
import tensorflow_probability as tfp
tfd = tfp.distributions


from tbnn.pdmp.frn import FRN


def gaussian_prior_loss(layer, var=0.20):
  var = tf.cast(var, dtype=tf.float32)
  # get the neg lop prob, though exclude the
  # constant term out the front.
  neg_log_prob = (tf.reduce_sum(tf.square(layer.kernel)) +
                  tf.reduce_sum(tf.square(layer.bias))) / (2.0 * var)
  return neg_log_prob


def my_conv(channels, kernel_size=[3, 3], strides=[1, 1],
            padding='same', kernel_initializer='he_normal'):
  conv = Conv2D(channels, kernel_size=kernel_size, strides=strides,
                padding=padding, kernel_initializer=kernel_initializer)
  # add prior
  def prior_fn():
    p_scale = tf.cast(np.sqrt(0.20), dtype=tf.float32)
    weight_priors = tfd.Normal(loc=0., scale=p_scale)
    bias_priors = tfd.Normal(loc=0., scale=p_scale)
    # get the neg log likelihood of these
    log_prob = tf.reduce_sum(
      weight_priors.log_prob(conv.kernel) + bias_priors.log_prob(conv.bias))
    return -1.0 * log_prob
  #prior_fn = lambda: gaussian_prior_loss(conv)
  conv.add_loss(prior_fn)
  return conv


def my_dense(units, activation='linear'):
  dense = Dense(units, activation=activation)
  # add prior
  def prior_fn():
    p_scale = tf.cast(np.sqrt(0.20), dtype=tf.float32)
    weight_priors = tfd.Normal(loc=0., scale=p_scale)
    bias_priors = tfd.Normal(loc=0., scale=p_scale)
    # get the neg log likelihood of these
    log_prob = tf.reduce_sum(
      weight_priors.log_prob(dense.kernel) + bias_priors.log_prob(dense.bias))
    return -1.0 * log_prob
  #prior_fn = lambda: gaussian_prior_loss(dense)
  dense.add_loss(prior_fn)
  return dense


def my_flr():
  flr = FilterResponseNormalization()
  # add prior
  def prior_fn():
    p_scale = tf.cast(np.sqrt(0.20), dtype=tf.float32)
    priors = tfd.Normal(loc=0., scale=p_scale)
    # get the neg log likelihood of these
    log_prob = tf.reduce_sum(
      priors.log_prob(flr.gamma) + priors.log_prob(flr.beta))
    return -1.0 * log_prob
  #prior_fn = lambda: gaussian_prior_loss(dense)
  flr.add_loss(prior_fn)
  return flr


def my_tlu():
  tlu = TLU()
  # add prior
  def prior_fn():
    p_scale = tf.cast(np.sqrt(0.20), dtype=tf.float32)
    priors = tfd.Normal(loc=0., scale=p_scale)
    # get the neg log likelihood of these
    log_prob = tf.reduce_sum(
      priors.log_prob(tlu.tau))
    return -1.0 * log_prob
  #prior_fn = lambda: gaussian_prior_loss(dense)
  tlu.add_loss(prior_fn)
  return tlu

class ResnetBlock(Model):
  """
  A standard resnet block.
  """

  def __init__(self, channels: int, down_sample=False, norm_layer=my_flr):#FilterResponseNormalization):
    """
    channels: same as number of convolution kernels
    """
    super().__init__()
    self.__channels = channels
    #self.__strides = [2, 1] if down_sample else [1, 1]
    self.__strides = 2 if down_sample else 1
    self.down_sample = down_sample
    KERNEL_SIZE = (3, 3)
    # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
    INIT_SCHEME = "he_normal"
    self.conv_1 = my_conv(self.__channels, strides=self.__strides,
                          kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
    #self.bn_1 = BatchNormalization()
    self.norm_1 = norm_layer()
    self.tlu_1 = my_tlu()#TLU()
    self.conv_2 = my_conv(self.__channels, strides=1,
               kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
    #self.bn_2 = BatchNormalization()
    self.norm_2 = norm_layer()
    self.tlu_2 = my_tlu()#TLU()
    self.merge = Add()

    if self.down_sample:
      # perform down sampling using stride of 2, according to [1].
      self.res_conv = my_conv(
        self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
      #self.res_bn = BatchNormalization()
      self.norm_res = norm_layer()
      self.tlu_res = my_tlu()#TLU()

  def call(self, inputs):
    res = inputs

    x = self.conv_1(inputs)
    x = self.norm_1(x)
    x = self.tlu_1(x)
    x = tf.keras.activations.swish(x)
    x = self.conv_2(x)
    x = self.norm_2(x)
    x = self.tlu_2(x)

    if self.down_sample:
      res = self.res_conv(res)
      res = self.norm_res(res)
      res = self.tlu_res(res)
    # if not perform down sample, then add a shortcut directly
    x = self.merge([x, res])
    out = tf.keras.activations.swish(x)
    return out


  def resnet_block_prior(self):
    """Creates prior fns for resnet block
    """
    weight_priors =  []
    bias_priors = []
    # need to add prior for two conv blocks
    # and possible a third if we are downsampling this layer
    # need the receptive fields and the input shapes for each layer
    receptive_field_sizes = [np.prod(self.conv_1.kernel_size),
                             np.prod(self.conv_2.kernel_size)]
    kernel_shapes = [self.conv_1.kernel.shape[-2],
                     self.conv_2.kernel.shape[-2]]
    # if we are downsampling, then add the final elements
    # pertaining to the downsampling conv block
    if self.down_sample:
      receptive_field_sizes.append(np.prod(self.res_conv.kernel_size))
      kernel_shapes.append(self.res_conv.kernel.shape[-2])
    for receptive_field_size, kernel_shape in zip(receptive_field_sizes,
                                                  kernel_shapes):
      fan_in = kernel_shape * receptive_field_size
      #p_scale = tf.cast(tf.sqrt(2.0 / fan_in), dtype=tf.float32)
      p_scale = tf.cast(np.sqrt(0.2), dtype=tf.float32)
      weight_priors.append(tfd.Normal(loc=0., scale=p_scale))
      bias_priors.append(tfd.Normal(loc=0., scale=p_scale))
    return weight_priors, bias_priors


class ResNet18(Model):

  def __init__(self, num_classes, **kwargs):
    """
      num_classes: number of classes in specific classification task.
    """
    super().__init__(**kwargs)
    self.likelihood_fn = tfd.OneHotCategorical
    self.conv_1 = my_conv(64, kernel_size=(7, 7), strides=2,
                          padding="same", kernel_initializer="he_normal")
    #self.init_bn = BatchNormalization()
    self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
    self.res_1_1 = ResnetBlock(64)
    self.res_1_2 = ResnetBlock(64)
    self.res_2_1 = ResnetBlock(128, down_sample=True)
    self.res_2_2 = ResnetBlock(128)
    self.res_3_1 = ResnetBlock(256, down_sample=True)
    self.res_3_2 = ResnetBlock(256)
    self.res_4_1 = ResnetBlock(512, down_sample=True)
    self.res_4_2 = ResnetBlock(512)
    self.avg_pool = GlobalAveragePooling2D()
    self.flat = Flatten()
    self.fc = my_dense(num_classes, activation="linear")

  def call(self, inputs):
    out = self.conv_1(inputs)
    #out = self.init_bn(out)
    out = tf.nn.elu(out)
    out = self.pool_2(out)
    for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
      out = res_block(out)
    out = self.avg_pool(out)
    out = self.flat(out)
    out = self.fc(out)
    return out


class ResNet20(Model):

  def __init__(self, num_classes, **kwargs):
    """
      num_classes: number of classes in specific classification task.
    """
    super().__init__(**kwargs)
    self.likelihood_fn = tfd.OneHotCategorical
    self.conv_1 = my_conv(16, kernel_size=(3, 3), strides=2,
                          padding="same", kernel_initializer="he_normal")
    self.norm_1 = FilterResponseNormalization()
    self.res_1_1 = ResnetBlock(16)
    self.res_1_2 = ResnetBlock(16)
    self.res_1_3 = ResnetBlock(16)
    self.res_2_1 = ResnetBlock(32, down_sample=True)
    self.res_2_2 = ResnetBlock(32)
    self.res_2_3 = ResnetBlock(32)
    self.res_3_1 = ResnetBlock(64, down_sample=True)
    self.res_3_2 = ResnetBlock(64)
    self.res_3_3 = ResnetBlock(64)
    self.avg_pool = GlobalAveragePooling2D()
    self.flat = Flatten()
    self.fc = my_dense(num_classes, activation="linear")

  def call(self, inputs):
    out = self.conv_1(inputs)
    out = self.norm_1(out)
    out = tf.nn.swish(out)
    for res_block in [self.res_1_1, self.res_1_2, self.res_1_3,
                      self.res_2_1, self.res_2_2, self.res_2_3,
                      self.res_3_1, self.res_3_2, self.res_3_3]:
      out = res_block(out)
    out = self.avg_pool(out)
    out = self.flat(out)
    out = self.fc(out)
    return out
