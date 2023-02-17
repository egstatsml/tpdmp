"""resnet.py


Code to make Resnets.
Adapted from:
https://github.com/google-research/google-research/blob/2ae96f96011ef7b74a941c3131519bf58bfda40e/bnn_hmc/utils/models.py#L293
"""
from tbnn.pdmp.frn import FRN
import numpy as np
import numbers
from tensorflow import keras
from tensorflow.keras.layers import (Dense, Conv2D, MaxPool2D, Flatten,
                                     GlobalAveragePooling2D, ReLU,
                                     BatchNormalization, Layer, Add, InputLayer)
from tensorflow.keras import Model, Input
import tensorflow as tf
from tensorflow_addons.layers import FilterResponseNormalization, TLU
import tensorflow_probability as tfp
tfd = tfp.distributions


class Linear(Layer):
    def __init__(self):
        super(Linear, self).__init__()

    def call(self, inputs):
        return inputs


# def get_prior_neg_log_prob_fn(prior,
#                               kernel_dims=None,
#                               use_bias=True,
#                               is_bias=False):
#     """will get the prior functions needed for this layer

#     Will handle two types of priors:
#       1) Gaussian with all same variance
#       2) Gaussian with variance based on fan in dimensions

#     The option to allow for fan in dimensions is to allow for models
#     which want to set the variance parameter to encourage the output
#     variance to be 1.0.

#     Args:
#       prior (float or str):
#         If the prior argument is a float, then will return a Gaussian with a
#         variance set to this value. If it is a string saying "fan_in", then
#         will scale the  variance such that the output has unit variance
#         (under the prior).
#       kernel_dims (list(int)):
#         dimensions of the kernel for the layer. If it is a a conv layer, will have
#         four dimensions, if dense will have two. These are only to be used if
#         a prior to scale the output variance is needed.
#       use_bias (bool):
#         whether the bias has been included or not. Is needed if aiming to scale
#         the output variance for a specific layer.
#       is_bias (bool):
#         boolean to say whether this is the bias variable or not.
#         if it is the bias variable, but `use_bias` is false, will just
#         set the prior for this variable to `None`

#     Returns:
#       prior negative log probability function as a specified from tfp.

#     Raises:
#       Not implemented error for scaling vartiance. Starting with simple
#       constant variance and will move to the add scaling in a bit. Check
#       my notes for more info.
#     """
#     # check if the specified prior is an float (or int) or a string to say we
#     # should wither have a constant variance.
#     if isinstance(prior, numbers.Number):
#         # if this is the prior for the bias variable, but we aren't using the bias,
#         # then we will just return `None`
#         if(is_bias and not use_bias):
#             return None
#         # otherwise, is either the kernel (which we should always have),
#         # or is the bias and we are actually using it
#         #
#         # then we should just have constant variance
#         # lets cast the this value to a tf float32
#         p_scale = tf.cast(prior, dtype=tf.float32)
#         prior_fn = tfd.Normal(loc=0., scale=p_scale)

#         def prior_constant_var_neg_log_prob(x):
#             log_prob = tf.reduce_sum(prior_fn.log_prob(x))
#             # return negative log prob
#             return -1.0 * log_prob
#         return prior_constant_var_neg_log_prob
#     else:
#         # otherwise need to scale the variance.
#         raise NotImplementedError('Currently only support constant var')


def get_prior_neg_log_prob_fn(prior,
                              kernel_dims=None,
                              use_bias=True,
                              is_bias=False):
  """Will get the prior functions needed for this layer.

  Will handle three types of priors:
    1) Gaussian with all same variance
    2) Gaussian with variance based on fan in dimensions
    3) No prior (if prior arg is None)

  The option to allow for fan in dimensions is to allow for models
  which want to set the variance parameter to encourage the output
  variance to be 1.0.

  Args:
    prior (float or str):
      If the prior argument is a float, then will return a Gaussian with a
      variance set to this value. If it is a string saying "fan_in", then
      will scale the  variance such that the output has unit variance
      (under the prior).
    kernel_dims (list(int)):
      dimensions of the kernel for the layer. If it is a a conv layer, will have
      four dimensions, if dense will have two. These are only to be used if
      a prior to scale the output variance is needed.
    use_bias (bool):
      whether the bias has been included or not. Is needed if aiming to scale
      the output variance for a specific layer.
    is_bias (bool):
      boolean to say whether this is the bias variable or not.
      if it is the bias variable, but `use_bias` is false, will just
      set the prior for this variable to `None`

  Returns
  -------
    prior negative log probability function as a specified from tfp.

  Raises
  ------
    Not implemented error for scaling vartiance. Starting with simple
    constant variance and will move to the add scaling in a bit. Check
    my notes for more info.
  """
  # check if the specified prior is an float (or int) or a string to say we
  # should wither have a constant variance.
  if isinstance(prior, numbers.Number):
    # if this is the prior for the bias variable, but we aren't using the bias,
    # then we will just return `None`
    if(is_bias and not use_bias):
      return None
    # if prior is set to zero, then not using a prior function so return None
    if prior == 0:
      return None
    # otherwise, is either the kernel (which we should always have),
    # or is the bias and we are actually using it
    return prior_constant_var_neg_log_prob(prior)
  else:
    # otherwise need to scale the variance.
    raise NotImplementedError('Currently only support constant var')


def prior_constant_var_neg_log_prob(scale):
  """Prior for constant variance and zero mean.

  Returns a callable that will act a loss function for weight parameters in a
  network.

  Parameters
  ----------
  scale : float
      scale/standard deviation for the prior distribution.

  Returns
  --------
      callable that computes the neg log likelihood over the params.
  """
  # then we should just have constant variance
  # lets cast the this value to a tf float32
  # p_scale = tf.cast(scale, dtype=tf.float32)
  # prior_fn = tfd.Normal(loc=0., scale=p_scale)
  # def _fn(x):
  #   log_prob = tf.reduce_sum(prior_fn.log_prob(x))
  #   # return negative log prob
  #   return -1.0 * log_prob
  # return _fn
  # return None
  return tf.keras.regularizers.L2(l2=1.0 / scale)




class Swish(Layer):
    def __init__(self):
        super(Swish, self).__init__()

    def call(self, inputs):
        return keras.activations.swish(inputs)


# def prior_fn(layer, var=0.5):
#   var = tf.cast(var, dtype=tf.float32)
#   def fn():
#     # get the neg lop prob, though exclude the
#     # constant term out the front.
#     neg_log_prob = (tf.reduce_sum(tf.square(layer.kernel)) +
#                     tf.reduce_sum(tf.square(layer.bias))) / (2.0 * var)
#     return neg_log_prob
#   return fn

def prior_fn(layer, var=5.0):
    def fn():
        p_scale = tf.cast(np.sqrt(var), dtype=tf.float32)
        weight_priors = tfd.Normal(loc=0., scale=p_scale)
        bias_priors = tfd.Normal(loc=0., scale=p_scale)
        # get the neg log likelihood of these
        log_prob = tf.reduce_sum(
            weight_priors.log_prob(layer.kernel)
            + bias_priors.log_prob(layer.bias))
        return -1.0 * log_prob
    return fn


class ResNetLayer(Layer):
    def __init__(self, filters, kernel_size=3,
                 strides=1, activation=Swish,
                 normalization_layer=FRN,
                 use_bias=True,
                 prior=1.0,
                 init_method="he_normal"):
        super(ResNetLayer, self).__init__()
        self.conv = Conv2D(filters, kernel_size, strides=strides,
                           padding="same", kernel_initializer=init_method,
                           kernel_regularizer=get_prior_neg_log_prob_fn(
                               prior, use_bias=use_bias),
                           bias_regularizer=get_prior_neg_log_prob_fn(
                               prior, use_bias=use_bias, is_bias=True),
                           use_bias=use_bias)
        self.norm = normalization_layer()
        self.activation = activation()
        self.use_bias = use_bias
        # self.add_loss(prior_fn(self.conv))

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        x = self.activation(x)
        # self.add_loss(prior_fn(self.conv))
        return x


def make_resnet(num_classes,
                depth,
                input_shape,
                width=16,
                activation=Swish,
                normalization_layer=FRN,
                prior=1.0):
    print('input dims = {}'.format(input_shape))
    print('prior val = {}'.format(prior))
    num_res_blocks = (depth - 2) // 6
    if (depth - 2) % 6 != 0:
        raise ValueError("depth must be 6n+2 (e.g. 20, 32, 44).")
    num_filters = width
    model_input = Input(shape=input_shape)
    print('model_input = {}'.format(model_input))
    # if the input is larger than 100 x 100, will do a conv stage
    # with 7 x 7 and a pooling stage similar to original resnet paper
    if input_shape[0] >= 100:
      x = Conv2D(7, 64, strides=2,
                 padding="same", kernel_initializer='he_normal',
                 kernel_regularizer=get_prior_neg_log_prob_fn(
                   prior, use_bias=True),
                 bias_regularizer=get_prior_neg_log_prob_fn(
                   prior, use_bias=True, is_bias=True),
                 use_bias=True)(model_input)
      x = MaxPool2D((3,3), strides=2)(x)
    else:
      x = model_input
    # first residual layer
    x = ResNetLayer(num_filters,
                    strides=1,
                    activation=activation,
                    normalization_layer=normalization_layer,
                    prior=prior)(x)
    # now go through the three stacks
    for stack in range(3):
        # now make the number of res blocks needed for different resnets
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = ResNetLayer(num_filters,
                            strides=strides,
                            activation=activation,
                            normalization_layer=normalization_layer,
                            prior=prior)(x)
            # next layer has no activation applied and a kernel size of three
            # and always a stride of one
            y = ResNetLayer(num_filters,
                            strides=1,
                            activation=Linear,
                            normalization_layer=normalization_layer,
                            prior=prior)(y)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # ie strides is equal to two
                # linear projection residual shortcut connection to match changed dims
                # always has a kernel of size one
                x = ResNetLayer(num_filters,
                                kernel_size=1,
                                strides=strides,
                                activation=Linear,
                                normalization_layer=normalization_layer,
                                prior=prior)(x)
            # now add residual to output of conv stages
            x = Add()([x, y])
            # apply final activation for block
            x = activation()(x)
        # increase the number of filters for the next stack
        num_filters *= 2
    # done with resnet blocks, use global averaging and move onto classification
    x = GlobalAveragePooling2D()(x)
    dense_out = Dense(num_classes,
                      kernel_regularizer=get_prior_neg_log_prob_fn(
                          prior, use_bias=True),
                      bias_regularizer=get_prior_neg_log_prob_fn(
                          prior, use_bias=True, is_bias=True),
                      activation=None)
    #dense_out.add_loss(prior_fn(dense_out, var=5.0))
    logits = dense_out(x)
    model = Model(inputs=model_input, outputs=logits)
    return model
