"""Code to make common network's."""

import numbers
import time
import functools

from tensorflow.keras.layers import (Dense, Conv2D, MaxPool2D, Flatten,
                                     AvgPool2D, Input, BatchNormalization)
# from tensorflow_probability.layers import (DenseReparameterization,
#                                            Convolutional2DReparameterization)
from tensorflow.keras import Model
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow_probability.python.distributions import independent as independent_lib
from tbnn.pdmp.resnet import make_resnet
from tbnn.pdmp.resnet50 import (make_resnet_nf_swish, make_resnet_bn_swish,
                                make_resnet18_bn_swish)
from tbnn.pdmp.nfnets import (create_ECA_NFNetF0, create_NFNet_small,
                              create_nf_resnet50)

tfd = tfp.distributions

# permitted values for the different arguments specified by cmdline args
VALID_NETWORKS = [
    'lenet5', 'resnet20', 'resnet18', 'resnet50', 'kaggle', 'small_cnn', 'small_regression',
    'retinopathy', 'uci_mlp', 'nfnet', 'nfnet2', 'resnet20_wide', 'med_mnist',
    'cifar_alexnet', 'resnet50_keras'
]
VALID_PRIOR_STRS = ['fan_in']
VALID_LIKELIHOOD = ['normal', 'bernoulli', 'categorical']


class InvalidLikelihood(ValueError):
  """Exception to be raised if invalid likelihood specified."""

  def __init__(self, arg):
    print('Invalid likelihood function specified in config file')
    print('Supplied {}, available loss functions are {}'.format(
        arg, ['normal', 'bernoulli', 'categorical']))
    ValueError.__init__(self)


def check_network_arg(arg):
  """Check command line argument for the network string is valid."""
  if arg.lower() not in VALID_NETWORKS:
    raise ValueError(
        'Invalid network type supplied. Expect one of {}, got {}'.format(
            VALID_NETWORKS, arg))


def check_format_prior_arg(arg):
  """Check command line argument for the prior and formats if needed.

  Will want to format it if the supplied argument is a number, as the arg will
  be interpretted from the commandline as a string. If it is a number, convert
  it to a float, otherwise just make sure it is valid
  """
  try:
    prior = float(arg)
    # if it made it past here, then it is a float
    # do some quick error checking to make sure it isn't negative
    if prior < 0:
      raise ValueError(
          'Invalid prior specified. If a float, must be positive. got {}'.
          format(prior))
  except ValueError:
    # just check the valid string
    if arg.lower() not in VALID_PRIOR_STRS:
      raise ValueError(
          'Invalid prior specified. If a str, must be one of {}. got {}'.format(
              VALID_PRIOR_STRS, prior))
    prior = arg
  return prior


def check_likelihood_arg(arg):
  """Check the likelihood argument supplied is valid."""
  if arg.lower() not in VALID_LIKELIHOOD:
    raise ValueError(
        'Invalid likelihood specified. Expected one of{}. got {}'.format(
            VALID_LIKELIHOOD, arg))


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
    if (is_bias and not use_bias):
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


def default_multivariate_normal_fn(dtype, shape, name, trainable,
                                   add_variable_fn, scale):
  """Creates multivariate standard `Normal` distribution.

  This is modified directly from tensorflow_probability source,
  excepts adds the ability to specify the prior variance.

  Args:
    dtype: Type of parameter's event.
    shape: Python `list`-like representing the parameter's event shape.
    name: Python `str` name prepended to any created (or existing)
      `tf.Variable`s.
    trainable: Python `bool` indicating all created `tf.Variable`s should be
      added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
    add_variable_fn: `tf.get_variable`-like `callable` used to create (or
      access existing) `tf.Variable`s.
    scale: float for prior std.
  Returns:
    Multivariate standard `Normal` distribution.
  """
  del name, trainable, add_variable_fn   # unused
  dist = normal_lib.Normal(
      loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(scale))
  batch_ndims = tf.size(dist.batch_shape_tensor())
  return independent_lib.Independent(
      dist, reinterpreted_batch_ndims=batch_ndims)


def get_vi_prior(prior, is_bias=False, use_bias=True):
  # creating a wrapper for my default normal fn so that it
  # is compatable with tfp, and can also accept prior scales
  def _multivariate_normal_wrapper(dtype, shape, name, trainable,
                                   add_variable_fn):
    return default_multivariate_normal_fn(dtype, shape,
                                          name, trainable,
                                          add_variable_fn, prior)

  # check if the specified prior is an float (or int) or a string to say we
  # should wither have a constant variance.
  #
  #
  if isinstance(prior, numbers.Number):
    # if this is the prior for the bias variable, but we aren't using the bias,
    # then we will just return `None`
    if (is_bias and not use_bias):
      return None
    # if prior is set to zero, then not using a prior function so return None
    if prior == 0:
      return None
    # otherwise return the wrapper for prior fn with the specified scale
    return _multivariate_normal_wrapper
  elif prior == None:
    return None
  else:
    raise ValueError(f'Invalid prior specified: {prior}')

def get_likelihood_fn(likelihood_str):
  """Get likelihood for the current model.

  For the MCMC models it should be the name of a TFP distribution.
  This method checks that the likelihood supplied is valid, and then
  sets the returns the corresponding likelihood fn.

  Args:
    output_dims (str):
      dimension of the output
    likelihood_str (str):
      string to specify distribution used to model the likelihood

  Returns
  -------
      the input argument if it is of a valid form

  Raises
  ------
      `ValueError()` if incorrect likelihood str
  """
  likelihood_str = likelihood_str.lower()
  # check the likelihood string
  check_likelihood_arg(likelihood_str)
  # normal dist
  if (likelihood_str == 'normal'):
    dist = tfd.Normal
  # binary classification
  elif (likelihood_str == 'bernoulli'):
    dist = tfd.Bernoulli
  # categorical classification
  elif (likelihood_str == 'categorical'):
    dist = tfd.OneHotCategorical
  else:
    raise ValueError(('Problem with the likelihood specified. '
                      'Should not of made it here, making it here means '
                      'that the error checking done to see if your '
                      'likelihood is valid is dang broke. '
                      'Expected something like {}, but got {}').format(
                          VALID_LIKELIHOOD, likelihood_str))
  return dist


def get_lenet_5(input_dims,
                num_classes,
                use_bias=True,
                prior=1.0,
                activation='swish',
                vi=False):
  # print('input dims = {}'.format(input_dims))
  # # build the model
  # inputs = Input(input_dims)
  # x = Conv2D(6,
  #            kernel_size=5,
  #            padding='SAME',
  #            kernel_regularizer=get_prior_neg_log_prob_fn(prior,
  #                                                         use_bias=use_bias),
  #            use_bias=use_bias,
  #            bias_regularizer=get_prior_neg_log_prob_fn(prior,
  #                                                       use_bias=use_bias,
  #                                                       is_bias=True),
  #            activation=activation)(inputs)
  # x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(x)
  # x = Conv2D(16,
  #            kernel_size=5,
  #            padding='SAME',
  #            kernel_regularizer=get_prior_neg_log_prob_fn(prior,
  #                                                         use_bias=use_bias),
  #            use_bias=use_bias,
  #            bias_regularizer=get_prior_neg_log_prob_fn(prior,
  #                                                       use_bias=use_bias,
  #                                                       is_bias=True),
  #            activation=activation)(x)
  # x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(x)
  # x = Conv2D(120,
  #            kernel_size=5,
  #            padding='SAME',
  #            kernel_regularizer=get_prior_neg_log_prob_fn(prior,
  #                                                         use_bias=use_bias),
  #            use_bias=use_bias,
  #            bias_regularizer=get_prior_neg_log_prob_fn(prior,
  #                                                       use_bias=use_bias,
  #                                                       is_bias=True),
  #            activation=activation)(x)
  # x = Flatten()(x)
  # x = Dense(84,
  #           kernel_regularizer=get_prior_neg_log_prob_fn(prior,
  #                                                        use_bias=use_bias),
  #           use_bias=use_bias,
  #           bias_regularizer=get_prior_neg_log_prob_fn(prior,
  #                                                      use_bias=use_bias,
  #                                                      is_bias=True),
  #           activation=activation)(x)
  # outputs = Dense(num_classes,
  #                 kernel_regularizer=get_prior_neg_log_prob_fn(
  #                     prior, use_bias=use_bias),
  #                 use_bias=use_bias,
  #                 bias_regularizer=get_prior_neg_log_prob_fn(prior,
  #                                                            use_bias=use_bias,
  #                                                            is_bias=True),
  #                 activation=None)(x)

  inputs = Input(input_dims)
  layer_one = get_conv_layer(6, 5, padding='SAME',
                              activation=activation,
                              prior=prior,
                              vi=vi)
  x = layer_one(inputs)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(x)
  layer_two = get_conv_layer(16, 5, padding='SAME',
                              activation=activation,
                              prior=prior,
                              vi=vi)
  x = layer_two(x)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(x)
  layer_three = get_conv_layer(120, 5, padding='SAME',
                              activation=activation,
                              prior=prior,
                              vi=vi)
  x = layer_three(x)
  x = Flatten()(x)
  layer_four = get_dense_layer(84,
                               activation=activation,
                              prior=prior,
                              vi=vi)
  x = layer_four(x)
  layer_five = get_dense_layer(num_classes,
                              activation=activation,
                              prior=prior,
                              vi=vi)
  outputs = layer_five(x)
  lenet5 = Model(inputs, outputs)
  return lenet5


def get_small_cnn(input_dims,
                  num_classes,
                  use_bias=True,
                  prior=1.0,
                  activation='relu'):
  # build the model
  inputs = Input(input_dims)
  x = Conv2D(16,
             kernel_size=3,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(inputs)
  x = Conv2D(32,
             kernel_size=3,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(x)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(x)
  x = Conv2D(32,
             kernel_size=3,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(x)
  x = Conv2D(64,
             kernel_size=3,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(x)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(x)
  x = Flatten()(x)
  x = Dense(256,
            kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                         use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                       use_bias=use_bias,
                                                       is_bias=True),
            activation=activation)(x)
  outputs = Dense(num_classes,
                  kernel_regularizer=get_prior_neg_log_prob_fn(
                      prior, use_bias=use_bias),
                  use_bias=use_bias,
                  bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                             use_bias=use_bias,
                                                             is_bias=True),
                  activation=None)(x)
  small_cnn = Model(inputs, outputs)
  return small_cnn


def get_conv_layer(units,
                   kernel_size,
                   strides=1,
                   padding='SAME',
                   activation=None,
                   prior=None,
                   use_bias=True,
                   vi=False):
  if not vi:
    kernel_prior_fn = get_prior_neg_log_prob_fn(prior,
                                                use_bias=use_bias,
                                                is_bias=False)
    bias_prior_fn = get_prior_neg_log_prob_fn(prior,
                                              use_bias=use_bias,
                                              is_bias=True)
    conv_layer = Conv2D(units,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         kernel_regularizer=kernel_prior_fn,
                         use_bias=use_bias,
                         bias_regularizer=bias_prior_fn,
                         activation=activation)
  else:
    kernel_prior_fn = get_vi_prior(prior, use_bias=use_bias)
    bias_prior_fn = get_vi_prior(prior, use_bias=use_bias, is_bias=True)
    kl_divergence_function = (
        lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
        tf.cast(1, dtype=tf.float32))
    conv_layer = tfp.layers.Convolution2DReparameterization(
        units,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_divergence_fn=kl_divergence_function,
        kernel_prior_fn=kernel_prior_fn,
        bias_prior_fn=bias_prior_fn,
        bias_divergence_fn=kl_divergence_function,
        activation=activation)
  return conv_layer

def get_dense_layer(units, activation=None, prior=None, use_bias=True, vi=False):
  if not vi:
    kernel_prior_fn = get_prior_neg_log_prob_fn(prior, use_bias=use_bias, is_bias=False)
    bias_prior_fn = get_prior_neg_log_prob_fn(prior,
                                              use_bias=use_bias,
                                              is_bias=True)
    print(kernel_prior_fn)
    print(bias_prior_fn)
    time.sleep(3)
    dense_layer = Dense(units,
                        kernel_regularizer=kernel_prior_fn,
                        use_bias=use_bias,
                        bias_regularizer=bias_prior_fn,
                        activation=activation)
  else:
    kernel_prior_fn = get_vi_prior(prior, use_bias=use_bias)
    bias_prior_fn = get_vi_prior(prior,
                                 use_bias=use_bias,
                                 is_bias=True)
    # kernel_prior_fn = None
    # bias_prior_fn = None
    print(kernel_prior_fn)
    # time.sleep(10)
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                            tf.cast(1, dtype=tf.float32))
    dense_layer = tfp.layers.DenseReparameterization(units,
                                                     kernel_divergence_fn=kl_divergence_function,
                                                     kernel_prior_fn=kernel_prior_fn,
                                                     bias_prior_fn=bias_prior_fn,
                                                     bias_divergence_fn=kl_divergence_function,
                                                     activation=activation)
  return dense_layer

def get_small_regression(input_dims,
                         out_dims,
                         prior=1.0,
                         use_bias=True,
                         vi=False,
                         activation='relu'):
  # inputs = Input(input_dims)
  # layer_one = get_dense_layer(100,
  #                             activation=activation,
  #                             prior=prior,
  #                             vi=vi)
  # x = layer_one(inputs)
  # layer_two = get_dense_layer(50,
  #                             activation=activation,
  #                             prior=prior,
  #                             vi=vi)
  # x = layer_two(x)
  # layer_three = get_dense_layer(out_dims,
  #                               activation=activation,
  #                               prior=prior,
  #                               vi=vi)
  # outputs = layer_three(x)
  # small_regression = Model(inputs, outputs)

  inputs = Input(input_dims)
  x = Dense(128,
            kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                         use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                       use_bias=use_bias,
                                                       is_bias=True),
            activation=activation)(inputs)
  x = Dense(128,
            kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                         use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                       use_bias=use_bias,
                                                       is_bias=True),
            activation=activation)(x)
  x = Dense(128,
            kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                         use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                       use_bias=use_bias,
                                                       is_bias=True),
            activation=activation)(x)

  # x = Dense(512,
  #           kernel_regularizer=get_prior_neg_log_prob_fn(prior,
  #                                                        use_bias=use_bias),
  #           use_bias=use_bias,
  #           bias_regularizer=get_prior_neg_log_prob_fn(prior,
  #                                                      use_bias=use_bias,
  #                                                      is_bias=True),
  #           activation=activation)(x)
  # x = Dense(512,
  #           kernel_regularizer=get_prior_neg_log_prob_fn(
  #             prior, use_bias=use_bias),
  #           use_bias=use_bias,
  #           bias_regularizer=get_prior_neg_log_prob_fn(
  #             prior, use_bias=use_bias, is_bias=True),
  #           activation=activation)(x)
  # x = Dense(512,
  #           kernel_regularizer=get_prior_neg_log_prob_fn(
  #             prior, use_bias=use_bias),
  #           use_bias=use_bias,
  #           bias_regularizer=get_prior_neg_log_prob_fn(
  #             prior, use_bias=use_bias, is_bias=True),
  #           activation=activation)(x)
  # x = Dense(10,
  #           kernel_regularizer=get_prior_neg_log_prob_fn(
  #             prior, use_bias=use_bias),
  #           use_bias=use_bias,
  #           bias_regularizer=get_prior_neg_log_prob_fn(
  #             prior, use_bias=use_bias, is_bias=True),
  #           activation='leaky_relu')(x)
  # x = Dense(10,
  #           kernel_regularizer=get_prior_neg_log_prob_fn(prior,
  #                                                        use_bias=use_bias),
  #           use_bias=use_bias,
  #           bias_regularizer=get_prior_neg_log_prob_fn(prior,
  #                                                      use_bias=use_bias,
  #                                                      is_bias=True),
  #           activation=activation)(x)
  outputs = Dense(out_dims,
                  kernel_regularizer=get_prior_neg_log_prob_fn(
                      prior, use_bias=use_bias),
                  use_bias=use_bias,
                  bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                             use_bias=use_bias,
                                                             is_bias=True),
                  activation=None)(x)
  small_regression = Model(inputs, outputs)
  return small_regression

  return small_regression


def get_uci_mlp(input_dims,
                out_dims,
                use_bias=True,
                prior=1.0,
                activation='relu'):
  inputs = Input(input_dims)
  x = Dense(50,
            kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                         use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                       use_bias=use_bias,
                                                       is_bias=True),
            activation=activation)(inputs)
  outputs = Dense(2,
                  kernel_regularizer=get_prior_neg_log_prob_fn(
                      prior, use_bias=use_bias),
                  use_bias=use_bias,
                  bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                             use_bias=use_bias,
                                                             is_bias=True),
                  activation=None)(x)
  uci_mlp = Model(inputs, outputs)
  return uci_mlp


def get_med_mnist(input_dims,
                  num_classes,
                  use_bias=True,
                  prior=1.0,
                  activation='relu'):
  print('input dims = {}'.format(input_dims))
  # build the model
  inputs = Input(input_dims)
  x = Conv2D(6,
             kernel_size=5,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(inputs)
  x = AvgPool2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(x)
  x = Conv2D(16,
             kernel_size=5,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(x)
  x = AvgPool2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(x)
  x = Flatten()(x)
  x = Dense(120,
            kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                         use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                       use_bias=use_bias,
                                                       is_bias=True),
            activation=activation)(x)
  x = Dense(84,
            kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                         use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                       use_bias=use_bias,
                                                       is_bias=True),
            activation=activation)(x)
  outputs = Dense(num_classes,
                  kernel_regularizer=get_prior_neg_log_prob_fn(
                      prior, use_bias=use_bias),
                  use_bias=use_bias,
                  bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                             use_bias=use_bias,
                                                             is_bias=True),
                  activation=None)(x)
  med_mnist = Model(inputs, outputs)
  return med_mnist


def get_cifar_alexnet(input_dims,
                      num_classes,
                      use_bias=True,
                      prior=1.0,
                      activation='swish'):
  print('input dims = {}'.format(input_dims))
  # build the model
  inputs = Input(input_dims)
  x = Conv2D(64,
             kernel_size=3,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(inputs)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='VALID')(x)
  x = Conv2D(128,
             kernel_size=3,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(x)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='VALID')(x)
  x = Conv2D(256,
             kernel_size=2,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(x)
  x = Conv2D(128,
             kernel_size=2,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(x)
  x = Conv2D(64,
             kernel_size=2,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(x)
  x = Flatten()(x)
  x = Dense(256,
            kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                         use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                       use_bias=use_bias,
                                                       is_bias=True),
            activation=activation)(x)
  x = Dense(256,
            kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                         use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                       use_bias=use_bias,
                                                       is_bias=True),
            activation=activation)(x)
  outputs = Dense(num_classes,
                  kernel_regularizer=get_prior_neg_log_prob_fn(
                      prior, use_bias=use_bias),
                  use_bias=use_bias,
                  bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                             use_bias=use_bias,
                                                             is_bias=True),
                  activation=None)(x)
  cifar_alexnet = Model(inputs, outputs)
  return cifar_alexnet


def get_retinopathy(input_dims,
                    num_classes,
                    use_bias=True,
                    prior=1.0,
                    activation='swish'):
  print('input dims = {}'.format(input_dims))
  # build the model
  inputs = Input(input_dims)
  x = Conv2D(32,
             kernel_size=3,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(inputs)
  x = Conv2D(32,
             kernel_size=3,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(x)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='VALID')(x)
  x = Conv2D(32,
             kernel_size=3,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(x)
  x = Conv2D(32,
             kernel_size=3,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(x)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='VALID')(x)
  x = Conv2D(16,
             kernel_size=3,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(x)
  x = Conv2D(16,
             kernel_size=3,
             padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                          use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                        use_bias=use_bias,
                                                        is_bias=True),
             activation=activation)(x)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='VALID')(x)
  x = Flatten()(x)
  x = Dense(128,
            kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                         use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                       use_bias=use_bias,
                                                       is_bias=True),
            activation=activation)(x)
  outputs = Dense(num_classes,
                  kernel_regularizer=get_prior_neg_log_prob_fn(
                      prior, use_bias=use_bias),
                  use_bias=use_bias,
                  bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                             use_bias=use_bias,
                                                             is_bias=True),
                  activation=None)(x)
  retinopathy = Model(inputs, outputs)
  return retinopathy


def get_linear(input_dims,
               out_dims,
               use_bias=True,
               prior=1.0,
               activation='None'):
  inputs = Input(input_dims)
  x = Dense(50,
            kernel_regularizer=get_prior_neg_log_prob_fn(prior,
                                                         use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(prior,
                                                       use_bias=use_bias,
                                                       is_bias=True),
            activation=activation)(inputs)

def load_keras_resnet50(prior, strategy):
  with strategy.scope():
    model = tf.keras.applications.resnet50.ResNet50(
      input_shape=(224, 224, 3),
      include_top=True,
      weights='imagenet',
      classifier_activation=None)
    # _ = model(tf.random.normal(1,224,224,3))
    # now need to add prior to all the conv and dense layers
    prior_fn = get_prior_neg_log_prob_fn(prior,
                                         kernel_dims=None,
                                         use_bias=True,
                                         is_bias=False)
    def _prior_for_variable(v):
      """Creates a regularization loss `Tensor` for variable `v`."""
      prior = prior_fn(v)
      return prior


    for layer in model.layers:
      if isinstance(layer, (Dense, Conv2D)):
        layer.add_loss(functools.partial(_prior_for_variable, layer.kernel))
        if layer.use_bias:
          layer.add_loss(functools.partial(_prior_for_variable, layer.bias))
      # if the current layer is batch noramalization, we should make
      # it NOT trainable now
      # This will make it run in inference mode, and not the params for it
      # be deterministic and static
      elif isinstance(layer, (BatchNormalization)):
        layer.trainable = False

  return model









def get_network(network,
                strategy,
                input_dims,
                output_dims,
                prior,
                vi=False,
                use_bias=True,
                activation='relu',
                map_training=False):
  # map training arg only needed for nfnet which we are fine tuning
  with strategy.scope():
    if network == 'lenet5':
      model = get_lenet_5(input_dims, output_dims, use_bias, prior, vi=vi)
    elif network == 'resnet20':
      model = make_resnet(output_dims, 20, input_dims, prior=prior)
    elif network == 'resnet20_wide':
      model = make_resnet(output_dims, 20, input_dims, width=64, prior=prior)

    elif network == 'resnet50':
      # TODO: Tidy this up and review this design
      # for resnet50, am currently only supporting use of constant variance
      # prior.
      # am calling the get_prior function, as it currently only supports
      # constant prior, and will throw an exception if not a number.
      # am setting use_bias and is_bias to True, so that it will definitely
      # return something for me and not None. See the get_prior fn for me info
      # about it's logic
      prior_loss = get_prior_neg_log_prob_fn(prior, use_bias=True, is_bias=True)

      # model = create_nf_resnet50(input_dims,
      # output_dims)
      # prior_loss=prior_loss)
      #
      # use_bias always true for these models
      model = make_resnet_nf_swish(input_dims,
                                   output_dims,
                                   prior_loss=prior_loss)
      # model = make_resnet_bn_swish(input_dims,
      #                              output_dims,
      #                              prior_loss=prior_loss)
      # model = tf.keras.applications.ResNet50(
      #   weights=None,  # Load weights pre-trained on ImageNet.
      #   input_shape=(192, 192, 3),
      #   include_top=True)
    elif network == 'resnet18':
      prior_loss = get_prior_neg_log_prob_fn(prior, use_bias=True, is_bias=True)
      model = make_resnet18_bn_swish(input_dims,
                                     output_dims,
                                     prior_loss=prior_loss)
    elif network == 'nfnet':
      prior_loss = get_prior_neg_log_prob_fn(prior, use_bias=True, is_bias=True)
      model = create_ECA_NFNetF0(prior_fn=prior_loss, fine_tuning=map_training)

    elif network == 'nfnet2':
      prior_loss = get_prior_neg_log_prob_fn(prior, use_bias=True, is_bias=True)
      model = create_NFNet_small(prior_fn=prior_loss, fine_tuning=map_training)
    elif network == 'small_cnn':
      model = get_small_cnn(input_dims, output_dims, prior, use_bias)
    elif network == 'small_regression':
      model = get_small_regression(input_dims, output_dims, prior, use_bias, vi)
    elif network == 'uci_mlp':
      model = get_uci_mlp(input_dims, 2, prior, use_bias)
    elif network == 'med_mnist':
      model = get_med_mnist(input_dims, 7, prior, use_bias)
    elif network == 'cifar_alexnet':
      model = get_cifar_alexnet(input_dims, 10, prior, use_bias)
    elif network == 'retinopathy':
      model = get_retinopathy(input_dims, output_dims, prior, use_bias)

    elif network == 'resnet50_keras':
      # try loading it in a bit
      pass
    else:
      raise ValueError('Invalid network supplied, got {}'.format(network))
  if network == 'resnet50_keras':
    model = load_keras_resnet50(prior, strategy)


  return model
