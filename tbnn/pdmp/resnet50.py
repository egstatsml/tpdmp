"""
Code to create resnet50.

Largely taken from
https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/applications/resnet.py#L440-L459 # pylint: disable=line-too-long
and just adapted for what I needed.

Am replacing the batch normalisation with filter response normalisation so the
output doesnt depend on any batch level stats, and switching to the swish
activation function.
"""
from typing import Optional, Callable
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

from tbnn.pdmp.nfnets import activation_by_name_with_gamma
from tbnn.pdmp.frn import FRN
from tbnn.pdmp import nfnets
from tbnn.pdmp.activations import activation_by_name


def apply_normalization(inputs, norm_layer, name):
  """Apply normalization for this module.

  Have formatted this module to be reasonably flexible when it comes to
  different regularization methods. It can handle batchnorm, FRN and no
  regularization (eg. when using nfnets). This function is a wrapper to ensure
  hyperparams are set correctly when handling the different norm layers. If it
  is batchnorm, will just set the epsilon value to 1.001e-5 (as is done in the
  keras models). For FRN will just use default values, and for no normalization,
  will just return the input.

  Parameters
  ----------
  inputs : tensor
      input to the normalization layers.
  norm_layer : Layer or None
      normalization layer to be applied.
  name : str
      string to name the layer

  Returns
  -------
      inputs through normalization layer, or None if norm layer is None.
  """
  if norm_layer is None:
    return inputs
  elif norm_layer is layers.BatchNormalization:
    return layers.BatchNormalization(epsilon=1.001e-5, name=name + 'bn')(inputs)
  elif norm_layer is FRN:
    return FRN(name=name + 'frn')(inputs)
  else:
    raise ValueError((f'Invalid/unsupported normalization layer specified.'
                      f'Got {norm_layer.__class__.__name__}'))


def apply_activation_layer(inputs, activation, name):
  """Apply activation layer for this module.

  Have formatted this module to be reasonably flexible when it comes to
  different activation layers. It can handle standard activations, or the
  scaled activations (eg. when using nfnets). This function is a wrapper to ensure
  hyperparams are set correctly when handling the different activation layers. If it
  is just a standard network, then, will just apply standard activation functions.
  For NF-Nets - we will apply the gamma adjusted/scaled activations.

  Parameters
  ----------
  inputs : tensor
      input to the normalization layers.
  norm_layer : Layer or None
      normalization layer to be applied.
  name : str
      string to name the layer

  Returns
  -------
      inputs through normalization layer, or None if norm layer is None.
  """
  if norm_layer is None:
    return inputs
  elif norm_layer is layers.BatchNormalization:
    return layers.BatchNormalization(epsilon=1.001e-5, name=name + 'bn')(inputs)
  elif norm_layer is FRN:
    return FRN(name=name + 'frn')(inputs)
  else:
    raise ValueError((f'Invalid/unsupported normalization layer specified.'
                      f'Got {norm_layer.__class__.__name__}'))


def ResNet(stack_fn,
           input_shape,
           output_shape,
           conv_layer=layers.Conv2D,
           activation_fn=tf.nn.relu,
           normalization=layers.BatchNormalization,
           use_bias=True,
           prior_loss=None,
           model_name='resnet',
           **kwargs):
  """Instantiate the ResNet, ResNetV2, and ResNeXt architecture.

    Largely taken from tensorflow.keras implementation
  (link in module docstring)
  and modified for what I need.

  Modifications include:
    + replacing batch normalisation with filter response normalisation
    + replacing relu with swish (now in tf >=2.9)

  Parameters
  ----------
  stack_fn : callable
      a function that returns output tensor for the
      stacked residual blocks.
  input_shape : tuple(ints)
      shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(224, 224, 3)` (with `channels_last` data format)
      or `(3, 224, 224)` (with `channels_first` data format).
      It should have exactly 3 inputs channels.
  output_shape : int
      number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
  conv_layer : keras.Layer
      layer that will perform the convolution ops.
  activation_fn : callable
      activation function to apply. This will be wrapped in the function
      `keras.layers.Activation` that will turn it into a callable layer.
  nomalization : keras.Layer
      Reference to the class reference of the normalization to use,
      (ie. normalization=keras.layers.BatchNorm)
       and NOT an initiated objexct,
      (ie. Don'T DO THIS! normalization=keras.layers.BatchNorm() )
  prior_arg : float or callable
      argiment to prior function
  prior_loss : float or callable
      loss function for prior
  use_bias : bool
      whether to use biases for convolutional layers or not
      (True for ResNet and ResNetV2, False for ResNeXt).
  model_name : string
        model name.
   **kwargs: For backwards compatibility only.

  Returns
  -------
      A `keras.Model` instance.
    """
  # get a loss for the bias params if needed
  bias_prior_loss = prior_loss if use_bias else None
  # Determine proper input shape
  img_input = layers.Input(shape=input_shape)
  # adding the first few layers of the model, to do some things like
  # zero padding etc.
  x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)),
                           name='conv1_pad')(img_input)
  # for imagenet, we are working with larger images and want to treat the a little
  # bit differently. For things like CIFAR-10 etc, we don't want to do such a large
  # conv op. with such a large stride, and we don't want to reduce the dimension
  # of our data that much with a pooling op.
  if output_shape == 1000:
    x = conv_layer(filters=64,
                   kernel_size=7,
                   strides=2,
                   kernel_regularizer=prior_loss,
                   use_bias=use_bias,
                   bias_regularizer=bias_prior_loss,
                   name='conv1_conv')(x)
  else:
    # use a much smaller kernel and stride
    x = conv_layer(filters=64,
                   kernel_size=3,
                   strides=1,
                   kernel_regularizer=prior_loss,
                   use_bias=use_bias,
                   bias_regularizer=bias_prior_loss,
                   name='conv1_conv')(x)
  x = apply_normalization(x, normalization, 'act_1_')
  x = activation_fn(x, name='norm1_')
  x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
  # again, with larger dim data from things such as imagenet, want to
  # reduce the size a bit so will do a pooling op.
  if output_shape == 1000:
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)
  # the stack of our resnet
  x = stack_fn(x)
  x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = layers.Dense(output_shape,
                   kernel_regularizer=prior_loss,
                   use_bias=use_bias,
                   bias_regularizer=bias_prior_loss,
                   name='predictions')(x)
  # Create model.
  model = Model(img_input, x, name=model_name)
  return model


def block1(x,
           filters,
           kernel_size=3,
           stride=1,
           conv_layer=layers.Conv2D,
           activation_fn=tf.nn.relu,
           normalization=layers.BatchNormalization,
           use_bias=True,
           prior_loss=None,
           conv_shortcut=True,
           name=None):
  """A residual block.

  Parameters
  ----------
  x : tensor
      input tensor.
  filters : integer
      filters of the bottleneck layer in a block.
  stride : int (default 2)
      stride of the first layer in the first block.
  conv_layer : keras.Layer
      layer that will perform the convolution ops.
  activation_fn : callable
      activation function to apply. This will be wrapped in the function
      `keras.layers.Activation` that will turn it into a callable layer.
  nomalization : keras.Layer
      Reference to the class reference of the normalization to use,
      (ie. normalization=keras.layers.BatchNorm)
       and NOT an initiated objexct,
      (ie. Don'T DO THIS! normalization=keras.layers.BatchNorm() )
  use_bias : bool
    whether to use biases for convolutional layers or not
    (True for ResNet and ResNetV2, False for ResNeXt).
  prior_loss : callable
      loss function based on the prior over the stack
  conv_shortcut : bool
      whether to use shortcut layer or not on input
  name : string
    stack label.


   Returns
   -------
      Output tensor for the residual block.
  """
  # get a loss for the bias params if needed
  bias_prior_loss = prior_loss if use_bias else None
  if conv_shortcut:
    shortcut = conv_layer(filters=4 * filters,
                          kernel_size=1,
                          strides=stride,
                          kernel_regularizer=prior_loss,
                          use_bias=use_bias,
                          bias_regularizer=bias_prior_loss,
                          name=name + '_0_conv')(x)
    shortcut = apply_normalization(shortcut, normalization, name + '_0_norm')
  else:
    shortcut = x
  x = conv_layer(filters=filters,
                 kernel_size=1,
                 strides=stride,
                 kernel_regularizer=prior_loss,
                 use_bias=use_bias,
                 bias_regularizer=bias_prior_loss,
                 name=name + '_1_conv')(x)
  x = apply_normalization(x, normalization, name + '_1_norm')
  x = activation_fn(x, name=name + '_1_')
  x = conv_layer(filters=filters,
                 kernel_size=kernel_size,
                 kernel_regularizer=prior_loss,
                 use_bias=use_bias,
                 bias_regularizer=bias_prior_loss,
                 padding='SAME',
                 name=name + '_2_conv')(x)
  x = apply_normalization(x, normalization, name + '_2_')
  x = activation_fn(x, name=name + '_2_')
  x = conv_layer(filters=4 * filters,
                 kernel_size=1,
                 kernel_regularizer=prior_loss,
                 use_bias=use_bias,
                 bias_regularizer=bias_prior_loss,
                 name=name + '_3_conv')(x)
  x = apply_normalization(x, normalization, name + '_4_')
  x = layers.Add(name=name + '_add')([shortcut, x])
  x = activation_fn(x, name=name + '_3_')
  return x


def block_basic(x,
                filters,
                kernel_size=3,
                stride=1,
                conv_layer=layers.Conv2D,
                activation_fn=tf.nn.relu,
                normalization=layers.BatchNormalization,
                use_bias=True,
                prior_loss=None,
                name=None):
  """A basic residual block, as used in resnet18.

  Parameters
  ----------
  x : tensor
      input tensor.
  filters : integer
      filters of the bottleneck layer in a block.
  stride : int (default 2)
      stride of the first layer in the first block.
  conv_layer : keras.Layer
      layer that will perform the convolution ops.
  activation_fn : callable
      activation function to apply. This will be wrapped in the function
      `keras.layers.Activation` that will turn it into a callable layer.
  nomalization : keras.Layer
      Reference to the class reference of the normalization to use,
      (ie. normalization=keras.layers.BatchNorm)
       and NOT an initiated objexct,
      (ie. Don'T DO THIS! normalization=keras.layers.BatchNorm() )
  use_bias : bool
    whether to use biases for convolutional layers or not
    (True for ResNet and ResNetV2, False for ResNeXt).
  prior_loss : callable
      loss function based on the prior over the stack
  name : string
    stack label.


   Returns
   -------
      Output tensor for the residual block.
  """
  # get a loss for the bias params if needed
  bias_prior_loss = prior_loss if use_bias else None
  shortcut = x
  x = conv_layer(filters=filters,
                 kernel_size=(3, 3),
                 strides=stride,
                 kernel_regularizer=prior_loss,
                 use_bias=use_bias,
                 padding='SAME',
                 bias_regularizer=bias_prior_loss,
                 name=name + '_1_conv')(x)
  x = apply_normalization(x, normalization, name + '_1_norm')
  x = activation_fn(x, name=name + '_1_')
  x = conv_layer(filters=filters,
                 kernel_size=kernel_size,
                 kernel_regularizer=prior_loss,
                 use_bias=use_bias,
                 bias_regularizer=bias_prior_loss,
                 padding='SAME',
                 name=name + '_2_conv')(x)
  x = apply_normalization(x, normalization, name + '_2_')
  if stride != 1:
    shortcut = conv_layer(filters=filters,
                          kernel_size=(1, 1),
                          strides=stride,
                          kernel_regularizer=prior_loss,
                          use_bias=use_bias,
                          bias_regularizer=bias_prior_loss,
                          name=name + '_shorcut_conv')(shortcut)
    shortcut = apply_normalization(shortcut, normalization, name + '_shortcut_')
  # add the shortcut now
  x = x + shortcut
  x = activation_fn(x, name=name + '_2_')
  return x


def stack_resnet50(x,
                   filters,
                   blocks,
                   conv_layer,
                   activation_fn,
                   normalization,
                   use_bias,
                   prior_loss,
                   stride1=2,
                   name=None):
  """Creates stacked residual blocks for resnet50.

  Parameters
  ----------
  x : tensor
      input tensor.
  filters : integer
      filters of the bottleneck layer in a block.
  blocks: integer
      blocks in the stacked blocks.
  conv_layer : keras.Layer
      layer that will perform the convolution ops.
  activation_fn : callable
      activation function to apply. This will be wrapped in the function
      `keras.layers.Activation` that will turn it into a callable layer.
  nomalization : keras.Layer
      Reference to the class reference of the normalization to use,
      (ie. normalization=keras.layers.BatchNorm)
       and NOT an initiated objexct,
      (ie. Don'T DO THIS! normalization=keras.layers.BatchNorm() )
  use_bias : bool
      whether to use biases for convolutional layers or not
      (True for ResNet and ResNetV2, False for ResNeXt).
  prior_loss : callable
      loss function based on the prior over the stack
  stride1 : int (default 2)
      stride of the first layer in the first block.
  name : string
      stack label.

  Returns
  -------
      Output tensor for the stacked blocks.
    """
  x = block1(x,
             filters,
             stride=stride1,
             conv_shortcut=True,
             conv_layer=conv_layer,
             activation_fn=activation_fn,
             normalization=normalization,
             use_bias=use_bias,
             prior_loss=prior_loss,
             name=name + '_block1')
  for i in range(2, blocks + 1):
    x = block1(x,
               filters,
               conv_shortcut=False,
               conv_layer=conv_layer,
               activation_fn=activation_fn,
               normalization=normalization,
               use_bias=use_bias,
               prior_loss=prior_loss,
               name=name + '_block' + str(i))
  return x


def stack_resnet18(x: tf.Tensor,
                   filters: int,
                   blocks: int,
                   conv_layer: tf.keras.layers.Layer,
                   activation_fn: Callable,
                   normalization: Callable,
                   use_bias: bool,
                   prior_loss: Callable,
                   stride1: int = 2,
                   name=None):
  """Creates stacked residual blocks for resnet18.

  Parameters
  ----------
  x : tensor
      input tensor.
  filters : integer
      filters of the bottleneck layer in a block.
  blocks: integer
      blocks in the stacked blocks.
  conv_layer : keras.Layer
      layer that will perform the convolution ops.
  activation_fn : callable
      activation function to apply. This will be wrapped in the function
      `keras.layers.Activation` that will turn it into a callable layer.
  nomalization : keras.Layer
      Reference to the class reference of the normalization to use,
      (ie. normalization=keras.layers.BatchNorm)
       and NOT an initiated objexct,
      (ie. Don'T DO THIS! normalization=keras.layers.BatchNorm() )
  use_bias : bool
      whether to use biases for convolutional layers or not
      (True for ResNet and ResNetV2, False for ResNeXt).
  prior_loss : callable
      loss function based on the prior over the stack
  stride1 : int (default 2)
      stride of the first layer in the first block.
  name : string
      stack label.

  Returns
  -------
      Output tensor for the stacked blocks.
    """
  x = block_basic(x,
                  filters,
                  stride=stride1,
                  conv_layer=conv_layer,
                  activation_fn=activation_fn,
                  normalization=normalization,
                  use_bias=use_bias,
                  prior_loss=prior_loss,
                  name=name + '_block1')
  for i in range(2, blocks + 1):
    x = block_basic(x,
                    filters,
                    stride=1,
                    conv_layer=conv_layer,
                    activation_fn=activation_fn,
                    normalization=normalization,
                    use_bias=use_bias,
                    prior_loss=prior_loss,
                    name=name + '_block' + str(i))
  return x


def ResNet50(input_shape,
             output_shape,
             conv_layer=layers.Conv2D,
             activation_fn=tf.nn.relu,
             normalization=layers.BatchNormalization,
             use_bias=True,
             prior_loss=None,
             pooling=None,
             **kwargs):
  """Instantiates the ResNet50 architecture.

  Generate Resnet50 architecture, with changes to make it more amenable to MCMC
  sampling. These changes include, making it able to support a prior function as
  an additional loss, changing activation to a smother, more we'll behaved
  activation such as the Swish activation, and replacing Batchnorm with Filter
  Response Normalisation.

  Parameters
  ----------
  input_shape : list[int, int, int]
      shape of input images.
  output_shape : int
      number of classes to classify.
  conv_layer : keras.Layer
      layer that will perform the convolution ops.
  activation_fn : callable
      function that will create and apply the relevent activation function
      for the given layer.
  nomalization : keras.Layer
      Reference to the class reference of the normalization to use,
      (ie. normalization=keras.layers.BatchNorm)
       and NOT an initiated objexct,
      (ie. Don'T DO THIS! normalization=keras.layers.BatchNorm() )
  use_bias : bool
      whether to use biases for convolutional layers or not
      (True for ResNet and ResNetV2, False for ResNeXt).
  prior_loss : callable
      function that will generate a neg log prob prior to be included as a loss
      for certain layers.
  pooling :
  **kwargs :

  Returns
  -------
      Keras Resnet50 Model
  """

  def stack_fn(x):
    """helper function to generate resnet50 stacks.

    Parameters
    ----------
    x : tensor
        input for the stack.

    Returns
    -------
        Output tensor for the stack.
    """
    x = stack_resnet50(x,
                       64,
                       3,
                       conv_layer=conv_layer,
                       activation_fn=activation_fn,
                       normalization=normalization,
                       use_bias=use_bias,
                       prior_loss=prior_loss,
                       stride1=1,
                       name='conv2')
    x = stack_resnet50(x,
                       128,
                       4,
                       conv_layer=conv_layer,
                       activation_fn=activation_fn,
                       normalization=normalization,
                       use_bias=use_bias,
                       prior_loss=prior_loss,
                       name='conv3')
    x = stack_resnet50(x,
                       256,
                       6,
                       conv_layer=conv_layer,
                       activation_fn=activation_fn,
                       normalization=normalization,
                       use_bias=use_bias,
                       prior_loss=prior_loss,
                       name='conv4')
    return stack_resnet50(x,
                          512,
                          3,
                          conv_layer=conv_layer,
                          activation_fn=activation_fn,
                          normalization=normalization,
                          use_bias=use_bias,
                          prior_loss=prior_loss,
                          name='conv5')

  return ResNet(stack_fn,
                input_shape,
                output_shape,
                conv_layer=conv_layer,
                activation_fn=activation_fn,
                normalization=normalization,
                use_bias=True,
                prior_loss=prior_loss,
                model_name='resnet50',
                **kwargs)


def ResNet18(input_shape: list[int],
             output_shape: list[int],
             conv_layer: keras.layers.Layer = layers.Conv2D,
             activation_fn: Callable = tf.nn.relu,
             normalization: Callable = layers.BatchNormalization,
             use_bias: bool = True,
             prior_loss: Callable = None,
             pooling: Optional[Callable] = None,
             **kwargs):
  """Instantiates the ResNet18 architecture.

  Generate Resnet18 architecture, with changes to make it more amenable to MCMC
  sampling. These changes include, making it able to support a prior function as
  an additional loss, changing activation to a smother, more we'll behaved
  activation such as the Swish activation, and replacing Batchnorm with Filter
  Response Normalisation.

  Parameters
  ----------
  input_shape : list[int, int, int]
      shape of input images.
  output_shape : int
      number of classes to classify.
  conv_layer : keras.Layer
      layer that will perform the convolution ops.
  activation_fn : callable
      function that will create and apply the relevent activation function
      for the given layer.
  nomalization : keras.Layer
      Reference to the class reference of the normalization to use,
      (ie. normalization=keras.layers.BatchNorm)
       and NOT an initiated objexct,
      (ie. Don'T DO THIS! normalization=keras.layers.BatchNorm() )
  use_bias : bool
      whether to use biases for convolutional layers or not
      (True for ResNet and ResNetV2, False for ResNeXt).
  prior_loss : callable
      function that will generate a neg log prob prior to be included as a loss
      for certain layers.
  pooling :
  **kwargs :

  Returns
  -------
      Keras Resnet50 Model
  """

  def stack_fn(x: tf.Tensor):
    """helper function to generate resnet50 stacks.

    Parameters
    ----------
    x : tensor
        input for the stack.

    Returns
    -------
        Output tensor for the stack.
    """
    x = stack_resnet18(x,
                       64,
                       2,
                       conv_layer=conv_layer,
                       activation_fn=activation_fn,
                       normalization=normalization,
                       use_bias=use_bias,
                       prior_loss=prior_loss,
                       stride1=1,
                       name='conv2')
    x = stack_resnet18(x,
                       128,
                       2,
                       conv_layer=conv_layer,
                       activation_fn=activation_fn,
                       normalization=normalization,
                       use_bias=use_bias,
                       prior_loss=prior_loss,
                       stride1=2,
                       name='conv3')
    x = stack_resnet18(x,
                       256,
                       2,
                       conv_layer=conv_layer,
                       activation_fn=activation_fn,
                       normalization=normalization,
                       use_bias=use_bias,
                       prior_loss=prior_loss,
                       stride1=2,
                       name='conv4')
    return stack_resnet18(x,
                          512,
                          2,
                          conv_layer=conv_layer,
                          activation_fn=activation_fn,
                          normalization=normalization,
                          use_bias=use_bias,
                          prior_loss=prior_loss,
                          stride1=2,
                          name='conv5')

  return ResNet(stack_fn,
                input_shape,
                output_shape,
                conv_layer=conv_layer,
                activation_fn=activation_fn,
                normalization=normalization,
                use_bias=True,
                prior_loss=prior_loss,
                model_name='resnet18',
                **kwargs)


def make_resnet_nf_swish(input_shape,
                         output_shape,
                         prior_loss=None,
                         pooling=None):

  def _act_fn(inputs, name):
    return activation_by_name_with_gamma(inputs, activation='swish', name=name)

  return ResNet50(input_shape,
                  output_shape,
                  conv_layer=nfnets.ScaledStandardizedConv2D,
                  activation_fn=_act_fn,
                  normalization=None,
                  use_bias=True,
                  prior_loss=prior_loss,
                  pooling=None)


def make_resnet_bn_swish(input_shape,
                         output_shape,
                         prior_loss=None,
                         pooling=None):

  def _act_fn(inputs, name):
    return activation_by_name(inputs, activation='swish', name=name)

  return ResNet50(input_shape,
                  output_shape,
                  conv_layer=layers.Conv2D,
                  activation_fn=_act_fn,
                  normalization=layers.BatchNormalization,
                  use_bias=True,
                  prior_loss=prior_loss,
                  pooling=None)


def make_resnet18_bn_swish(input_shape,
                           output_shape,
                           prior_loss=None,
                           pooling=None):

  def _act_fn(inputs, name):
    return activation_by_name(inputs, activation='swish', name=name)

  return ResNet18(input_shape,
                  output_shape,
                  conv_layer=layers.Conv2D,
                  activation_fn=_act_fn,
                  normalization=layers.BatchNormalization,
                  use_bias=True,
                  prior_loss=prior_loss,
                  pooling=None)
