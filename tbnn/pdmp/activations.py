"""
extra activation functions
# taken from
# https://github.com/leondgarse/keras_cv_attention_models/blob/0c3e6719eb06935ca0eb01974269033dc07e8642/keras_cv_attention_models/common_layers.py
"""
import tensorflow as tf
from tensorflow import keras

# @tf.keras.utils.register_keras_serializable(package="kecamCommon")
def hard_swish(inputs):
  """`out = xx * relu6(xx + 3) / 6`, arxiv: https://arxiv.org/abs/1905.02244"""
  return inputs * tf.nn.relu6(inputs + 3) / 6


# @tf.keras.utils.register_keras_serializable(package="kecamCommon")
def hard_sigmoid_torch(inputs):
  """https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html
    toch.nn.Hardsigmoid: 0 if x <= −3 else (1 if x >= 3 else x / 6 + 1/2)
    keras.activations.hard_sigmoid: 0 if x <= −2.5 else (1 if x >= 2.5 else x / 5 + 1/2) -> tf.clip_by_value(inputs / 5 + 0.5, 0, 1)
    """
  return tf.clip_by_value(inputs / 6 + 0.5, 0, 1)


# @tf.keras.utils.register_keras_serializable(package="kecamCommon")
def mish(inputs):
  """Mish: A Self Regularized Non-Monotonic Neural Activation Function.
    Paper: [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)
    Copied from https://github.com/tensorflow/addons/blob/master/tensorflow_addons/activations/mish.py
    """
  return inputs * tf.math.tanh(tf.math.softplus(inputs))


# @tf.keras.utils.register_keras_serializable(package="kecamCommon")
def phish(inputs):
  """Phish is defined as f(x) = xTanH(GELU(x)) with no discontinuities in the f(x) derivative.
    Paper: https://www.techrxiv.org/articles/preprint/Phish_A_Novel_Hyper-Optimizable_Activation_Function/17283824
    """
  return inputs * tf.math.tanh(tf.nn.gelu(inputs))


def activation_by_name(inputs, activation="relu", name=None):
  """Typical Activation layer added hard_swish and prelu."""
  if activation is None:
    return inputs

  layer_name = name
  # if the activation is supplied as a string
  if isinstance(activation, str):
    if activation == "hard_swish":
      return keras.layers.Activation(activation=hard_swish,
                                     name=layer_name)(inputs)
    elif activation == "mish":
      return keras.layers.Activation(activation=mish, name=layer_name)(inputs)
    elif activation == "phish":
      return keras.layers.Activation(activation=phish, name=layer_name)(inputs)
    elif activation.lower() == "prelu":
      shared_axes = list(range(1, len(inputs.shape)))
      shared_axes.pop(-1 if K.image_data_format() == "channels_last" else 0)
      # print(f"{shared_axes = }")
      return keras.layers.PReLU(shared_axes=shared_axes,
                                alpha_initializer=tf.initializers.Constant(0.25),
                                name=layer_name)(inputs)
    elif activation.lower().startswith("gelu/app"):
      # gelu/approximate
      return tf.nn.gelu(inputs, approximate=True, name=layer_name)
    elif activation.lower() == ("hard_sigmoid_torch"):
      return keras.layers.Activation(activation=hard_sigmoid_torch,
                                     name=layer_name)(inputs)
    else:
      return keras.layers.Activation(activation=activation,
                                     name=layer_name)(inputs)
  # otherwise is supplied a function
  else:
    return keras.layers.Activation(activation=activation,
                                   name=layer_name)(inputs)
