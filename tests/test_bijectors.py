import numpy as np
import tensorflow as tf
from tensorflow.test import TestCase
import tensorflow_probability as tfp

from tbnn.pdmp.model import build_network, get_model_state, set_model_params
from tbnn.utils import utils

import numpy as np
import tensorflow as tf
from tensorflow.test import TestCase
import tensorflow_probability as tfp

from tbnn.pdmp.model import build_network, get_model_state, set_model_params
from tbnn.utils import utils

import matplotlib.pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors

if __name__ == '__main__':

  concentration = tf.constant(3.0)
  scale = tf.constant(2.0)
  base_dist = tfd.InverseGamma(concentration, scale)
  #base_dist = tfd.HalfCauchy(0.0, 1.0)

  transformed_dist = tfd.TransformedDistribution(base_dist, tfb.Log())

  # now create a wrapper for this
  def log_prob(x):
    with tf.GradientTape() as tape:
      print('x = {}'.format(x))
      tape.watch(x)
      lp = base_dist.log_prob(x)

    grads = tape.gradient(lp, x)
    return lp, grads

  def log_prob_transformed(x):
    with tf.GradientTape() as tape:
      tape.watch(x)
      lp = transformed_dist.log_prob(x)

    grads = tape.gradient(lp, x)
    return lp, grads

  data = tf.convert_to_tensor(np.linspace(0.1, 1.0, 10).astype(np.float32))
  print(base_dist)
  base_grad_list = []
  base_lp_list = []
  t_grad_list = []
  t_lp_list = []
  for i in range(tf.size(data)):
    lp_base, grad_base = log_prob(data[i])
    base_lp_list.append(lp_base)
    base_grad_list.append(grad_base)
    lp_t, grad_t = log_prob_transformed(tf.math.log(data[i]))
    t_lp_list.append(lp_t)
    t_grad_list.append(grad_t)

  print(base_grad_list)
  print(t_grad_list)

  print(base_lp_list)
  print(t_lp_list)

  analytic = -concentration + scale * tf.math.exp(-tf.math.log(data))
  print(analytic)
  plt.figure()
  plt.plot(tf.math.log(data), t_grad_list)
  plt.plot(tf.math.log(data), analytic)
  plt.savefig('bijector_inv_gamma.png')
