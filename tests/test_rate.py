import numpy as np
import tensorflow as tf
from tensorflow.test import TestCase
import tensorflow_probability as tfp

from tbnn.pdmp.model import build_network,get_model_state, set_model_params
from tbnn.utils import utils

class TestRate(TestCase):
  """Test the calculation of the rate function
  """

  def test_rate_simple_gaussian(self):
    """compare rate found through method with analytical rate
    for simple gaussian"""

    def grad_gaussian(x, y, w_start, velocity, time, prior_mean,
                      prior_var, likelihood_var):
      w = w_start + velocity * time
      prior_grad = (w - prior_mean) / prior_var
      likelihood_grad = -(x * (y - x * w))/likelihood_var
      return prior_grad + likelihood_grad

    def rate_no_max(grad, velocity):
      """won't take the max, just want to know my way is working"""
      return velocity * grad

    # load in super simple dataset with one point
    data = utils.load_dataset('test_b', split=None)
    # now lets create a simple model
    config = './test_rate_config.json'
    model = build_network(config, data.x_train, data.dimension_dict)
    # lets print a quick summary of it
    print(model.summary())
    # now lets create the likelihood and prior
    prior_mean = 0.0
    prior_var = 1.0
    likelihood_var = 1.0
    prior_fn = tfp.distributions.Normal(loc=prior_mean, scale=prior_var)
    # now create joint log prob
    def neg_joint_log_prob(model, prior, x, y, weights):
      lp = tf.reduce_sum(prior_fn.log_prob(weights))
      # set the model params
      model.layers[0].kernel = weights
      pred = model(x)
      # create a distribution based of thiss
      likelihood_fn = tfp.distributions.Normal(pred, scale=likelihood_var)
      # now add the log likelihood
      lp += tf.reduce_sum(likelihood_fn.log_prob(y))
      # now take negative of it
      return -1.0 * lp

    # lets find the grads now and make sure it all works ok
    velocity = np.array(0.5)
    time = 0.1
    weights = np.array(1.0)
    weights_tf = weights + velocity * time
    weights_tf = tf.Variable(weights_tf.reshape(model.layers[0].kernel.shape).astype(np.float32))
    print('weights_tf = {}'.format(weights_tf))
    print('type(weights_tf) = {}'.format(type(weights_tf)))
    with tf.GradientTape() as tape:
      neg_lp = neg_joint_log_prob(model, prior_fn, data.x_train,
                                  data.y_train, weights_tf)
    grad = tape.gradient(neg_lp, weights_tf)
    print('neg_lp = {}'.format(neg_lp))
    # now lets see how it compares
    print('grad = {}'.format(grad))

    # now compare that with the value found analytically
    grad_analytical = grad_gaussian(data.x_train, data.y_train,
                                    weights, velocity, time,
                                    prior_mean, prior_var,
                                    likelihood_var)
    print('grad_analytical = {}'.format(grad_analytical))


    def neg_joint_log_prob_no_sum(model, prior, weights):
      def fn(x, y):
        #x, y = arg
        x = tf.expand_dims(x, 0)
        y = tf.expand_dims(y, 0)
        with tf.GradientTape() as tape:
          lp = tf.reduce_sum(prior_fn.log_prob(weights))
          # set the model params
          model.layers[0].kernel = weights
          pred = model(x)
          # create a distribution based of thiss
          likelihood_fn = tfp.distributions.Normal(pred, scale=likelihood_var)
          # now add the log likelihood
          lp += likelihood_fn.log_prob(y)
          # now take negative of it
          lp = -1.0 * lp
        return lp, tape.gradient(lp, weights)
      return fn


    x = np.array([data.x_train, data.x_train]).reshape(-1, 1)
    y = np.array([data.y_train, data.y_train]).reshape(-1, 1)
    grad_fn = neg_joint_log_prob_no_sum(model, prior_fn, weights_tf)
    #neg_lp, grads = tf.vectorized_map(grad_fn, (x, y))
    neg_lp, grads = tf.scan(grad_fn, (x, y))
    print('neg_lp = {}'.format(neg_lp))
    # now lets see how it compares
    print('grads = {}'.format(grads))


  def test_hull_concave(self):
    # create the model



if __name__ == "__main__":
  tf.test.main()
