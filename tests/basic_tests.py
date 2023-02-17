import numpy as np
import tensorflow as tf
from tensorflow.test import TestCase
#import the helper function that will automatically select
#the type of prior/posterior for our distribution
#will also import all possible model classes
from tbnn.network import network
#import some helper functions
from tbnn.utils import utils, display, summarise

from tbnn.network.factorised_gaussian_layer import FactorisedGaussianLayer

import argparse
import gzip
import pickle
import random
import os
import sys
import time

tfe = tf.contrib.eager

#setting fraction of GPU usage
#tf.enable_eager_execution()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
#setting seed
#np.random.seed(1)
#tf.set_random_seed(1)


def testing_train_epoch(sess, optimize, batch_num):
  """similar to train epoch from normal training script

  The function differes in that metadata and  summarising
  functionalities have been removed
  """
  #train on this shuffled EPOCH
  current_batch = 1
  while True:
    try:
      _ = sess.run(optimize, feed_dict={batch_num: np.float(current_batch)})
      current_batch += 1
    # if we are done processing all current batches in epoch
    except tf.errors.OutOfRangeError:
      break


class TestKL(TestCase):
  """Test the KL(q(w|theta) || p(w)) calculations

  Have setup for computing analytically when possible, as it should be
  much faster since it only needs to be calculated once (though the
  expected log-likelihood still requires MC integration).
  Will be analytical solution when prior and post are both Gaussian.

  If it isn't Gaussian, will use MC integration as well.
  This class tests that:
    1. mechanism for analytic calculation is used correctly
      a. only computes KL(q | p) once per batch
      b. recomputes the KL term for the next batch correctly
    2. That the MC integration method is properly called when required.
  """

  def test_kl_mechanism_analytic_once(self):
    """ Tests 1a. listed above.
    want to test my mechanism for computing the KL between
    approx. post and prior is correct"""

    def eager_test_kl_mechanism_analytic_once():
      # setup mock data
      x = np.random.rand(10, 1).astype(np.float32)
      # mock configuration data
      dense_param = {'use_bias': True, 'out_std': 1}
      # mock prior parameters
      prior_param = {
          "prior_input": "uninformative",
          "prior_type": "gaussian",
          "prior_ratio": 1.0,
          "out_std": 1.0
      }
      config_data = {
          'dense_param': dense_param,
          'activation': 'Linear',
          'name': 'test_kl_mechanism',
          'type': 'dense',
          'weight_init': 'orthogonal',
          'prior_param': prior_param
      }
      # 1-D Input signal, 10-D output
      dims = [1, 10]
      # now create the layer
      factorised_gaussian = FactorisedGaussianLayer(config_data, dims, 1,
                                                    {})  # empty prior dict
      # number of forward passes we will do during this test
      num_fwd = 10
      # creating a numpy array to make sure that all of the KL terms
      # are the same for each forward pass
      kl_array = np.zeros(num_fwd)
      # now test it
      init = tf.global_variables_initializer()

      for i in range(0, num_fwd):
        _ = factorised_gaussian.layer_forward_prop(x)
        kl_array[i] = factorised_gaussian.kl_qp
        # now manually change the prior
        # if all is working properly, the KL divergence should
        # not change, as it should only have been calculated once
        factorised_gaussian.weight_loc_prior = factorised_gaussian.weight_loc_prior + 1
      return (kl_array)

    # call the above function in eager mode
    kl_analytic_once = tfe.py_func(eager_test_kl_mechanism_analytic_once, [],
                                   tf.float32)
    with tf.Session() as sess:
      kl_array = sess.run(kl_analytic_once)
    # done the forward passes, check they are all equal
    self.assertTrue((kl_array == kl_array[0]).all())

  def test_kl_mechanism_analytic_resets(self):
    """ Tests 1b. listed above.
    Want to make sure that the flag that tells us that we only
    need to compute the KL term once per batch (when can solve
    analytically) resets correctly and allows us to calculate
    for the next batch.
    """

    # some testing params
    # will split data into 2 mini batches, and run two epochs
    num_epochs = 2
    num_batches = 2

    # INITIALISING
    # Simplified setup copied from bin/train
    # DATA
    # setup mock data
    dataset = utils.load_dataset('test_a', {})  # no train test split needed
    training_data = tf.data.Dataset.from_tensor_slices(
        (dataset.x_train, dataset.y_train))
    # shuffle the training data set, repeat and batch it
    training_data = training_data.shuffle(dataset.x_train.shape[0]).batch(
        dataset.num_train)
    # making initialisers for iterator
    data_shape = utils.data_shape(dataset.x_train, dataset.y_train)
    iterator = tf.data.Iterator.from_structure(training_data.output_types,
                                               data_shape)
    training_init = iterator.make_initializer(training_data)
    inputs, labels = iterator.get_next()

    # define BNN type
    bnn = network.get_tbnn('kl.json', dataset.dimension_dict)
    # build the graph
    outputs = bnn.build_model(inputs)
    batch_num = tf.placeholder(tf.float32, name="batch_num")
    global_step = tf.Variable(0, trainable=False)
    #create cost function, as specified by config file
    cost = bnn.build_cost(labels, outputs, batch_num=batch_num, n_samples=1)
    # defining optimiser
    optimize = bnn.build_optimiser(cost, global_step)
    # now setup an array that will save the KL terms from each layer
    # across the batches. Will be of dimension:
    # [num_layers, num_epochs * num_batches]
    kl_array = np.zeros([bnn.num_layers, num_epochs * num_batches])
    #initialise all variables
    init = tf.global_variables_initializer()
    # now lets simulate the training
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      sess.run(init)
      for epoch in range(0, num_epochs):
        # initialise iterator for this epoch
        sess.run(training_init)
        # train this full epoch
        current_batch = 1
        while True:
          try:
            _ = sess.run(optimize,
                         feed_dict={batch_num: np.float(current_batch)})
            # now get the kl term from the model at each layer
            for jj in range(0, bnn.num_layers):
              # am subtracting one because the current batch is not zero based
              kl_array[jj, epoch * num_epochs + current_batch -
                       1] = (bnn.layer[jj].kl_qp.eval())
            # increment the batch counter
            current_batch += 1
            # if we are done processing all current batches in epoch
          except tf.errors.OutOfRangeError:
            break

    # now need to check that they all don't equal the same
    # as we should have a new KL term for each layer across
    # each batch from each epoch
    kl_batch_different = np.array([[(kl_array[0, 0] != kl_array[0, 1:]).all()],
                                   [(kl_array[1, 0] != kl_array[1, 1:]).all()]])
    self.assertTrue(kl_batch_different.all())


if __name__ == "__main__":
  tf.test.main()
