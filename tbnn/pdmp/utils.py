"""
Implements a few functions that will be used for PDMP methods

#### References
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

import collections


def compute_l2_norm(grads_target_log_prob):
  """ Computes the L2 norm of the gradient of the target

  Function exists to compute norms for the TFP format,
  where all the variables are stored in a list. The built-in
  tf.norm function expects the input to be a single tensor,
  not a list of tensors, so we need to compute it manually.

  Args:
    grads_target_log_prob (list(array)):
      List of arrays for the gradient of each variable.

  Returns:
    L2 norm of the gradient
  """
  # grads_target_log_prob is a list of arrays, so need to compute the
  # norm in multiple steps. Will first element-wise square each array in list
  grads_square = [tf.square(x) for x in grads_target_log_prob]
  # now get the sum over all the variables
  grads_square_sum = sum_list(grads_square)
  # now take sqrt to find the L2 Norm
  grads_norm = tf.math.sqrt(grads_square_sum)
  return grads_square_sum#grads_norm


def compute_dot_prod(input_a, input_b):
  """ Compute dot product between two vectors.

  Args:
    grads_target_log_prob (list(array)):
      List of arrays for the gradient of each variable.
    velocity (list(array)):
      List of arrays for the velocity of each variable.

   Returns:
     dot product of the two
  """
  # to compute the dot product, will perform element-wise
  # multiplication, and then sum everything
  element_wise_prod = [tf.multiply(g, v) for g,v in
                       zip(input_a, input_b)]
  #                     for i in range(len(velocity))]
  #element_wise_prod = [tf.multiply(grads_target_log_prob[i], velocity[i])
  #                     for i in range(len(velocity))]
  # now sum it all together
  dot_prod = sum_list(element_wise_prod)
  return dot_prod


def sum_list(array_list):
  """ Will find the total sum of all array items in a list

  This helper is here because TFP operates over lists of variables,
  and for this kernel we need to find the total sum over all
  elements many times.
  This method basically just calls tf.reduce_sum multiple times:
  once for each array item in the list, and then a second time to sum
  the individual summations of each item found in the previous step.

  Args:
    array_list (list(array)):
      list of arrays which contain the variables we are ineterested in

  Returns:
    Scalar containing the total sum

  Raises:
    ValueError: if input is not a list of tf arrays
  """
  if not mcmc_util.is_list_like(array_list):
    raise ValueError('Expected input arg to be a list of tf variables')
  # find the total sum over each element
  item_sum = [tf.reduce_sum(x) for x in array_list]
  # Now can sum up all the elements above, since each dimension is the same
  # (each element is a scalar, so tensorflow can handle it)
  total_sum = tf.reduce_sum(item_sum)
  return total_sum

def check_make_dir(dir_path, create_parent=False):
  """checks if directory exists, and if it doesn't will make it

  If the directory already exists, this function will just exit.
  If the supplied directory doesn't exist, it will make the directory if
  the parent directory path is valid (already exists). If the parent directory
  doesn't exist, unless the `create_parent` arg is set to True, this function
  will raise an exception. If `create_parent` is True, it will make the complete
  directory structure.

  Args:
    dir_path (str):
      path to directory to be checked/made
    create_parent (bool):
      whether we should force making the parent directory structure
      (if it doesn't already exist)

    Returns:
      NA

    Raises:
      `ValueError` if the parent dir doesn't exist and `create_parent` is False
  """
  # see if directory already exists or not. If it does exist we don't need
  # to do anything
  if not (os.path.isdir(dir_path)):
    # see if parent directory exists
    # first going to make sure there isn't a trailing
    # slash (/) on the path so that we can make sure os.path.split
    # will work properly when splitting base dir from parent dir.
    dir_path_strip = dir_path.rstrip('/')
    dir_parent, dir_name = os.path.split(dir_path_strip)
    if not os.path.isdir(dir_parent):
      if not(create_parent):
        raise IOError(("Can\'t create directory {}. Parent doesn\'t exist"
                       "and create_parent = False.'").format(dir_path))
      # otherwise will make the parent directory structure as well
      print("Parent dir {} doesn\'t exist, making now".format(parent_dir))
      os.makedirs(dir_parent)
    # if made it here, need to make the final directory
    print('Making dir {}'.format(dir_path_strip))
    os.mkdir(dir_path_strip)
