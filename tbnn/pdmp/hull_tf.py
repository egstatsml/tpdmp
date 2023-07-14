import matplotlib.pyplot as plt
import random
import string

import tensorflow as tf
import numpy as np
import time
import tensorflow_probability as tfp

# import logging
# logging.basicConfig(filename='/home/ethan/code/tbnn/bin/log.log', filemode='w',
#                     format='%(name)s - %(levelname)s - %(message)s',
#                     level=logging.DEBUG)

EPSILON64 = tf.cast(2.22044604925e-16, tf.float64)

def sample_poisson_thinning(hull_samples, time):
  with tf.name_scope('sample_poisson_thinning'):
    #print('hull_samples = {}'.format(hull_samples))
    #print('time = {}'.format(np.sort(time)))
    S, fS = sort_time_target(time, hull_samples)
    S = tf.cast(tf.convert_to_tensor(S), tf.float64)
    fS = tf.cast(tf.convert_to_tensor(fS), tf.float64)
    # fS = 1.25 * fS# tf.cast(tf.convert_to_tensor(fS), tf.float64)

    # logging.debug('starting new sample/n')
    # logging.debug('hull samples = {}'.format(hull_samples))
    # logging.debug('time samples = {}'.format(time))
    #print('S = {}'.format(S))
    #print('fS = {}'.format(fS))
    m_array, b_array, left_array, right_array = compute_hulls(
      S, fS, domain=[0.0, 10.0])
    # print('m arary shape = {}'.format(m_array))
    # now get the ranges for the individual segments
    (integrated_constant_array,
     integrated_range_lower,
     integrated_range_upper,
     inverse_integrated_domain_lower,
     inverse_integrated_domain_upper,
     m_tensor, b_tensor,
     left_tensor, right_tensor) = compute_domain_range(m_array, b_array,
                                                       left_array, right_array)

    force_refresh, sample_time, hull_eval = sample_poisson(
      integrated_constant_array,
      integrated_range_lower,
      integrated_range_upper,
      inverse_integrated_domain_lower,
      inverse_integrated_domain_upper,
      m_tensor, b_tensor,
      left_tensor, right_tensor)

    #hull = Hull(m_array, b_array, left_array, right_array)
    #sample_time = hull.sample_poisson()
    # find the value of the hull at this location
    #sample_hull = hull.eval_hull(sample_time)
    # now lets sample it
    # def random_string_generator(str_size, allowed_chars=string.ascii_letters):
    #   return ''.join(random.choice(allowed_chars) for x in range(str_size))
    # plot_str = random_string_generator(12)
    # plt.figure()
    # #plt.plot(S_hi, fS_hi, c='b', alpha=0.5, linewidth=3, label='target')
    # #plt.scatter(S, fS, alpha=0.25, c='b', label='samples', s=150)
    # for node in hull.hull_list[:-1]:
    #   plt.plot([node.left, node.right],
    #            [node.left * node.m + node.b, node.right * node.m + node.b],
    #            alpha=1.0, c='r', linestyle='--')
    #   plt.scatter(np.array([node.left, node.right]),
    #               np.array([node.left * node.m + node.b, node.right * node.m + node.b]),
    #               alpha=0.5, c='r')
    # plt.plot([hull.hull_list[-1].left, hull.hull_list[-1].right],
    #          [hull.hull_list[-1].left * hull.hull_list[-1].m + hull.hull_list[-1].b,
    #           hull.hull_list[-1].right * hull.hull_list[-1].m + hull.hull_list[-1].b],
    #          c='r', alpha=1.0, linestyle='--', label='envelope')
    # # plt.scatter(np.array([hull.hull_list[-1].right]),
    # #             np.array([hull.hull_list[-1].right * hull.hull_list[-1].m + hull.hull_list[-1].b]),
    # #             c='r', alpha=0.5)
    # plt.legend(loc=0)
    # plt.savefig('adaptive_test/{}_hull.png'.format(plot_str))
    # plt.clf()
    # try:
    #   time, integrated = hull.eval_integrated()
    #   inv_time, inv_integrated = hull.eval_inverse_integrated(time=integrated)
    #   plt.subplot(211)
    #   plt.plot(inv_time, inv_integrated, label='inverse')
    #   plt.subplot(212)
    #   plt.plot(time, integrated, label='integrated')
    #   plt.savefig('adaptive_test/{}.png'.format(plot_str))
    #   plt.close()
    # except:
    #   pass
    # print('this is sample {}'.format(plot_str))
    # print('sampled time = {}, hull_eval = {}'.format(sample_time, hull_eval))
    # print(sample_time)
    return tf.cast(tf.reshape(force_refresh, ()), tf.bool), tf.cast(tf.reshape(sample_time, ()), tf.float32), tf.cast(tf.reshape(hull_eval, ()), tf.float32)


def sort_time_target(time, hull_samples):
  """Sort the time and hull samples"""
  with tf.name_scope('sort_time_target'):
    # get sort indivies based on time
    _, indicies = tf.math.top_k(time, k=tf.size(time))
    # now apply them to sort, and also flip as it the indicies are currently
    # in decending order, I want accending
    S = tf.experimental.numpy.flip(tf.gather(time, indicies))
    fS = tf.experimental.numpy.flip(tf.gather(hull_samples, indicies))
    return S, fS


def compute_domain_range(m_array, b_array, left_array, right_array,
                         upper_limit=100000.0):
  with tf.name_scope('compute_domain_range'):
    upper_limit = tf.convert_to_tensor(upper_limit, tf.float64)
    # create the tensor arrays we need
    array_size = m_array.size()
    index = tf.reshape(tf.constant(0, dtype=tf.int32), ())
    # print('index = {}'.format(index))
    # print('index shape = {}'.format(index.shape))
    integrated_range_lower = tf.TensorArray(
      tf.float64, size=0, dynamic_size=True, clear_after_read=False)
    integrated_range_upper = tf.TensorArray(
      tf.float64, size=0, dynamic_size=True, clear_after_read=False)
    inverse_integrated_domain_lower = tf.TensorArray(
      tf.float64, size=0, dynamic_size=True, clear_after_read=False)
    inverse_integrated_domain_upper = tf.TensorArray(
      tf.float64, size=0, dynamic_size=True, clear_after_read=False)
    integrated_constant_array = tf.TensorArray(
      tf.float64, size=0, dynamic_size=True, clear_after_read=False)
    # initialise the constant term
    integrated_constant = tf.convert_to_tensor(0.0, dtype=tf.float64)
    # integrated_constant, _, _ = compute_domain_range_single_hull(
    #   m_array.read(index),
    #   b_array.read(index),
    #   left_array.read(index),
    #   right_array.read(index))
    #integrated_constant = tf.convert_to_tensor(0.0, dtype=tf.float32)
    # now create the condition for the loop, and a wrapper for the loop itself
    # we have the compute_domain_range_loop_fn below, but it requires the
    # left, right, m and b arrays which aren't updated and not used by the
    # condition function.
    # the other arrays aren't used by the condition function either, but are
    # updated in the loop fn, so need to parse them to the condition function as
    # the number or args to each function must be the same within the
    # tf.while_loop construct. Therefore am going to abstract them away from the
    # cond fn with a wrapper for the loop fn
    #
    # the _w suffix is to make it easier for me to differentiate variable names
    # and make it more explicit that the arguments parsed to the
    # compute_domain_range_loop_fn are the input arguments here, not those
    # within the scope of the compute_domain_range function itself.
    def cond(index_w, *args):
      return tf.less(index_w, array_size)

    def loop_wrapper(index_w, integrated_constant_w,
                     integrated_constant_array_w,
                     integrated_range_lower_w,
                     integrated_range_upper_w,
                     inverse_integrated_domain_lower_w,
                     inverse_integrated_domain_upper_w):
      return compute_domain_range_loop_fn(index_w, integrated_constant_w,
                                          integrated_constant_array_w,
                                          integrated_range_lower_w,
                                          integrated_range_upper_w,
                                          inverse_integrated_domain_lower_w,
                                          inverse_integrated_domain_upper_w,
                                          m_array, b_array,
                                          left_array, right_array)
    loop_vars = (index, integrated_constant,
                 integrated_constant_array,
                 integrated_range_lower,
                 integrated_range_upper,
                 inverse_integrated_domain_lower,
                 inverse_integrated_domain_upper)
    # now run the while loop
    (index, integrated_constant,
     integrated_constant_array,
     integrated_range_lower,
     integrated_range_upper,
     inverse_integrated_domain_lower,
     inverse_integrated_domain_upper) = tf.while_loop(
       cond, loop_wrapper, loop_vars=(index, integrated_constant,
                                      integrated_constant_array,
                                      integrated_range_lower,
                                      integrated_range_upper,
                                      inverse_integrated_domain_lower,
                                      inverse_integrated_domain_upper))
    # add a new element to the end, that will serve as the extrapolation point
    # m_array = m_array.write(index, tf.reshape(tf.convert_to_tensor(0.1, dtype=tf.float64), (1,)))
    m_array = m_array.write(index, tf.abs(m_array.read(index - 1) + EPSILON64))
    b_array = b_array.write(index, b_array.read(index - 1))
    # make the left new bound the value of the upper bound of the region
    left_array = left_array.write(index, right_array.read(index - 1))
    inverse_integrated_domain_lower =  inverse_integrated_domain_lower.write(
      # index, inverse_integrated_domain_upper.read(index - 1))
      index, integrated_constant)
    inverse_integrated_domain_upper =  inverse_integrated_domain_upper.write(
      index, tf.reshape(tf.convert_to_tensor(100000.0, dtype=tf.float64), ()))
    integrated_lower = evaluate_integrated(right_array.read(index-1),
                                           m_array.read(index-1),
                                           b_array.read(index-1))
    integrated_constant_array =  integrated_constant_array.write(
      index, tf.reshape(tf.convert_to_tensor(
        # integrated_constant - integrated_range_lower.read(index - 1), dtype=tf.float64), ()))
        integrated_constant - integrated_lower, dtype=tf.float64), ()))
    # print('integrated_constant_array = {}'.format(integrated_constant_array))
    #
    # logging.debug('integrated_array = {}'.format(right_array))
    # logging.debug('int const {}'.format(integrated_constant_array.stack()))
    # logging.debug('int low {}'.format(integrated_range_lower.stack()))
    # logging.debug('int  up {} '.format(integrated_range_upper.stack()))
    # logging.debug('int inv  low {}'.format(inverse_integrated_domain_lower.stack()))
    # logging.debug('int inv up {}'.format(inverse_integrated_domain_upper.stack()))
    # logging.debug('m {}'.format(m_array.stack()))
    # logging.debug('b {}'.format(b_array.stack()))
    # logging.debug('l {}'.format(left_array.stack()))
    # logging.debug('r {}'.format(right_array.stack()))
    # now stack and return
    return (integrated_constant_array.stack(),
            integrated_range_lower.stack(),
            integrated_range_upper.stack(),
            inverse_integrated_domain_lower.stack(),
            inverse_integrated_domain_upper.stack(),
            m_array.stack(), b_array.stack(),
            left_array.stack(), right_array.stack())


# now iterate over them them
def compute_domain_range_loop_fn(index, integrated_constant,
                                 integrated_constant_array,
                                 integrated_range_lower,
                                 integrated_range_upper,
                                 inverse_integrated_domain_lower,
                                 inverse_integrated_domain_upper,
                                 m_array, b_array,
                                 left_array, right_array):
  with tf.name_scope('compute_domain_range_loop_fn'):
    # now compute the range for this segment in the integrated rate
    integrated_lower, integrated_upper, definite_integral = compute_domain_range_single_hull(
      m_array.read(index),
      b_array.read(index),
      left_array.read(index),
      right_array.read(index))
    # now can save this all to the tensor arrays
    integrated_range_lower = integrated_range_lower.write(
      index, integrated_constant)
    integrated_range_upper = integrated_range_upper.write(
      index, definite_integral + integrated_constant)
    # now lets get the domain of the inverse of the integral, which will be
    # the range of the integral
    # TODO: Review this choice
    # I might get rid of this in the future, as it stores the exact same thing,
    # but it currently makes the code more explicit and easier to read, and that
    # is a massive plus so I am going to keep it for now
    inverse_integrated_domain_lower = inverse_integrated_domain_lower.write(
      index, integrated_constant)
    inverse_integrated_domain_upper = inverse_integrated_domain_upper.write(
      index, definite_integral + integrated_constant)
    #integrated_constant = integrated_constant - integrated_lower
    # update the integrated constant term
    # print('integrated_constant = {}'.format(integrated_constant))
    # update the integrated constant array item
    integrated_constant_array = integrated_constant_array.write(
      # index, integrated_constant + definite_integral)
      index, integrated_constant - integrated_lower)
    # now add increment the constant term, which is the difference
    # established by the definite integral of this hull

    integrated_constant = integrated_constant + definite_integral
    # print('definite_integral = {}'.format(definite_integral))
    # increment the index for the loop counter
    index = index + 1
    return (index, integrated_constant,
            integrated_constant_array,
            integrated_range_lower,
            integrated_range_upper,
            inverse_integrated_domain_lower,
            inverse_integrated_domain_upper)


def compute_domain_range_single_hull(m, b, left, right):
  with tf.name_scope('compute_domain_range_hulls'):
    integrated_lower = evaluate_integrated(left, m, b)
    integrated_upper = evaluate_integrated(right, m, b)
    # compute the definite integral for this line segment.
    # the lower range for this segment will just be the constant term
    # the upper range will be the value of the definite integral for this line
    # segment, plus the cumulative integral of all the previous line segments,
    # which is stored in the value of the constant term
    definite_integral = integrated_upper - integrated_lower
    # print('m = {}, b = {}, left = {}, right = {}, definite_integral = {}'.format(m, b, left, right,  definite_integral))
    return integrated_lower, integrated_upper, definite_integral


# def evaluate_inverse(self, sample):#, constant):
#   """ will evaluate the inverse rate of this segment at `sample`"""
#   if(tf.math.abs(self.m) > self.epsilon):
#      return (-self.b + tf.math.sqrt(np.square(self.b) - 2.0 * self.m * (self.constant - sample))) / self.m
#   else:
#      return sample / self.b


def envelope(left, right, m, b, t):
  env = np.zeros(np.size(t))
  for i in range(0, np.size(t)):
    # find the hull segment that fits this sample
    try:
      seg_idx = np.where(t[i] <= right)[0][0]
      env[i] = m[seg_idx] * t[i] + b[seg_idx]
    except:
      pass
  return env


def integrated_envelope(left, right, m, b, c, t):
  env = np.zeros(np.size(t))
  for i in range(0, np.size(t)):
    #print('t = {}'.format(t[i]))
    # find the hull segment that fits this sample
    try:
      seg_idx = np.where(t[i] <= right)[0][0]
    except IndexError:
      # we reached the end so will extrapolate.
      # set the index to the last element
      seg_idx = right.size - 1
    env[i] = 0.5 * m[seg_idx] * t[i] **2.0 + b[seg_idx] * t[i] + c[seg_idx]
  return env



def inverse_integrated_envelope(int_left, int_right, m, b, c, y):
  """
  Note: the left and right vars are now the limits of the range of
  each segment in the integrated rate. The range in the integrated rate will
  serve as the domain for it's inverse
  """
  env = np.zeros(np.size(y))
  for i in range(0, np.size(y)):
    # find the hull segment that fits this sample
    try:
      seg_idx = np.where(y[i] <= int_right)[0][0]
    except IndexError:
      # we reached the end so will extrapolate.
      # set the index to the last element
      seg_idx = int_right.size - 1
    env[i] = (-b[seg_idx] + tf.math.sqrt(tf.square(b[seg_idx]) - 2.0 * m[seg_idx] * (c[seg_idx] - y[i]))) / m[seg_idx]
  return env


def sample_poisson(integrated_constant_tensor,
                   integrated_range_lower, integrated_range_upper,
                   inverse_integrated_domain_lower,
                   inverse_integrated_domain_upper,
                   m_tensor, b_tensor, left_tensor, right_tensor):
  """Performs sampling from the poisson process

  To sample using the inversion method, need
  to find
  t = t + \Lambda^{-1}(E + \Lambda(t))

  where E ~ exponential
  For our scenario, we are only trying to find the first sample
  in a poisson process, so T initially equals zero.
  Since our original hull (\lambda) is piecewise linear, that means
  that \Lambda is piewcewise quadratic and each segment is represented as
  \Lambda(t) = (1/2) * a * t^2 + b * t
  meaning that \Lambda(0) = 0, so we don't need to evaluate this, we only
  need to evaluate the inverse w.r.t E.
  """
  exp_dist = tfp.distributions.Exponential(1.0)
  E_sample = tf.cast(exp_dist.sample(), dtype=tf.float64)
  # now need to find which segment E is within the range of
  # for the inverse function
  #
  # logging.debug('E sample {}'.format(E_sample))
  less_than_upper = tf.cast(
    tf.reshape(tf.where(tf.less(E_sample, inverse_integrated_domain_upper)), [-1]),
    tf.int32)
  # print('less than test = {}'.format(tf.less(E_sample, inverse_integrated_domain_upper)))
  # need to handle edge case where the proposed sample is outside of the
  # upper bound of the final segment. In this case, we want to set the index to
  # be that of the final segment, and will extrapolate.
  # if this is the case, the dimensions of `less_than_upper` will be zero.
  # if this happens, will replace the values for the `less_than_upper` variable
  # with a scaler that holds the index to the last hull_segment, which will then
  # be used to sample from this possion process.
  cond_extrapolate = tf.math.equal(tf.size(less_than_upper), 0)
  # print('shape of inv_integrated_domain_upper = {}'.format(tf.shape(inverse_integrated_domain_upper)))
  one = tf.convert_to_tensor(1, dtype=tf.int32)
  fn_extrapolate = lambda: tf.reshape(
    tf.size(m_tensor) - 1, [-1])
  # if it is in range, the hull segment we are interested in is the first
  # hull segment indentified from the `less_than_upper` variable
  # print('less_than_upper = {}'.format(less_than_upper))
  # print('less_than_upper size  = {}'.format(tf.size(less_than_upper)))
  # print('inverse_integrated_upper_domain = {}'.format(inverse_integrated_domain_upper))
  # print('EE_sample = {}, inv_domain_upper = {}'.format(E_sample, inverse_integrated_domain_upper))
  # print('cond_extrapolate = {}'.format(cond_extrapolate))
  fn_in_range = lambda: tf.slice(less_than_upper, [0], [1])
  # fn_in_range = lambda: less_than_upper
  hull_segment_idx = tf.cond(cond_extrapolate, fn_extrapolate, fn_in_range)
  # print('integrated constant = {}, b = {}, idx = {}'.format(integrated_constant_tensor, b_tensor, hull_segment_idx))
  # print('Cond extrapolate = {}'.format(cond_extrapolate))
  # can now get the index for the appropriate hull segment by evaluating the
  # given sample for the inverse of the intergrated rate fn
  # print('less_than_upper = {}'.format(less_than_upper))
  # print('m_tensor = {}'.format(m_tensor))
  # print('right tensor shape = {}'.format(right_tensor.shape))
  # hull_segment_idx = tf.slice(less_than_upper, [0], [1])
  # print('hull segment ind = {}, size = {}'.format(hull_segment_idx,
  # tf.shape(hull_segment_idx)))
  # logging.debug('hull idx {}'.format(hull_segment_idx))
  force_refresh, poisson_sample, hull_eval = evaluate_hull_segment_inverse(
    E_sample, hull_segment_idx, m_tensor, b_tensor,  left_tensor, right_tensor,
    integrated_constant_tensor)
  # logging.debug('time sample {}'.format(poisson_sample))
  # logging.debug('hull eval {}'.format(hull_eval))
  #time.sleep(5)
  return force_refresh, poisson_sample, hull_eval



def evaluate_integrated(sample, m, b):#, constant=0.0):
  """ will evaluate the inverse rate of this segment at `sample`"""
  return tf.reshape(0.5 * m * sample ** 2.0 + b * sample, ())


def evaluate_hull_segment_inverse(sample, hull_segment_idx, m_tensor,
                                  b_tensor, left_tensor, right_tensor,
                                  constant_tensor,
                                  epsilon=1e-6):
  """ will evaluate the inverse rate of this segment at `sample`"""
  # print('hull segment ind = {}, size = {}'.format(hull_segment_idx,
  #                                                 tf.shape(hull_segment_idx)))
  m = tf.slice(tf.reshape(m_tensor, [-1]), hull_segment_idx, [1])
  b = tf.slice(tf.reshape(b_tensor, [-1]), hull_segment_idx, [1])
  # print('m tensor shape = {}'.format(m_tensor.shape))
  # print('left tensor shape = {}'.format(left_tensor.shape))
  # print('hull segment idx = {}'.format(hull_segment_idx))
  left = tf.slice(tf.reshape(left_tensor, [-1]), hull_segment_idx, [1])
  integrated_constant = tf.slice(
    tf.reshape(constant_tensor, [-1]), hull_segment_idx, [1])
  integrated_lower = evaluate_integrated(left, m, b)
  # when evaluating the inverse for this type of hull segment, we will be
  # dividing by the gradient of the original hull segment.
  # want to make sure this isn't too small, as it can result in overflow
  # If it is small, will just treat this hull segment as a flat line,
  # ie. m = 0
  cond = tf.math.greater(tf.math.abs(m), epsilon)
  def m_valid():
    force_refresh = False
    eval_inv_segment = (-b + tf.math.sqrt(tf.square(b) - 2.0 * m * (integrated_constant - sample))) / m
    less_than_upper = tf.cast(
      tf.reshape(tf.where(tf.less(eval_inv_segment, right_tensor)), [-1]),
      tf.int32)
    # now will put in an edge case if it is a case for extrapolation
    cond_extrapolate = tf.math.equal(tf.size(less_than_upper), 0)
    fn_extrapolate = lambda: tf.reshape(
      tf.size(m_tensor) - 1, [-1])
    # if it is in range, the hull segment we are interested in is the first
    fn_in_range = lambda: tf.slice(less_than_upper, [0], [1])
    env_segment_idx = tf.cond(cond_extrapolate, fn_extrapolate, fn_in_range)
    # now let's get the values for m and b we need for this segment
    m_env = tf.slice(tf.reshape(m_tensor, [-1]), env_segment_idx, [1])
    b_env = tf.slice(tf.reshape(b_tensor, [-1]), env_segment_idx, [1])
    # eval_hull_segment = m_env * eval_inv_segment + b_env
    eval_hull_segment = m * eval_inv_segment + b
    return force_refresh, eval_inv_segment, eval_hull_segment

  # def m_small():
  #   force_refresh = False
  #   eval_hull_segment = integrated_constant + b + epsilon
  #   eval_inv_segment = 1 / eval_hull_segment
  #   return force_refresh, eval_inv_segment, eval_hull_segment

  def m_small():
    return True, tf.convert_to_tensor(0.0, dtype=tf.float64), tf.convert_to_tensor(0.0, dtype=tf.float64)

  # m_valid = lambda: (-b + tf.math.sqrt(tf.square(b) - 2.0 * m * (integrated_constant - sample))) / m
  # m_small = lambda: (integrated_constant - sample) / b
  print('cond small m = {}'.format(cond))
  print('cond small m = {}'.format(cond))
  force_refresh, eval_inv_segment, eval_hull_segment = tf.cond(cond, m_valid, m_small)
  # this is the psuedo code we are running
  #   # if(tf.math.abs(m) > epsilon):
  #   #    return (-b + tf.math.sqrt(tf.square(b) - 2.0 * m * (constant - sample))) / m
  #   # else:
  #   #    return (integrated_constant - sample) / self.b
  # now evaluate the hull segment
  # need to get the index that the proposed time lies in
  # less_than_upper = tf.cast(
  #   tf.reshape(tf.where(tf.less(eval_inv_segment, right_tensor)), [-1]),
  #   tf.int32)
  # # now will put in an edge case if it is a case for extrapolation
  # cond_extrapolate = tf.math.equal(tf.size(less_than_upper), 0)
  # fn_extrapolate = lambda: tf.reshape(
  #   tf.size(m_tensor) - 1, [-1])
  # # if it is in range, the hull segment we are interested in is the first
  # fn_in_range = lambda: tf.slice(less_than_upper, [0], [1])
  # env_segment_idx = tf.cond(cond_extrapolate, fn_extrapolate, fn_in_range)
  # # now let's get the values for m and b we need for this segment
  # m_env = tf.slice(tf.reshape(m_tensor, [-1]), env_segment_idx, [1])
  # b_env = tf.slice(tf.reshape(b_tensor, [-1]), env_segment_idx, [1])
  # eval_hull_segment = m * eval_inv_segment + b
  return force_refresh, eval_inv_segment, eval_hull_segment



def eval_hull(self, time):
  # find the hull segment we should sample from
  less_than_upper = [time <= x.left for x in self.hull_list]
  # now find the first hull segment where our exponential variable is
  # greater than the lower bound of the range, as this will be the one
  # where we need to sample from inverse
  try:
    hull_segment_idx = np.where(less_than_upper)[0][-1]
  except IndexError:
    hull_segment_idx = 0
  # now evaluate the inverse
  return self.hull_list[hull_segment_idx].evaluate_hull(time)




class Hull(object):
  """a class that contains all the hull nodes

  Is used for sampling from the Hull for a Poisson Process
  """
  def __init__(self, hull_list):
    self.hull_list = hull_list
    # print('\nHull items')
    # for i in range(0, len(hull_list)):
    #   # print(self.hull_list[i])
    # now want to update the domain and ranges of the intgrated hulls
    constant = self.hull_list[0].evaluate_integrated(self.hull_list[0].left)
    for i in range(0, len(self.hull_list)):
      integrated_lower = self.hull_list[i].evaluate_integrated(self.hull_list[i].left)
      integrated_upper = self.hull_list[i].evaluate_integrated(self.hull_list[i].right)
      # if(i > 0):
      #   constant -= integrated_lower
      # print('constant = {}'.format(constant))
      # print('lower = {}'.format(integrated_lower))
      # print('constant - lower = {}'.format(constant - integrated_lower))
      self.hull_list[i].constant = constant - integrated_lower
      # now set the range for the integrated rate and the
      # domain for the inverse of the integrated rate, which will
      # be the same
      self.hull_list[i].integrated_range_lower = self.hull_list[i].constant
      self.hull_list[i].integrated_range_upper = integrated_upper + self.hull_list[i].constant
      self.hull_list[i].inverse_integrated_domain_lower =  constant
      self.hull_list[i].inverse_integrated_domain_upper = integrated_upper + self.hull_list[i].constant
      # now add increment the constant term, which is the difference
      # established by the integral of this hull
      # print('hull {}, int_range = ({}, {}], int_domain = ({}, {}]'.format(i, self.hull_list[i].integrated_range_lower,
      #                                                                     self.hull_list[i].integrated_range_upper,
      #                                                                     self.hull_list[i].integrated_domain_lower,
      #                                                                     self.hull_list[i].integrated_domain_upper))
      constant += integrated_upper - integrated_lower



  def sample_poisson(self):
    """ Performs sampling from the poisson process

    To sample using the inversion method, need
    to find
    t = t + \Lambda^{-1}(E + \Lambda(t))

    where E ~ exponential
    For our scenario, we are only trying to find the first sample
    in a poisson process, so T initially equals zero.
    Since our original hull (\lambda) is piecewise linear, that means
    that \Lambda is piewcewise quadratic and each segment is represented as
    \Lambda(t) = (1/2) * a * t^2 + b * t
    meaning that \Lambda(0) = 0, so we don't need to evaluate this, we only
    need to evaluate the inverse w.r.t E.
    """
    E_sample = np.random.exponential(1.0)
    # now need to find which segment E is within the range of
    # for the inverse function
    less_than_upper = [E_sample <= x.inverse_integrated_domain_upper for x in self.hull_list]
    # now find the first hull segment where our exponential variable is
    # greater than the lower bound of the range, as this will be the one
    # where we need to sample from inverse
    try:
      hull_segment_idx = np.where(less_than_upper)[0][0]
      # now evaluate the inverse
      # first need to find the additive constant for our integrated rate,
      # which comes from the evaluation of all the complete segments prior to
      # the one where our value of E_sample lies
      #constant = np.sum([x.inverse_int_domain_upper - x.inverse_int_domain_lower for x in self.hull_list[0:hull_segment_idx]])
      poisson_sample = self.hull_list[hull_segment_idx].evaluate_inverse(E_sample)#, constant)
      return poisson_sample
    except IndexError:
      # print('HULL SEGMENT ERROR')# - INDEX PROPOSED = {}, length = {}'.format(hull_segment_idx, len(self.hull_list)))
      # print('less than upper = {}'.format(less_than_upper))
      inverse_upper = [x.inverse_integrated_domain_upper for x in self.hull_list]
      # print('E_SAMPLE = {}'.format(E_sample))
      # print('Inverse_upper = {}'.format(inverse_upper))
      # use the final segment to sample
      # print('approximating with final segment to sample')
      return self.hull_list[-1].evaluate_inverse(E_sample)#, constant)


  def eval_hull(self, time):
    # find the hull segment we should sample from
    less_than_upper = [time <= x.left for x in self.hull_list]
    # now find the first hull segment where our exponential variable is
    # greater than the lower bound of the range, as this will be the one
    # where we need to sample from inverse
    try:
      hull_segment_idx = np.where(less_than_upper)[0][-1]
    except IndexError:
      hull_segment_idx = 0
    # now evaluate the inverse
    return self.hull_list[hull_segment_idx].evaluate_hull(time)


  def eval_inverse_integrated(self, time=None, sample=100):
    if(time is None):
      time = np.linspace(self.hull_list[0].inverse_integrated_domain_lower,
                         self.hull_list[-1].inverse_integrated_domain_upper, 100)
    # for hull in self.hull_list:
    #   # print(hull)
    #   # print('inverse domain = ( {}, {} )'.format(hull.inverse_integrated_domain_lower,
    #   #                                            hull.inverse_integrated_domain_upper))
    #   # print('constant = {}'.format(hull.constant))
    inverse = []
    for t in time:
      #print('t = {}, num_samples = {}'.format(t, len(inverse)))
      less_than_upper = [t <= x.inverse_integrated_domain_upper for x in self.hull_list]
      #print(less_than_upper)
      #print(np.where(less_than_upper)[0])
      hull_segment_idx = np.where(less_than_upper)[0][0]
      #constant = np.sum([x.inverse_integrated_range_upper - x.inverse_integrated_range_lower for x in self.hull_list[0:hull_segment_idx]])
      inverse.append(self.hull_list[hull_segment_idx].evaluate_inverse(t))#, constant))
    return time, np.array(inverse)


  def eval_integrated(self, time=None, sample=100):
    if(time is None):
      time = np.linspace(self.hull_list[0].integrated_domain_lower,
                         self.hull_list[-1].integrated_domain_upper, 100)
    integrated = []
    for t in time:
      #print('t = {}'.format(t))
      less_than_upper = [t <= x.integrated_domain_upper for x in self.hull_list]
      #print(less_than_upper)
      #print(np.where(less_than_upper)[0])
      hull_segment_idx = np.where(less_than_upper)[0][0]
      #constant = np.sum([x.inverse_integrated_domain_upper - x.inverse_integrated_domain_lower for x in self.hull_list[0:hull_segment_idx]])
      integrated.append(self.hull_list[hull_segment_idx].evaluate_integrated(t))#, constant))
    return time, np.array(integrated)



class HullNode(object):
  def __init__(self, m, b, left, right):
    self.m = m
    self.b = b
    self.left = left
    self.right = right
    # adding range for the method as it will help us out later
    self.range_lower = self.m * self.left + self.b
    self.range_upper = self.m * self.right + self.b
    self.integrated_domain_lower = 0.0 #self.left
    self.integrated_domain_upper = self.right
    # need to evaluate with any previous nodes to find the range
    self.integrated_range_lower = 0.0
    self.integrated_range_upper = 0.0
    # similarly, need to evaluate with any previous nodes to find the domain of
    # the inverse of the integral, which will be the same as the range of the
    # integrated rate
    self.inverse_integrated_domain_lower = 0.0
    self.inverse_integrated_domain_upper = 0.0
    self.inverse_integrated_range_lower = 0.0 #self.left
    self.inverse_integrated_range_upper = self.right
    # add a constant term, which will be found when evaluating the integral
    # of all the hull sections
    self.constant = 0.0
    self.epsilon = 0.0000001

  def evaluate_inverse(self, sample):#, constant):
    """ will evaluate the inverse rate of this segment at `sample`"""
    if(tf.math.abs(self.m) > self.epsilon):
       return (-self.b + tf.math.sqrt(np.square(self.b) - 2.0 * self.m * (self.constant - sample))) / self.m
    else:
       return sample / self.b

  def evaluate_integrated(self, sample):#, constant=0.0):
    """ will evaluate the inverse rate of this segment at `sample`"""
    if(tf.math.abs(self.m) > self.epsilon):
       return self.m * sample ** 2.0 / 2.0 + self.b * sample + self.constant
    else:
       return self.b * sample

  def evaluate_hull(self, sample):
    return self.m * sample + self.b

  def __eq__(self, other):
    from math import isclose

    def close(a, b):
      if a is b is None:
        return True
      if (a is None and b is not None) or (b is None and a is not None):
        return False
      return isclose(a, b, abs_tol=1e-02)

    return all((
      close(self.m, other.m), close(self.left, other.left),
      close(self.right, other.right)
    ))

  def __repr__(self):
    return "HullNode(m={m}, b={b}, left={left}, right={right})".format(
      m=self.m, b=self.b, left=self.left, right=self.right)

  def __hash__(self):
    return hash(str(self))


def slice_one(input_, index):
  return tf.slice(input_, [index], size=[1])

def slice_one_from_end(input_, index):
  """Slice from end, index must be negative!!"""
  if(index > 0):
    raise ValueError('Index in slice_one_from_end must be negative')
  return tf.slice(input_, [tf.size(input_) + index], size=[1])

def compute_hulls(S, fS, domain):
  """
  (Re-)compute upper and lower hull given
  the segment points `S` with function values
  `fS` and the `domain` of the logpdf.

  Parameters
  ----------
  S : np.ndarray (N, 1)
     Straight-line segment points accumulated thus far.

  fS : tuple
    Value of the `logpdf` under sampling for each
    of the given segment points in `S`.

  domain : Tuple[float, float]
    Domain of `logpdf`.
    May be unbounded on either or both sides,
    in which case `(float("-inf"), float("inf"))`
    would be passed.
    If this domain is unbounded to the left,
    the derivative of the logpdf
    for x<= a must be positive.
    If this domain is unbounded to the right          the derivative of the logpdf for x>=b
    must be negative.

  Returns
  ----------
  lower_hull: List[arspy.hull.HullNode]
  upper_hull: List[arspy.hull.HullNode]

  """
  # create an index for the loop
  loop_idx = tf.constant(0, dtype=tf.int32)
  # creating tensor Arrays for the gradient, intercept,
  # start and stop points for each hull segment
  # gradient = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
  #                           clear_after_read=False)
  # intercept = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
  #                            clear_after_read=False)
  # left = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
  #                           clear_after_read=False)
  # right = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
  #                        clear_after_read=False)
  # first line
  segment_index = tf.constant(0, dtype=tf.int32)
  m_array = tf.TensorArray(tf.float64, size=0,
                           dynamic_size=True, clear_after_read=False)
  b_array = tf.TensorArray(tf.float64, size=0,
                           dynamic_size=True, clear_after_read=False)
  left_array = tf.TensorArray(tf.float64, size=0,
                           dynamic_size=True, clear_after_read=False)
  right_array = tf.TensorArray(tf.float64, size=0,
                           dynamic_size=True, clear_after_read=False)
  m = (slice_one(fS, 1) - slice_one(fS, 0)) / (slice_one(S, 1) - slice_one(S, 0))
  m_array = m_array.write(segment_index, m)
  b_array = b_array.write(segment_index, slice_one(fS, 1) - m * slice_one(S, 1))
  left_array = left_array.write(segment_index, slice_one(S, 0))
  right_array = right_array.write(segment_index, slice_one(S, 1))
  segment_index = segment_index + 1
  # interior lines
  # there are two lines between each abscissa
  sample_index = tf.constant(1)
  def loop_cond(sample_index, *args):
    return tf.less(sample_index, tf.size(S) - 2)
  # wrapper for the loop function
  # abstracts out the sS anf fS arrays, as they aren't
  # to be altered
  def loop_wrapper(sample_index, segment_index,
                   m_array, b_array,
                   left_array, right_array):
    return compute_hull_loop_fn(sample_index, segment_index,
                                m_array, b_array,
                                left_array, right_array,
                                S, fS)
  (sample_index, segment_index,
   m_array, b_array,
   left_array, right_array) = tf.while_loop(
     loop_cond, loop_wrapper, loop_vars=(
       sample_index, segment_index, m_array, b_array, left_array, right_array))

  # last line
  m = (slice_one_from_end(fS, -2) - slice_one_from_end(fS, -3)) / (slice_one_from_end(S, -2) - slice_one_from_end(S, -3))
  b = slice_one_from_end(fS, -2) - m * slice_one_from_end(S, -2)
  m_array = m_array.write(segment_index, m)
  b_array = b_array.write(segment_index, b)
  left_array = left_array.write(segment_index, slice_one_from_end(S, -2))
  right_array = right_array.write(segment_index, slice_one_from_end(S, -1))
  # print('m array shape = {}'.format(m_array.stack()))
  # print('left array shape = {}'.format(left_array.stack()))
  return m_array, b_array, left_array, right_array

def compute_hull_loop_fn(sample_index, segment_index,  m_array, b_array,
                         left_array, right_array, S, fS):
  # slicing values for needed for this iteration
  S_i = slice_one(S, sample_index)
  S_i_m1 = slice_one(S, sample_index -1)
  S_i_p1 = slice_one(S, sample_index + 1)
  S_i_p2 = slice_one(S, sample_index + 2)
  fS_i = slice_one(fS, sample_index)
  fS_i_m1 = slice_one(fS, sample_index -1)
  fS_i_p1 = slice_one(fS, sample_index + 1)
  fS_i_p2 = slice_one(fS, sample_index + 2)
  # calculating the proposed gradients and biases for the segments
  m1 = (fS_i - fS_i_m1) / (S_i - S_i_m1)
  b1 = fS_i - m1 * S_i
  m2 = (fS_i_p2 - fS_i_p1) / (S_i_p2 - S_i_p1)
  b2 = fS_i_p1 - m2 * S_i_p1
  # differentials, will be used to find where they intersetct
  dx1 = S_i - S_i_m1
  df1 = fS_i - fS_i_m1
  dx2 = S_i_p2 - S_i_p1
  df2 = fS_i_p2 - fS_i_p1
  # evaluated points
  # TODO: Remove these, they are just repeated variables
  f1 = fS_i
  f2 = fS_i_p1
  x1 = S_i
  x2 = S_i_p1
  # this is where the lines intersect
  # more numerically stable than above
  # print('fs_i = {}, fs_i_m1 = {}, sample_index = {}'.format(fS_i, fS_i_m1, sample_index))
  # print('dx1 = {}, dx2 = {}, df1 = {}, df2 = {}'.format(dx1, dx2, df1, df2))
  # print('denominator = {}'.format((df2 * dx1 - df1 * dx2)))
  ix = ((f1 * dx1 - df1 * x1) * dx2 - (f2 * dx2 - df2 * x2) * dx1) / (df2 * dx1 - df1 * dx2)
  # now doing some error checking incase some infs appeared
  # overwrite the value of ix if there was any infs that appeared
  ix = check_intercept(ix, m1, m2, S_i, S_i_p1)
  # now do error checking to see if the segment for a strict upper hull
  # is well defined. If not, need to create line segment that connects the
  # two points.
  not_concave_segment_cond = check_concave_segment(ix, m1, m2, b1,
                                                   fS_i, S_i, S_i_p1)
  concave_segment_lambda = lambda: concave_segment_fn(
    segment_index, ix, m1, b1, m2, b2, S, sample_index,
    m_array, b_array,
    left_array, right_array)
  not_concave_segment_lambda = lambda: not_concave_segment_fn(
    segment_index, S, fS, sample_index,
    m_array, b_array,
    left_array, right_array)
  segment_index, m_array, b_array, left_array, right_array = tf.cond(
    not_concave_segment_cond, not_concave_segment_lambda, concave_segment_lambda)
  # increment the sample index
  sample_index = sample_index + 1
  return (sample_index, segment_index, m_array, b_array, left_array, right_array)


def concave_segment_fn(segment_index, ix, m1, b1, m2, b2, S, sample_index,
                       m_array, b_array,
                       left_array, right_array):
  # print('adding concave segment')
  m_array = m_array.write(segment_index, m1)
  b_array = b_array.write(segment_index, b1)
  left_array = left_array.write(segment_index, slice_one(S, sample_index))
  right_array = right_array.write(segment_index, ix)
  segment_index = segment_index + 1
  m_array = m_array.write(segment_index, m2)
  b_array = b_array.write(segment_index, b2)
  left_array = left_array.write(segment_index, ix)
  right_array = right_array.write(segment_index, slice_one(S, sample_index + 1))
  segment_index = segment_index + 1
  return segment_index, m_array, b_array, left_array, right_array


def not_concave_segment_fn(segment_index, S, fS, sample_index,
                           m_array, b_array,
                           left_array, right_array):
  # only add on segment that goes from
  # (slice_one(S, li), slice_one(fS, li)) to (slice_one(S, li + 1), slice_one(fS, li + 1))
  # print('adding non-concave segment')
  m = (slice_one(fS, sample_index + 1) - slice_one(fS, sample_index)) / (slice_one(S, sample_index + 1) - slice_one(S, sample_index))
  b = slice_one(fS, sample_index + 1) - m * slice_one(S, sample_index + 1)
  m_array = m_array.write(segment_index, m)
  b_array = b_array.write(segment_index, b)
  left_array = left_array.write(segment_index, slice_one(S, sample_index))
  right_array = right_array.write(segment_index, slice_one(S, sample_index + 1))
  # increment segment_index
  segment_index = segment_index + 1
  return segment_index, m_array, b_array, left_array, right_array


def check_intercept(ix, m1, m2, S, S_plus_one):
  """error checking to accomodate any weird errors that could
  pop up regarding the interercept of two hull sections"""
  m1_inf_cond = tf.logical_or(tf.math.is_inf(m1),
                              tf.less(tf.math.abs(m1 - m2),  10.0 ** -7))
  m1_inf = lambda: S
  m2_inf_cond = tf.math.is_inf(m2)
  m2_inf = lambda: S_plus_one
  default_fn = lambda: ix
  # print('m1_inbf_cond = {}, m2_inf_cond ={}'.format(m1_inf_cond, m2_inf_cond))
  ix = tf.case([(m1_inf_cond, m1_inf), (m2_inf_cond, m2_inf)], default=default_fn)
  return ix


def check_concave_segment(ix, m1, m2, b1, fS_i, S, S_plus_one):
  # want to conver the following code into a single statement
  # (ix < slice_one(S, li)) or (ix > slice_one(S, li + 1)) or tf.math.is_inf(ix)
  #    or tf.math.is_nan(ix) or (tf.math.is_inf(m1) and tf.math.is_inf(m2)):
  # will do this by seperating into smaller statements and then combining
  cond_one = tf.logical_or(tf.math.less(ix, S), tf.math.greater(ix, S_plus_one))
  cond_two = tf.logical_or(tf.math.is_inf(ix), tf.math.is_nan(ix))
  # TODO: Check should this be an `or` condition?
  cond_three = tf.logical_and(tf.math.is_inf(m1), tf.math.is_inf(m2))
  # check that the difference between the proposed intercept is of
  # sufficient distance apart from the start and end position
  # if it isn't, will just treat as non-concave segment
  # as we won't loos anything meaningful by including it
  intercept_eps = tf.constant(1.0 ** 10e-7, dtype=tf.float64)#1.0 ** 10e-7)
  left_diff = tf.math.abs(ix - S)
  right_diff = tf.math.abs(ix - S_plus_one)
  cond_four = tf.logical_or(tf.less(left_diff, intercept_eps),
                            tf.less(right_diff, intercept_eps))
  # if the the gradient of m1 or m2 is equal to zero,
  # then we say we are at a clipped point where the event rate has been
  # less than zero and then clipped
  cond_five = tf.logical_or(tf.less_equal(m1, intercept_eps),
                            tf.less_equal(m2, intercept_eps))
  # one more condition, if the value of the hull at the found intercept is
  # less than the value at fS_i, then won't be a convex section
  cond_six = tf.less_equal(m1 * ix + b1, fS_i)

  # print('cond_one = {}, cond_two = {}, cond_three = {}, cond_four = {}'.format(
  #   cond_one, cond_two, cond_three, cond_four))
  # print('ix = {}, S = {}, S_plus_one = {}'.format(ix, S, S_plus_one))
  # now combining
  cond_one_or_two = tf.logical_or(cond_one, cond_two)
  cond_three_or_four =  tf.logical_or(cond_three, cond_four)
  cond_one_two_three_four = tf.logical_or(cond_one_or_two, cond_three_or_four)
  cond_five_six = tf.logical_or(cond_five, cond_six)
  cond_all = tf.logical_or(cond_one_two_three_four, cond_five_six)
  return cond_all
