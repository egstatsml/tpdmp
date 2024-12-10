"""
Implements the different Poisson Process Samplers

There is a couple different methods to use. Most of them
will revolve around some form of thinning, but also nice to implement
analytic sampling when possible.

#### References

"""
from typing import Tuple
import abc
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
import sys
import collections
import time as timepy

from tbnn.pdmp.utils import compute_dot_prod
from tbnn.nn.mlp import MLP
from tbnn.pdmp.hull_tf import sample_poisson_thinning


class IPPSampler(object, metaclass=abc.ABCMeta):
  """Base class for Inhomogeneous Poisson Process Sampling

  Is an abstract class. It won't do anything, will only define the
  methods that child classes need to implement, and the attributes
  that are shared accross all instances.
  """

  def __init__(self, exact=False):
    # add the unit exponential r.v. which is used for sampling
    # from pretty much all the different methods
    self.exp_dist = tfp.distributions.Exponential(1.0)
    # variable to describe when the sampler is exact or not
    self.exact = exact
    # also adding a summary write so I can track some things
    self.summary_writer = train_summary_writer = tf.summary.create_file_writer('./logdir')
    self.summary_idx = 0#tf.constant(0, dtype=tf.int64)
    self.init_envelope = np.array([0.0]).astype(np.float32)
    self.num_init = tf.convert_to_tensor(self.init_envelope.size)
    self.init_envelope = [tf.convert_to_tensor(x) for x in self.init_envelope]
    self.max_iter = 10
    self.max_time = tf.constant(50.0, dtype=tf.float32)#, trainable=False)
    self.proposal_uniform = tfd.Uniform()
    self.iteration = tf.constant(0, dtype=tf.int32)#tf.Variable(0, dtype=tf.int32)#, trainable=False)


  @abc.abstractmethod
  def simulate_bounce_time(self):
    raise NotImplementedError('Abstract method: child class most overwrite')



class InterpolationSampler(IPPSampler):
  """Piecewise Linear Interpolation for hull formation

  The SBPS looks at picewise bayesian linear regression,
  but I want to look at how the piecewise linear method would work for
  iterpolation.
  """

  def __init__(self, batch_size=1, data_size=1):
    super().__init__()
    self.init_slope = 0
    self.init_intercept = 0
    self.gamma = tf.convert_to_tensor(1.0, dtype=tf.float32)
    self.min_event = tf.convert_to_tensor(0.01, dtype=tf.float32)

  def evaluate_rate(self, target_log_prob_fn, state, velocity,
                    iteration, G, X, time):
    print('IN EVAL RATE, iteration = {}'.format(iteration))
    with tf.name_scope('add_sample_upper_bound'):
      next_state = [s + v * time for s, v in zip(state, velocity)]
      grads_target_log_prob = target_log_prob_fn(next_state)
      event_rate = compute_dot_prod(velocity, grads_target_log_prob)
      print('next state = ', next_state)
      print('grads = ', grads_target_log_prob)
      print('state = ', state)
      print('velocity = ', velocity)
      print('time = ', time)
      print('here')
      # timepy.sleep(100)
      G, X, iteration = self.update_envelope_and_time(event_rate, time,
                                                      G, X, iteration)
    return iteration, G, X, event_rate


  def initialise_hull(self, target_log_prob_fn, state, velocity,
                      G, X, iteration):
    """Initialise samples for that will form the hull"""
    # now call the function to evaluate the hull at this loc
    start_time = tf.convert_to_tensor(0.0, dtype=tf.float32)
    dt = tf.convert_to_tensor(10.0, dtype=tf.float32)
    # now evaluate the starting points
    event_zero = self.evaluate_initialise_hull(
      target_log_prob_fn,
      state, velocity,
      0)

    event_end = self.evaluate_initialise_hull(
      target_log_prob_fn,
      state, velocity,
      dt)
    # now see where it intersects the x axis
    # if it intersects x axis for x < 0, then just set the first
    # eval rate to that at event_zero and set first element of X
    # to be zero
    # otherwise, x value to where it intersects and the event rate to be zero
    self.init_slope = (self.gamma * event_end - self.gamma * event_zero) / dt
    self.init_intercept = self.gamma * event_end - self.init_slope * dt
    x_intercept = - self.init_intercept / self.init_slope
    neg_x_intercept_cond = tf.less(x_intercept, 0)

    def x_intercept_negative():
      return G.write(iteration, event_zero), X.write(iteration, tf.reshape([0, 1.0], [1, 2]))

    def x_intercept_positive():
      return G.write(iteration, 0), X.write(iteration, tf.reshape([x_intercept, 1.0], [1, 2]))

    G, X = tf.cond(neg_x_intercept_cond, x_intercept_negative, x_intercept_positive)
    iteration = iteration + 1
    print(f'x_intercept = {x_intercept}')
    print(f'event_zero = {event_zero}')
    print(f'event_end = {event_end}')
    print(f'dt = {dt}')
    print(f'init_slope = {self.init_slope}')
    print(f'init_intercept = {self.init_intercept}')
    return G, X, iteration


  def evaluate_initialise_hull(self, target_log_prob_fn, state, velocity, time):
    with tf.name_scope('eval_init_upper_bound'):
      next_state = [s + v * time for s, v in zip(state, velocity)]
      print(f'eval_init_hull {next_state[-1]}')
      grads_target_log_prob = target_log_prob_fn(next_state)
      event_rate = compute_dot_prod(velocity, grads_target_log_prob)
      return event_rate


  def update_envelope_and_time(self, envelope_value, time, G, X, iteration):
    # make sure time is correct shape here
    time = tf.reshape(time, ())
    G = G.write(iteration, envelope_value)
    X = X.write(iteration, tf.reshape([time, 1.0], [1, 2]))
    iteration = iteration + 1
    return G, X, iteration


  def simulate_bounce_time(self, target_log_prob_fn, state, velocity):
    def accepted_fn(proposal_u, ratio, iteration, proposed_time, linear_time, G, X):
      """function that just determines if the proposal was accepted or not
      based on the `accepted` arg.
      """
      return tf.math.logical_or(tf.math.greater(proposal_u, ratio),
                               tf.math.greater(ratio, 2.0))
    # make sure the iteration index is set to zero
    iteration = tf.convert_to_tensor(0, dtype=tf.int32)
    # initialise linear time and proposed time vars.
    delta_time = tf.convert_to_tensor(0.0, dtype=tf.float32)
    proposed_time = tf.reshape(
      tf.convert_to_tensor(1.0, dtype=tf.float32), ())
    # tensor arrays for holding the gradients and X values used for the linear approx.
    G = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    X = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    # itialise vars. for samples from the proposal distribution and the
    G, X, iteration = self.initialise_hull(target_log_prob_fn, state,
                                           velocity, G, X, iteration)
    print('here')
    print(G.size())
    print(X.size())


    # acceptance ratio
    proposal_u = tf.reshape(tf.convert_to_tensor(1.0, dtype=tf.float32), (1,))
    ratio = tf.reshape(tf.convert_to_tensor(0.0, dtype=tf.float32), (1,))
    with tf.name_scope(mcmc_util.make_name('ipp', 'interpolation', 'simulate_bounce')):
      proposal_u, ratio, iteration, proposed_time, delta_time, G, X = tf.while_loop(
        accepted_fn,
        body=lambda proposal_u, ratio, iteration, proposed_time, delta_time, G, X: self.simulate_bounce_loop(
          target_log_prob_fn, state, velocity, proposal_u, ratio, iteration, proposed_time, delta_time, G, X),
        loop_vars=(proposal_u, ratio, iteration, proposed_time, delta_time, G, X),
        maximum_iterations=self.max_iter)
    # if the maximum number of iterations was reached, set the proposed time
    # to be really large so that a refresh will happen at this point
    max_reach = lambda: self.max_time
    valid_proposal = lambda: proposed_time
    bounce_time = tf.cond(tf.math.greater_equal(self.iteration - 1, self.max_iter),
                         max_reach, valid_proposal)
    print('bounce__time = {}'.format(bounce_time))
    return bounce_time, tf.reshape(ratio, ())

  def simulate_bounce_loop(self, target_log_prob_fn, state, velocity,
                           proposal_u, ratio, iteration,
                           proposed_time, delta_time, G, X):
    with tf.name_scope('interpolation_bounce_loop'):
      # get the slope and intercept for the current iteration
      # and the current integrated constant
      slope, intercept = self.get_slope_intercept(G, X, iteration)
      print(f'slope = {slope}')
      print(f'intercept = {intercept}')
      proposed_time, gamma = self.interpolation_propose_time(
        G, X, iteration, slope, intercept)
      # compute the event rate at this proposed time
      iteration, G, X, event_rate = self.evaluate_rate(target_log_prob_fn,
                                                       state,
                                                       velocity,
                                                       iteration,
                                                       G, X, proposed_time)
      print('event_rate raw = {}'.format(event_rate))
      event_rate = tf.math.maximum(event_rate, 0.0)
      ratio = event_rate / gamma
      # now make sure that if a Nan or a inf krept in here (very possible)
      # that we attenuate it to zero
      # the is_finite function will find if it is a real number
      ratio = tf.where(~tf.math.is_finite(ratio), tf.zeros_like(ratio), ratio)
      # now draw a uniform sample to see if we accepted this proposal
      proposal_u = self.proposal_uniform.sample(1)
      print('event_rate = {}'.format(event_rate))
      print('gamma = {}'.format(gamma))
      print('proposal_u = {}'.format(proposal_u))
      print('ratio = {}'.format(ratio))
      tf.print('ratio = {}'.format(ratio))
      print('X = {}'.format(X))
      refresh_cond  = tf.math.less(proposed_time, 0)
      force_refresh_fn = lambda: (1.0, 10e6)
      accept_fn = lambda: (ratio, proposed_time)
      ratio, proposed_time = tf.cond(refresh_cond, force_refresh_fn, accept_fn)
      return tf.reshape(proposal_u, (1,)), tf.reshape(ratio, (1,)), iteration, tf.reshape(proposed_time, ()), delta_time, G, X

  def get_slope_intercept(self, G: tf.TensorArray,
                          X: tf.TensorArray,
                          iteration: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Slope/gradient, intercept and integrated constant

    These variables are needed to describe the current linear segment in our
    interpolated scheme.

    Parameters
    ----------
    G : TensorArray
        Array of hull segments
    X : TensorArray
        Array of time samples
    iteration : int
        iteration for hull formation
    """
    # get the current slope and intercept
    # if the number of samples in our current hull is only one, then
    # we will set the initial slope variable to be an arbitrary constant.
    # for here we will just say it is zero, and will keep the value of the hull
    # sample as the intercept for the current iteration.
    print(f'iteration = {iteration}')
    cond = tf.math.less_equal(iteration, 1)
    def first_iteration():
      # don't need to multiply by gamma here
      # did it when initialising
      slope = self.init_slope
      intercept = self.init_intercept
      tf.print(f'first iter', output_stream=sys.stdout)
      tf.print(f'X shape ={ X.read(iteration - 1)[0,0]}', output_stream=sys.stdout)
      print('first iteration')
      print(f'first slope = {slope}')
      print(f'first intercept = {intercept}')
      return slope, intercept

    def following_iteration():
      # G_one = tf.math.maximum(G.read(iteration - 1), self.gamma)
      # G_two = tf.math.maximum(G.read(iteration - 2), self.gamma)
      G_one = G.read(iteration - 1) * self.gamma
      G_two = G.read(iteration - 2) * self.gamma

      X_one = X.read(iteration - 1)[0, 0]
      X_two = X.read(iteration - 2)[0, 0]

      slope = (G_one - G_two) / (X_one - X_two)
      # tf.print(f'X shape ={ X.read(iteration - 1).shape}', output_stream=sys.stdout)
      print('following iteration')
      intercept = G_one - slope * X_one
      return slope, intercept

    slope, intercept = tf.cond(cond, first_iteration, following_iteration)
    return tf.reshape(slope, ()), tf.reshape(intercept, ())

  def interpolation_propose_time(self,
                                 G: tf.TensorArray,
                                 X: tf.TensorArray,
                                 iteration: int,
                                 slope: tf.Tensor,
                                 intercept: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Propose event time for current hull

    Use current linear segment and information from previous segments stored in
    the integrated constant to sample from a proposal IPP.

    This method is capable of sampling from affine linear segments of the form
    ax + b. Checks need to be done to evaluate whether the slope "a" approximately
    equals zero. If a < epsilon, we treat this as a == 0, and treat the current linear
    segment as a constant.

    This function will also do some checks to make sure the proposed time/linear
    segment is valid. These checks include,
    Check 1:
    - making sure that the proposed time is not within a negative segment of the current
      hull. For example, if the slope is negative, and with the sampled time from the
      random variate ends up in a position such that the line segment is negative, we
      need to correct it, as the proposal of the time will end up with a Nan value.
    Fix:
      Two fixes.
      first is just to treat the current segment with a constant bounds that is at the
      maximum value of the segment (which will be the start, so a * t_{i-1} * b)
      where t_{i-i} was the time of the previous segment.

      Also have a case that will do another check to see if is invalid, and if it is will
      return the end of the segment that is positive and that time.
    Parameters
    ----------
    X : tf.TensorArray
        Time sample array
    iteration : int
        iteration in hull formation process
    slope : tf.Tensor
        slope of current linear segment in the proposal hull
    intercept : tf.Tensor
        intercept of current linear segment in the proposal hull
    integrated_constant : tf.Tensor
        cumulative sum of the integral of all   prior hull segments
    epsilon : float
        minimum value used to determine if the current slope value
        is suitable for inversion with quadratic formula, or should use
        constant approximation

    Returns
    -------
    tf.Tensor
        proposed event time
    tf.Tensor
        proposed event rate of hull at the proposed event time
    """
    print(f'iteration = {iteration}')
    exp_sample = np.random.exponential(1.0)
    # time for the last event
    last_time = X.read(iteration - 1)[0][0]
    # now update the time with exponentially sampled random variable
    # updated_time = last_time + exp_sample
    # going to cast all the values needed to double precision to avoid numeric
    # overflow/underflow
    last_time = tf.cast(last_time, dtype=tf.float64)
    # updated_time = tf.cast(updated_time + exp_sample, dtype=tf.float64)
    slope = tf.cast(slope, dtype=tf.float64)
    intercept = tf.cast(intercept, dtype=tf.float64)
    min_slope = tf.cast(1e-7, dtype=tf.float64)
    # creating a condition to check whether the current slope is an affine
    # linear function, or is a constant
    # if the slope is negative, will treat as a constant
    affine_cond = tf.math.greater(slope, min_slope)
    # affine_cond = tf.math.greater(tf.abs(slope), min_slope)
    print(f'casted slope = {slope}')

    def affine_segment():
      print('affine segment')
      # need to sample a log-uniform variate
      u = 1.0 - tf.random.uniform([1], dtype=tf.float64)
      U = tf.math.log(u)
      print(f'u = {u}')
      print(f'U = {U}')
      prev_terms = slope ** 2.0 * last_time ** 2.0 + 2 * slope * intercept * last_time
      print(f'prev_terms = {prev_terms}')
      proposed_time = (
        -intercept + tf.math.sqrt(
          intercept ** 2.0 + prev_terms - 2 * slope * U)) / slope
      proposed_hull_eval = slope * proposed_time + intercept
      return proposed_time, proposed_hull_eval

    def constant_segment():
      u = 1.0 - tf.random.uniform([1], dtype=tf.float64)
      U = tf.math.log(u)
      print('constant segment')
      eval_rate = intercept + last_time * slope
      proposed_time = -U / eval_rate + last_time
      proposed_hull_eval = eval_rate
      return proposed_time, proposed_hull_eval

    proposed_time, proposed_hull_eval = tf.cond(affine_cond, affine_segment, constant_segment)
    print(f'last_time = {X.read(iteration - 1)[0][0]}')
    print(f'proposed_time = {proposed_time}')
    print(f'proposed_hull_eval = {proposed_hull_eval}')
    # if the proposed time or hull eval is nan, then the hull segment at the proposed time went
    # negative, so we have to correct the proposed time, proposed eval at that time and
    # the slope and intercept for the current linear segment.
    # The method for adjusting the linear segment is described in the docstring
    valid_cond = tf.math.logical_not(
      tf.math.logical_or(
        tf.math.is_nan(proposed_time), tf.math.is_nan(proposed_hull_eval)))
    # check make sure isn't negative
    valid_cond = tf.math.logical_or(valid_cond, tf.math.greater(proposed_hull_eval, 0))
    valid_cond = tf.math.logical_or(valid_cond, tf.math.greater(proposed_time, 0))

    def valid_proposal() -> Tuple[tf.Tensor, tf.Tensor]:
      """Proposed time and hull evaluation at this time is valid

      Will simply return the proposed time and hull eval, and the original slope
      and intercept

      Returns
      -------
      tf.Tensor
          original proposed event time
      tf.Tensor
          original proposed event rate of hull at the proposed event time
      """
      return proposed_time, proposed_hull_eval

    def invalid_proposal() -> Tuple[tf.Tensor, tf.Tensor]:
      """Proposed time and hull evaluation at this time is invalid

      We will update the proposed hull eval to the minumum hull value that
      we will permit for our hull definition, and we will update the proposed time
      to the value where the current line segment will intersect with such region

      Returns
      -------
      tf.Tensor
          original proposed event time
      tf.Tensor
          original proposed event rate of hull at the proposed event time
      """
      tf.print(f'updated_slope!!!', output_stream=sys.stdout)
      min_event = tf.cast(self.min_event, tf.float64)
      proposed_time = (min_event - intercept) / slope
      # proposed_hull_eval = tf.reshape(gamma,  (1,))
      proposed_hull_eval = tf.reshape(min_event,  (1,))
      return proposed_time, proposed_hull_eval

    proposed_time, proposed_hull_eval = tf.cond(valid_cond, valid_proposal, invalid_proposal)
    # now cast returns back to single precision
    proposed_time = tf.cast(proposed_time, dtype=tf.float32)
    proposed_hull_eval = tf.cast(proposed_hull_eval, dtype=tf.float32)
    print(f'proposed time when proposed = {proposed_time}')
    return proposed_time, tf.reshape(proposed_hull_eval, (1,))



class InterpolationPSampler(InterpolationSampler):
  """Piecewise Linear Interpolation for hull formation with Preconditioning

  The SBPS looks at picewise linear regression, but I want to look at how the
  piecewise linear method would work for iterpolation.
  """

  def __init__(self, batch_size=1, data_size=1):
    super().__init__(batch_size, data_size)

  def evaluate_rate(self, target_log_prob_fn, state, velocity,
                    preconditioner, iteration, G, X, time):
    print('IN EVAL RATE, iteration = {}'.format(iteration))
    with tf.name_scope('add_sample_upper_bound'):
      next_state = [s + v * p * time for s, v, p in zip(state, velocity, preconditioner)]
      grads_target_log_prob = target_log_prob_fn(next_state)
      precond_grads = [tf.math.multiply(g, a) for g, a in zip(grads_target_log_prob, preconditioner)]
      # compute the dot product of this with the velocity
      event_rate = compute_dot_prod(velocity, precond_grads)
      G, X, iteration = self.update_envelope_and_time(event_rate, time,
                                                      G, X, iteration)
    return iteration, G, X, event_rate


  def initialise_hull(self, target_log_prob_fn, state, velocity,
                      preconditioner, G, X, iteration):
    """Initialise samples for that will form the hull"""
    # now call the function to evaluate the hull at this loc
    start_time = tf.convert_to_tensor(0.0, dtype=tf.float32)
    dt = tf.convert_to_tensor(1.0, dtype=tf.float32)
    event_zero = self.evaluate_initialise_hull(
      target_log_prob_fn,
      state, velocity, preconditioner,
      0)

    event_end = self.evaluate_initialise_hull(
      target_log_prob_fn,
      state, velocity, preconditioner,
      dt)
    # now see where it intersects the x axis
    # if it intersects x axis for x < 0, then just set the first
    # eval rate to that at event_zero and set first element of X
    # to be zero
    # otherwise, x value to where it intersects and the event rate to be zero
    self.init_slope = (self.gamma * event_end - event_zero) / dt
    self.init_intercept = self.gamma * event_end - self.init_slope * dt
    x_intercept = - self.init_intercept / self.init_slope
    neg_x_intercept_cond = tf.less(x_intercept, 0)

    def x_intercept_negative():
      return G.write(iteration, event_zero), X.write(iteration, tf.reshape([0, 1.0], [1, 2]))

    def x_intercept_positive():
      return G.write(iteration, 0), X.write(iteration, tf.reshape([x_intercept, 1.0], [1, 2]))

    G, X = tf.cond(neg_x_intercept_cond, x_intercept_negative, x_intercept_positive)
    iteration = iteration + 1
    print(f'x_intercept = {x_intercept}')
    print(f'event_zero = {event_zero}')
    print(f'event_end = {event_end}')
    print(f'dt = {dt}')
    print(f'init_slope = {self.init_slope}')
    print(f'init_intercept = {self.init_intercept}')
    return G, X, iteration


  def evaluate_initialise_hull(self, target_log_prob_fn, state, velocity, preconditioner, time):
    with tf.name_scope('eval_init_upper_bound'):
      next_state = [s + v * p * time for s, v, p in zip(state, velocity, preconditioner)]
      print(f'eval_init_hull {next_state[-1]}')
      grads_target_log_prob = target_log_prob_fn(next_state)
      print(f'grads {grads_target_log_prob[-1]}')
      precond_grads = [tf.math.multiply(g, a) for g, a in zip(grads_target_log_prob, preconditioner)]
      print(f'precond {preconditioner[-1]}')
      print(f'precond grads {precond_grads[-1]}')
      # compute the dot product of this with the velocity
      event_rate = compute_dot_prod(velocity, precond_grads)
      return event_rate


  def simulate_bounce_time(self, target_log_prob_fn, state, velocity, preconditioner):
    def accepted_fn(proposal_u, ratio, iteration, proposed_time, linear_time, G, X):
      """function that just determines if the proposal was accepted or not
      based on the `accepted` arg.
      """
      return tf.math.logical_or(tf.math.greater(proposal_u, ratio),
                               tf.math.greater(ratio, 2.0))
    # make sure the iteration index is set to zero
    iteration = tf.convert_to_tensor(0, dtype=tf.int32)
    # initialise linear time and proposed time vars.
    delta_time = tf.convert_to_tensor(0.0, dtype=tf.float32)
    proposed_time = tf.reshape(
      tf.convert_to_tensor(1.0, dtype=tf.float32), ())
    # tensor arrays for holding the gradients and X values used for the linear approx.
    G = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    X = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    # itialise vars. for samples from the proposal distribution and the
    G, X, iteration = self.initialise_hull(target_log_prob_fn, state,
                                           velocity, preconditioner, G, X, iteration)
    # acceptance ratio
    proposal_u = tf.reshape(tf.convert_to_tensor(1.0, dtype=tf.float32), (1,))
    ratio = tf.reshape(tf.convert_to_tensor(0.0, dtype=tf.float32), (1,))
    with tf.name_scope(mcmc_util.make_name('ipp', 'interpolation', 'simulate_bounce')):
      proposal_u, ratio, iteration, proposed_time, delta_time, G, X = tf.while_loop(
        accepted_fn,
        body=lambda proposal_u, ratio, iteration, proposed_time, delta_time, G, X: self.simulate_bounce_loop(
          target_log_prob_fn, state, velocity, preconditioner, proposal_u, ratio, iteration, proposed_time, delta_time, G, X),
        loop_vars=(proposal_u, ratio, iteration, proposed_time, delta_time, G, X),
        maximum_iterations=self.max_iter)
    # if the maximum number of iterations was reached, set the proposed time
    # to be really large so that a refresh will happen at this point
    max_reach = lambda: self.max_time
    valid_proposal = lambda: proposed_time
    bounce_time = tf.cond(tf.math.greater_equal(self.iteration - 1, self.max_iter),
                         max_reach, valid_proposal)
    print('bounce__time = {}'.format(bounce_time))
    return bounce_time, tf.reshape(ratio, ())

  def simulate_bounce_loop(self, target_log_prob_fn, state, velocity,
                           preconditioner, proposal_u, ratio, iteration,
                           proposed_time, delta_time, G, X):
    with tf.name_scope('interpolation_bounce_loop'):
      # get the slope and intercept for the current iteration
      # and the current integrated constant
      slope, intercept = self.get_slope_intercept(G, X, iteration)
      print(f'slope = {slope}')
      print(f'intercept = {intercept}')
      proposed_time, gamma = self.interpolation_propose_time(
        G, X, iteration, slope, intercept)
      # compute the event rate at this proposed time
      iteration, G, X, event_rate = self.evaluate_rate(target_log_prob_fn,
                                                       state,
                                                       velocity,
                                                       preconditioner,
                                                       iteration,
                                                       G, X, proposed_time)
      print('event_rate raw = {}'.format(event_rate))
      event_rate = tf.math.maximum(event_rate, 0.0)
      ratio = event_rate / gamma
      # now make sure that if a Nan or a inf krept in here (very possible)
      # that we attenuate it to zero
      # the is_finite function will find if it is a real number
      ratio = tf.where(~tf.math.is_finite(ratio), tf.zeros_like(ratio), ratio)
      # now draw a uniform sample to see if we accepted this proposal
      proposal_u = self.proposal_uniform.sample(1)
      #print('G_hat = {}'.format(G_hat))
      print('event_rate = {}'.format(event_rate))
      print('gamma = {}'.format(gamma))
      print('proposal_u = {}'.format(proposal_u))
      print('ratio = {}'.format(ratio))
      tf.print('ratio = {}'.format(ratio))
      print('X = {}'.format(X))
      # if proposed_time < 0:
      #   print('found negative event time')
      #   print(proposed_time)
      #   time.sleep(10000)
      refresh_cond  = tf.math.less(proposed_time, 0)
      force_refresh_fn = lambda: (1.0, 10e6)
      accept_fn = lambda: (ratio, proposed_time)
      ratio, proposed_time = tf.cond(refresh_cond, force_refresh_fn, accept_fn)
      return tf.reshape(proposal_u, (1,)), tf.reshape(ratio, (1,)), iteration, tf.reshape(proposed_time, ()), delta_time, G, X


class InterpolationBoomSampler(InterpolationSampler):
  """Piecewise Linear Interpolation for hull formation for boomerang

  The SBPS looks at picewise linear regression, but I want to look at how the
  piecewise linear method would work for iterpolation.
  """
  def __init__(self, batch_size=1, data_size=1):
    super().__init__(batch_size, data_size)
    self.init_slope = 0
    self.init_intercept = 0
    self.min_event = tf.convert_to_tensor(10, dtype=tf.float32)


  def evaluate_rate(self, target_log_prob_fn, dynamics_fn, state, velocity,
                    iteration, G, X, time):
    print('IN EVAL RATE, iteration = {}'.format(iteration))
    with tf.name_scope('add_sample_upper_bound'):
      # next_state = [s + v * time for s, v in zip(state, velocity)]
      x, v = dynamics_fn(state, velocity, time)
      grads_target_log_prob = target_log_prob_fn(x)
      event_rate =  compute_dot_prod(v, grads_target_log_prob)
      event_rate_max = tf.math.maximum(self.min_event, event_rate)
      G, X, iteration = self.update_envelope_and_time(event_rate_max, time,
                                                      G, X, iteration)
    return iteration, G, X, event_rate


  def initialise_hull(self, target_log_prob_fn, state, velocity,
                      dynamics_fn, G, X, iteration):
    """Initialise samples for that will form the hull"""
    # now call the function to evaluate the hull at this loc
    start_time = tf.convert_to_tensor(0.0, dtype=tf.float32)
    dt = tf.convert_to_tensor(1.0, dtype=tf.float32)
    x, v = dynamics_fn(state, velocity, start_time)
    event_zero = self.evaluate_initialise_hull(
      target_log_prob_fn,
      x, v)

    x, v = dynamics_fn(state, velocity, dt)
    event_end = self.evaluate_initialise_hull(
      target_log_prob_fn,
      x, v)
    # now see where it intersects the x axis
    # if it intersects x axis for x < 0, then just set the first
    # eval rate to that at event_zero and set first element of X
    # to be zero
    # otherwise, x value to where it intersects and the event rate to be zero
    self.init_slope = (self.gamma * event_end - self.gamma * event_zero) / dt
    self.init_intercept = self.gamma * event_end - self.init_slope * dt
    x_intercept = - self.init_intercept / self.init_slope
    neg_x_intercept_cond = tf.less(x_intercept, 0)

    def x_intercept_negative():
      # G_ = G.write(iteration, event_zero)
      # X_ = X.write(iteration, 0)

      return G.write(iteration, event_zero), X.write(iteration, tf.reshape([0, 1.0], [1, 2]))

    def x_intercept_positive():
      # G_ = G.write(iteration, 0)
      # X_ = X.write(iteration, x_intercept)
      # return G_, X_
      return G.write(iteration, 0), X.write(iteration, tf.reshape([x_intercept, 1.0], [1, 2]))

    G, X = tf.cond(neg_x_intercept_cond, x_intercept_negative, x_intercept_positive)
    iteration = iteration + 1
    print(f'x_intercept = {x_intercept}')
    print(f'event_zero = {event_zero}')
    print(f'event_end = {event_end}')
    print(f'dt = {dt}')
    print(f'init_slope = {self.init_slope}')
    print(f'init_intercept = {self.init_intercept}')
    return G, X, iteration

  def evaluate_initialise_hull(self, target_log_prob_fn, state, velocity):
    with tf.name_scope('eval_init_upper_bound'):
      grads_target_log_prob = target_log_prob_fn(state)
      event_rate = tf.math.maximum(self.min_event, compute_dot_prod(velocity, grads_target_log_prob))
      return event_rate

  def simulate_bounce_time(self, target_log_prob_fn, state, velocity, dynamics_fn):
    def accepted_fn(proposal_u, ratio, iteration, proposed_time, linear_time, G, X):
      """function that just determines if the proposal was accepted or not
      based on the `accepted` arg.
      """
      tf.print(f'proposal = {proposal_u}', output_stream=sys.stdout)
      tf.print(f'ratio = {ratio}', output_stream=sys.stdout)
      timepy.sleep(0.5)
      return tf.math.logical_or(tf.math.greater(proposal_u, ratio),
                               tf.math.greater(ratio, 2.0))

    # make sure the iteration index is set to zero
    iteration = tf.convert_to_tensor(0, dtype=tf.int32)
    # initialise linear time and proposed time vars.
    delta_time = tf.convert_to_tensor(0.0, dtype=tf.float32)
    proposed_time = tf.reshape(
      tf.convert_to_tensor(1.0, dtype=tf.float32), ())
    # tensor arrays for holding the gradients and X values used for the linear approx.
    G = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    X = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    # itialise vars. for samples from the proposal distribution and the
    G, X, iteration = self.initialise_hull(target_log_prob_fn, state,
                                           velocity, dynamics_fn, G, X, iteration)
    print('here')
    print(G.size())
    print(X.size())

    # acceptance ratio
    proposal_u = tf.reshape(tf.convert_to_tensor(1.0, dtype=tf.float32), (1,))
    ratio = tf.reshape(tf.convert_to_tensor(0.0, dtype=tf.float32), (1,))
    with tf.name_scope(mcmc_util.make_name('ipp', 'interpolation', 'simulate_bounce')):
      proposal_u, ratio, iteration, proposed_time, delta_time, G, X = tf.while_loop(
        accepted_fn,
        body=lambda proposal_u, ratio, iteration, proposed_time, delta_time, G, X: self.simulate_bounce_loop(
          target_log_prob_fn, dynamics_fn, state, velocity, proposal_u, ratio, iteration, proposed_time, delta_time, G, X),
        loop_vars=(proposal_u, ratio, iteration, proposed_time, delta_time, G, X),
        maximum_iterations=self.max_iter)
    # if the maximum number of iterations was reached, set the proposed time
    # to be really large so that a refresh will happen at this point
    max_reach = lambda: self.max_time
    valid_proposal = lambda: proposed_time
    bounce_time = tf.cond(tf.math.greater_equal(self.iteration - 1, self.max_iter),
                         max_reach, valid_proposal)
    print('bounce__time = {}'.format(bounce_time))
    return bounce_time, tf.reshape(ratio, ())


  def simulate_bounce_loop(self, target_log_prob_fn, dynamics_fn, state, velocity,
                           proposal_u, ratio, iteration,
                           proposed_time, delta_time, G, X):
    with tf.name_scope('interpolation_bounce_loop'):
      # get the slope and intercept for the current iteration
      # and the current integrated constant
      slope, intercept = self.get_slope_intercept(G, X, iteration)
      print(f'slope = {slope}')
      print(f'intercept = {intercept}')
      proposed_time, gamma = self.interpolation_propose_time(
        G, X, iteration, slope, intercept)
      # compute the event rate at this proposed time
      iteration, G, X, event_rate = self.evaluate_rate(target_log_prob_fn,
                                                       dynamics_fn,
                                                       state,
                                                       velocity,
                                                       iteration,
                                                       G, X, proposed_time)
      print('event_rate raw = {}'.format(event_rate))
      print('proposed rate = {}'.format(gamma))
      event_rate = tf.math.maximum(event_rate, 0.0)
      ratio = event_rate / gamma
      # now make sure that if a Nan or a inf krept in here (very possible)
      # that we attenuate it to zero
      # the is_finite function will find if it is a real number
      ratio = tf.where(~tf.math.is_finite(ratio), tf.ones_like(ratio), ratio)
      # now draw a uniform sample to see if we accepted this proposal
      proposal_u = self.proposal_uniform.sample(1)
      #print('G_hat = {}'.format(G_hat))
      print('event_rate = {}'.format(event_rate))
      print('gamma = {}'.format(gamma))
      print('proposal_u = {}'.format(proposal_u))
      print('ratio = {}'.format(ratio))
      tf.print('ratio = {}'.format(ratio))
      print('X = {}'.format(X))
      # setting forced refresh conditions
      # if the hull went negative, then our event rate
      # is negative. This means that searching for a valid time
      # would need to explore much larger time for the event rate to go back
      # positive, in which case it would almost surely not be used as the refreshment
      # rate would be smaller.
      # If this happens, will just force the ratio to one and the proposed time
      # to something very large so that it won't be used and will force a refresh event
      refresh_cond  = tf.math.less(proposed_time, 0)
      refresh_cond = tf.logical_or(refresh_cond, ~tf.math.is_finite(proposed_time))
      refresh_cond = tf.logical_or(refresh_cond, tf.math.is_nan(proposed_time))
      force_refresh_fn = lambda: (1.0, 10e6)
      accept_fn = lambda: (ratio, proposed_time)
      ratio, proposed_time = tf.cond(refresh_cond, force_refresh_fn, accept_fn)
      return tf.reshape(proposal_u, (1,)), tf.reshape(ratio, (1,)), iteration, tf.reshape(proposed_time, ()), delta_time, G, X


class ConstantSampler(InterpolationBoomSampler):
  """Piecewise Linear Interpolation for hull formation for boomerang

  The SBPS looks at picewise linear regression, but I want to look at how the
  piecewise linear method would work for iterpolation. Forming the regression
  can be a bit troublesome, but interpolation is much easier.
  """
  def __init__(self, batch_size, data_size):
    super().__init__()
    self.init_slope = 0
    self.init_intercept = 0
    self.gamma = tf.convert_to_tensor(1.0, dtype=tf.float32)
    self.min_event = tf.convert_to_tensor(0.001, dtype=tf.float32)
    self.dt = tf.convert_to_tensor(1.0, dtype=tf.float32)


  def simulate_bounce_time(self, target_log_prob_fn, state, velocity, dynamics_fn):
    G = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    X = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    # will just be doing this with super naive bounds
    ratio = 0
    accepted = False
    bounds = 2
    proposed_time = 0
    iteration = 0
    while not accepted:
      # evaluate the rate
      u = self.proposal_uniform.sample(1)
      proposed_time = proposed_time + -tf.math.log(u) / bounds
      # evaluate the rate at this time
      # compute the event rate at this proposed time
      iteration, G, X, event_rate = self.evaluate_rate(target_log_prob_fn,
                                                       dynamics_fn,
                                                       state,
                                                       velocity,
                                                       iteration,
                                                       G, X, proposed_time)
      D = self.proposal_uniform.sample(1)
      ratio = event_rate / bounds
      if ratio > 1.0:
        print('bounds exceeded')
      if D <= ratio:
        return tf.reshape(proposed_time, ()), tf.reshape(ratio, ())
