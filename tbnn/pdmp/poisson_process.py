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
import time

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


  @abc.abstractmethod
  def simulate_bounce_time(self):
    raise NotImplementedError('Abstract method: child class most overwrite')


class IsotropicGaussianSampler(IPPSampler):
  """ Sampler for standard 2D Isotropic Gaussian """

  def __init__(self):
    super(). __init__(exact=True)

  def simulate_bounce_time(self, state_parts, velocity, time):
    """ Simulate bounce time for system with isotropic Gaussian target

    Employs the inversion method to simulate from the IPP for bounce times
    [1, pg. 252]

    L(t) = \int l(t) dt
    E ~ exponential()
    T_{i+1} = T_{i} + L^{-1}(E + L(T_{i}))

    Inversion can be performed analytically for simple Gaussian system,
    since l(t) = <v, d/dx{-ln(pi(x))}>
    This evolves oout to a quadratic form in terms of our parameters,
    which we can then invert to find the quadratic form of t.

    Args:
      state_parts (list(Tensors)):
        current state of system
      velocity (list(Tensors)):
        current velocity
      time (float):
        current time at step (i)

    Returns:
      Simulated first arrival/bounce from the IPP

    #### References:
    [1] Devroye, Luc. "Nonuniform random variate generation."
      Handbooks in operations research and management science 13 (2006): 83-121.
    """
    exp_random = self.exp_dist.sample(1)
    int_rate = self.integrated_rate(state_parts, velocity, 0.0)
    # apply the inverse fn
    tau = exp_random#int_rate + exp_random
    # now computing the inverse, which is basically from the quadratic formula
    # doing it in multiple bits so it doesnt look gross
    # TODO: Tidy this up
    neg_b = -1. * compute_dot_prod(velocity, state_parts)
    b_squared = 0.0#tf.math.square(neg_b)
    two_a = compute_dot_prod(velocity, velocity)
    four_ac = 2. * two_a * tau
    delta_time = (neg_b + tf.math.sqrt(b_squared + four_ac)) / two_a
    bounce_time = delta_time
    print('int_rate = {}'.format(int_rate))
    print('tau = {}'.format(tau))
    print('b_squared = {}'.format(b_squared))
    print('four_ac = {}'.format(four_ac))
    print('delta time = {}'.format(delta_time))
    print('bounce time = {}'.format(bounce_time))
    bounce_max = tf.maximum(0, bounce_time)
    print('bounce max = {}'.format(bounce_max))
    return bounce_max


  def integrated_rate(self, state, velocity, time):
    # int_rate = -self.compute_dot_grad_velocity(state, velocity) / self.compute_dot_grad_velocity(velocity, velocity)
    int_rate = (0.5 * compute_dot_prod(velocity, velocity) * time**2. +
                compute_dot_prod(state, velocity) * time)# +
                #1.5 * compute_dot_grad_velocity(state, state))
    return int_rate


class SBPSampler(IPPSampler):
  """Implements probabilistic linear from SBPS paper

    #### References:
    [1] Pakman, Ari, et al. "Stochastic bouncy particle sampler." ICML. 2017.
  """

  def __init__(self, batch_size=1, data_size=1):
    super().__init__(exact=False)
    self.batch_size = tf.convert_to_tensor(batch_size, dtype=tf.float32)
    self.data_size = tf.convert_to_tensor(data_size, dtype=tf.float32)
    self.max_iter = tf.convert_to_tensor(100, dtype=tf.int32)
    self.G = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    self.X = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    #self.k = tf.convert_to_tensor(0.0000001, dtype=tf.float64)
    self.k = tf.convert_to_tensor(1.0, dtype=tf.float64)
    # initialise the time variable used to approximate the linear function of the
    # upper bound
    self.num_iters = tf.constant(0)
    self.max_time = tf.constant(50.0, dtype=tf.float32)#, trainable=False)
    self.time_dt = tf.constant(0.10, dtype=tf.float32)
    self.linear_var_dist = tfd.Normal(loc=0, scale=1)
    self.proposal_uniform = tfd.Uniform()
    self.rho = tf.constant(1.0, dtype=tf.float32)#, trainable=False)
    self.b_0 = tf.reshape(tf.constant([1., 1.], dtype=tf.float32), [2,1])
    self.b_1_prior = tf.reshape(tf.constant(1.0, dtype=tf.float32), (1,))
    self.beta_prior_cov = tf.eye(2, dtype=tf.float32) * 1e3
    self.beta_prior_prec = tf.linalg.inv(self.beta_prior_cov)
    self.inv_adjust = 0.0001 * tf.eye(2, dtype=tf.float32)
    self.likelihood_var = tf.convert_to_tensor(1.0, dtype=tf.float32)
    self.likelihood_prec = 1.0 / self.likelihood_var
    self.iteration = tf.constant(0, dtype=tf.int32)#tf.Variable(0, dtype=tf.int32)#, trainable=False)
    self.global_step = tf.constant(0, dtype=tf.int64)#, dtype=tf.int64)# tf.Variable(0, dtype=tf.int32)#, trainable=False)
    # adding steps for init method
    self.gamma = tf.convert_to_tensor(1.0, dtype=tf.float32)
    print('am in BPS')

  def simulate_bounce_time(self, target_log_prob_fn, state, velocity):
    """initialises the `tf.while_loop` for finding a proposed bounce time"""
    def accepted_fn(proposal_u, ratio, iteration, proposed_time, linear_time, G, X, int_constant):
      """function that just determines if the proposal was accepted or not
      based on the `accepted` arg.
      """
      #return tf.math.logical_or(tf.math.greater(proposal_u, ratio), tf.math.greater(ratio, 1.1))
      return tf.math.greater(proposal_u, ratio)
      #return tf.math.greater(proposal_u, ratio)
    # make sure the iteration index is set to zero
    iteration = tf.convert_to_tensor(0, dtype=tf.int32)
    # initialise linear time and proposed time vars.
    int_constant = tf.constant(0.0, dtype=tf.float32)#, trainable=False)
    linear_time = tf.convert_to_tensor(0.0, dtype=tf.float32)
    proposed_time = tf.reshape(
      tf.convert_to_tensor(1.0, dtype=tf.float32), ())
    # tensor arrays for holding the gradients and X values used for the linear approx.
    G = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    X = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    iteration, G, X = self.initialise_rate_samples(target_log_prob_fn, state,
                                                   velocity,
                                                   iteration, G, X)
    # itialise vars. for samples from the proposal distribution and the
    # acceptance ratio
    proposal_u = tf.reshape(tf.convert_to_tensor(1.0, dtype=tf.float32), (1,))
    ratio = tf.reshape(tf.convert_to_tensor(0.0, dtype=tf.float32), (1,))
    with tf.name_scope(mcmc_util.make_name('ippppp', 'sbps', 'simulate_bounce')):
      proposal_u, ratio, iteration, proposed_time, linear_time, G, X, int_constant = tf.while_loop(
        accepted_fn,
        body=lambda proposal_u, ratio, iteration, proposed_time, linear_time, G, X, int_constant: self.simulate_bounce_loop(
          target_log_prob_fn, state, velocity, proposal_u, ratio, iteration, proposed_time, linear_time, G, X, int_constant),
        loop_vars=(proposal_u, ratio, iteration, proposed_time, linear_time, G, X, int_constant),
        maximum_iterations=self.max_iter)
    # if the maximum number of iterations was reached, set the proposed time
    # to be really large so that a refresh will happen at this point
    print('Proposed bounce_time = {}'.format(proposed_time))
    print('Proposed ratio = {}, iters = {}'.format(ratio, iteration))
    max_reach = lambda: self.max_time
    valid_proposal = lambda: proposed_time
    bounce_time = tf.cond(tf.math.greater_equal(self.iteration - 1, self.max_iter),
                         max_reach, valid_proposal)
    print('Returned bounce_time = {}'.format(bounce_time))
    return bounce_time, tf.reshape(ratio, ())


  def initialise_rate_samples(self, target_log_prob_fn, state, velocity,
                              iteration, G, X):
    """Initialise samples for that will form the hull"""
    def for_loop_cond(iteration, G, X):
      """function to tell us to exit this while loop for initialising the
      hulls, which is really just a for loop
      If our iteration value is less than the number of samples within our
      init array, then keep on going
      """
      return tf.math.less(iteration, self.num_init)

    def initialise_rate_loop(target_log_prob_fn, state, velocity,
                             iteration, G, X):
      """Loop function to initialise the samples of the rate that
      we will use to form our Hull
      """
      # get the time we want for this sample
      time = tf.gather(self.init_envelope, iteration)
      print('Time = {}'.format(time))
      # now call the function to evaluate the hull at this loc and add it
      iteration, G, X, _ = self.evaluate_rate(target_log_prob_fn,
                                              state, velocity,
                                              iteration, G, X, time)
      return iteration, G, X

    with tf.name_scope('initialise_upper_bound'):
      iteration, G, X = tf.while_loop(
        for_loop_cond,
        body=lambda iteration, G, X: initialise_rate_loop(
          target_log_prob_fn, state, velocity, iteration, G, X),
        loop_vars=(iteration, G, X),
        maximum_iterations=self.num_init)
      return iteration, G, X


  def evaluate_rate(self, target_log_prob_fn, state, velocity,
                    iteration, G, X, time):
    print('IN EVAL RATE, iteration = {}'.format(iteration))
    with tf.name_scope('add_sample_upper_bound'):
      next_state = [s + v * time for s, v in zip(state, velocity)]
      grads_target_log_prob = target_log_prob_fn(next_state)
      event_rate = compute_dot_prod(velocity, grads_target_log_prob)
      G, X, iteration = self.update_envelope_and_time(event_rate, time,
                                                      G, X, iteration)
    return iteration, G, X, event_rate


  def simulate_bounce_loop(self, target_log_prob_fn, state, velocity,
                           proposal_u, ratio, iteration,
                           proposed_time, linear_time, G, X, int_constant):
    with tf.name_scope('sbps_bounce_loop'):

      beta_mean, beta_cov = self.sbps_beta_posterior(G, X, iteration)
      proposed_time, gamma, int_constant = self.sbps_propose_time_posterior(X, beta_mean,
                                                                            beta_cov, int_constant)
      # compute the gradient at this proposed time
      iteration, G, X, event_rate = self.evaluate_rate(target_log_prob_fn,
                                                       state,
                                                       velocity,
                                                       iteration,
                                                       G, X, proposed_time)
      # now compare true rate with envelope rate
      ratio = event_rate / gamma
      # now make sure that if a Nan or a inf krept in here (very possible)
      # that we attenuate it to zero
      # the is_finite function will find if it is a real number
      ratio = tf.where(~tf.math.is_finite(ratio), tf.zeros_like(ratio), ratio)
      # now draw a uniform sample to see if we accepted this proposal
      proposal_u = self.proposal_uniform.sample(1)
      #print('G_hat = {}'.format(G_hat))
      print('gamma = {}'.format(gamma))
      print('proposal_u = {}'.format(proposal_u))
      print('ratio = {}'.format(ratio))
      return proposal_u, ratio, iteration, tf.reshape(proposed_time, ()), linear_time, G, X, int_constant


  def sbps_beta_posterior(self, G, X, iteration,
                          sigma=1, tau=1.0, b_0=1.0):
    """get posterior for beta

    Using Normal prior on beta, and following [1, pg. 232]
    p(beta) ~ N(beta | b_0, tau^2 * I)
    p(y | beta) ~ N(X * beta, sigma^2 * I)

    Then the posterior
    p(beta | y) = p(y | beta) * p(beta) / p(y)

    =
    N(beta | beta_n, V_n)
    V_n = sigma^2 * (sigma^2 / tau^2 + X^T * X)^{-1}
    b_n = V_n * (1/tau^2) * b_0 + (1/sigma^2) * V_n * X^T * y
    """
    with tf.name_scope('sbps_beta_posterior'):
      # get G in vector form from the TensorArray
      G_vector = tf.maximum(tf.reshape(G.stack(), shape=[-1, 1]), self.gamma)
      print('G = {}'.format(G))
      # need to solve to get beta
      # following papers terms where
      # x = [t, 1]
      # beta = [beta_1, beta_0]^T
      # get x_time in matrix form from the TensorArray
      x_time = tf.reshape(X.stack(), [-1, 2])
      print('X_time = {}'.format(x_time))
      print('X^TX = {}'.format(tf.transpose(x_time) @ x_time + 0.001 * tf.eye(2)))
      print(self.b_1_prior.shape)
      # setting the second value of beta prior to be that of beta, so that it
      # anything in the prior eval won't contribute
      # beta_prior_mu = tf.reshape(
      #   tf.stack([self.b_1_prior, tf.gather(beta, 1, axis=0)]), (2, 1))
      #
      #beta_cov = tau**2.0 * tf.linalg.inv(0.001 * tf.eye(2) + tf.transpose(x_time) @ x_time)
      beta_cov = tf.linalg.inv(
        self.beta_prior_prec + self.likelihood_prec * tf.transpose(x_time)@ x_time)
      print('G_vector = {}'.format(G_vector))
      print('beta_cov = {}'.format(beta_cov))
      beta_mean = beta_cov @ self.beta_prior_prec @ self.b_0 + self.likelihood_prec * beta_cov @ tf.transpose(x_time)  @ G_vector
      print('beta_cov = {}'.format(beta_cov))
      print('beta_mean = {}'.format(beta_mean))
      return beta_mean, beta_cov


  def sbps_update_linear(self, grads_target_log_prob, velocity, G, X, linear_time, iteration):
    """updates the linear model"""
    #G.append(self.compute_dot_grad_velocity(grads_target_log_prob, velocity))
    with tf.name_scope('sbps_update_linear'):
      print('iteration = {}'.format(iteration))
      print(
        'nans in gradient = {}'.format(
          tf.reduce_sum(
            [tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.float32)) for x in grads_target_log_prob])))
      print(
        'nans in velocity = {}'.format(
          tf.reduce_sum(
            [tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.float32)) for x in velocity])))
      G = G.write(iteration,
                  compute_dot_prod(grads_target_log_prob, velocity))
      G_vector = tf.reshape(G.stack(), shape=[-1, 1])
      print('nans in G_vector = {}'.format(tf.reduce_sum(tf.cast(tf.math.is_nan(G_vector), tf.float32))))
      # need to solve to get beta
      # following papers terms where
      # x = [t, 1]
      # beta = [beta_1, beta_0]^T
      X = X.write(iteration,
                  tf.reshape([linear_time, 1.0], [1, 2]))
      #X.append(tf.reshape(tf.Variable([time, 1], dtype=tf.float32), [1, 2]))
      #x_time = tf.concat(X, axis=0)
      x_time = tf.reshape(X.stack(), [-1, 2])
      print('x_time = {}'.format(x_time))
      print('x_time.shape = {}'.format(x_time.shape))
      print('x_time^T @ x_time = {}'.format(tf.transpose(x_time) @ x_time))
      beta = tf.linalg.inv(tf.transpose(x_time) @ x_time + 0000.1 * tf.eye(2)) @ tf.transpose(x_time) @ G_vector
      return G, X, beta


  def update_envelope_and_time(self, envelope_value, time, G, X, iteration):
    # make sure time is correct shape here
    time = tf.reshape(time, ())
    G = G.write(iteration, envelope_value)
    X = X.write(iteration, tf.reshape([time, 1.0], [1, 2]))
    iteration = iteration + 1
    return G, X, iteration


  def sbps_propose_time_posterior(self, X, beta_mean, beta_cov, constant):
    with tf.device('/CPU:0'):
      x_time = tf.reshape(X.stack(), [-1, 2])
      last_time = x_time[-1, 0]
      #x = tf.reshape(X.read(iteration - 1), (1, 2))
      #print('beta_mean = {}'.format(beta_mean))
      #print('time proop = {}'.format(time))
      # can now apply inversion method to get the new suggested time
      beta_one = beta_mean[0, 0]
      beta_zero = beta_mean[1, 0]
      x = tf.reshape(x_time[-1, :], (1, 2))
      # additional term in eq 13, k * rho(t)
      # rho(t) = x^T @ Sigma @ x + c^2
      # where c^2 is given by the linear_var argument
      intercept = tf.cast(beta_zero, tf.float64)
      slope = tf.cast(beta_one, tf.float64)
      beta_cov = tf.cast(beta_cov, tf.float64)
      likelihood_var = tf.cast(self.likelihood_var, tf.float64)
      x = tf.cast(x, dtype=tf.float64)
      # print('beta_zero = {}'.format(beta_zero))
      # print('x = {}'.format(x))
      # print('shape x = {}'.format(x.shape))
      # print('beta_cov = {}'.format(beta_cov))
      # additional_term = self.k * tf.reshape(x @ beta_cov @ tf.transpose(x) + likelihood_var, beta_zero.shape)
      # print('beta_zero before = {}'.format(beta_zero))
      # print('additional_term = {}'.format(additional_term))
      # # beta_zero = beta_zero + additional_term
      # print('beta_one = {}'.format(beta_one))
      # print('beta_zero = {}'.format(beta_zero))
      # print('rho = {}'.format(self.rho))
      # exp_random = self.exp_dist.sample(1)
      # # apply the inverse fn
      # # tau = exp_random + prev_time
      # print('tau = {}'.format(tau))
      # # now computing the inverse, which is basically from the quadratic formula
      # doing it in multiple bits so it doesnt look gross
      # neg_b = -1. * (beta_zero + k * rho)
      # sqrt_term = 2. * (tau * beta_one)
      # two_a = beta_one
      # delta_time = (neg_b + tf.math.sqrt(sqrt_term)) / two_a
      constant = tf.cast(constant, tf.float64)
      # prev_time = tf.cast(prev_time, tf.float64)
      # tau = tf.cast(tau, tf.float64)
      # neg_b = tf.cast(-1. * beta_zero, dtype=tf.float64)
      # b_squared = tf.cast(beta_zero**2., dtype=tf.float64)
      # # neg_four_ac = tf.cast(2. * (tau * beta_one), dtype=tf.float64)
      # neg_four_ac = tf.cast(2. * beta_one *  (tau - constant), dtype=tf.float64)
      # two_a = tf.cast(beta_one, dtype=tf.float64)
      # delta_time = (neg_b + tf.math.sqrt(b_squared + neg_four_ac)) / two_a
      # bounce_time = delta_time
      # #print('here')
      # #print(bounce_time)
      # print('neg_b = {}'.format(neg_b))
      # print('b^2 = {}'.format(b_squared))
      # print('neg_four_ac = {}'.format(neg_four_ac))
      # print('delta time = {}'.format(delta_time))
      # print('two_a = {}'.format(two_a))
      # print('proposed bounce time = {}'.format(bounce_time))
      # print('sqrt = {}'.format(tf.math.sqrt(b_squared + neg_four_ac)))
      # print('numerator = {}'.format(neg_b + tf.math.sqrt(b_squared + neg_four_ac)))
      # bounce_max = prev_time + tf.maximum(tf.convert_to_tensor(0.0, dtype=tf.float64), bounce_time)
      last_time = tf.cast(last_time, dtype=tf.float64)
      min_slope = tf.cast(1e-7, dtype=tf.float64)
      affine_cond = tf.math.greater(tf.math.abs(slope), min_slope)
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
            intercept ** 2.0 + prev_terms - 2 * slope * U)) / (2.0 * slope)
        proposed_hull_eval = slope * proposed_time + intercept
        return proposed_time, proposed_hull_eval

      def constant_segment():
        u = 1.0 - tf.random.uniform([1], dtype=tf.float64)
        U = tf.math.log(u)
        print('constant segment')
        proposed_time = -U / intercept + last_time
        proposed_hull_eval = intercept
        return proposed_time, proposed_hull_eval

      proposed_time, proposed_hull_eval = tf.cond(affine_cond, affine_segment, constant_segment)



      # return the constant value
      # add the new integral from current starting point to the new one
      # int_const = (0.5 * beta_one * bounce_max ** 2.0 + beta_zero * bounce_max) - (0.5 * beta_one * prev_time ** 2.0 + beta_zero * prev_time)
      # new_constant = tf.cast(constant, tf.float64) + int_const
      # evaluate this upper bound at this proposed time
      # beta zero here now includes the additional term
      # bounce_eval = beta_one * bounce_max + beta_zero
      #print('bounce max = {}'.format(bounce_max))
      # return tf.cast(bounce_max, dtype=tf.float32), tf.cast(bounce_eval, dtype=tf.float32), tf.reshape(tf.cast(int_const, dtype=tf.float32), ())
      return tf.reshape(tf.cast(proposed_time, dtype=tf.float32), (1,)), tf.reshape(tf.cast(proposed_hull_eval, dtype=tf.float32), (1,)), tf.reshape(tf.cast(0.0, dtype=tf.float32), ())


  def sbps_integrated_rate(self, linear_time, beta_one, beta_zero):#, time, beta_one, beta_zero):
    print('time = {}'.format(linear_time))
    beta_zero_adj = beta_zero + self.k * self.rho
    return 0.5* beta_one * linear_time**2.0 + beta_zero_adj * linear_time


class PSBPSampler(SBPSampler):
  """Pretty much the same deal as the SBPSampler, but included preconditioner
  term

    #### References:
    [1] Pakman, Ari, et al. "Stochastic bouncy particle sampler." ICML. 2017.
  """

  def __init__(self, batch_size=1, data_size=1):
    super().__init__(batch_size, data_size)


  def simulate_bounce_time(self, target_log_prob_fn, state, velocity,
                           preconditioner):
    def accepted_fn(proposal_u, ratio, iteration, proposed_time, linear_time, G, X):
      """function that just determines if the proposal was accepted or not
      based on the `accepted` arg.
      """
      return tf.math.greater(proposal_u, ratio)
    # make sure the iteration index is set to zero
    iteration = tf.convert_to_tensor(0, dtype=tf.int32)
    # initialise linear time and proposed time vars.
    linear_time = tf.convert_to_tensor(0.0, dtype=tf.float32)
    proposed_time = tf.reshape(
      tf.convert_to_tensor(1.0, dtype=tf.float32), ())
    # tensor arrays for holding the gradients and X values used for the linear approx.
    G = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    X = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    # itialise vars. for samples from the proposal distribution and the
    iteration, G, X = self.initialise_rate_samples(target_log_prob_fn, state,
                                                   velocity, preconditioner,
                                                   iteration, G, X)
    # acceptance ratio
    proposal_u = tf.reshape(tf.convert_to_tensor(1.0, dtype=tf.float32), (1,))
    ratio = tf.reshape(tf.convert_to_tensor(0.0, dtype=tf.float32), (1,))
    with tf.name_scope(mcmc_util.make_name('ippppp', 'sbps', 'simulate_bounce')):
      proposal_u, ratio, iteration, proposed_time, linear_time, G, X = tf.while_loop(
        accepted_fn,
        body=lambda proposal_u, ratio, iteration, proposed_time, linear_time, G, X: self.simulate_bounce_loop(
          target_log_prob_fn, state, velocity, preconditioner, proposal_u, ratio, iteration, proposed_time, linear_time, G, X),
        loop_vars=(proposal_u, ratio, iteration, proposed_time, linear_time, G, X),
        maximum_iterations=self.max_iter)
    # if the maximum number of iterations was reached, set the proposed time
    # to be really large so that a refresh will happen at this point
    max_reach = lambda: self.max_time
    valid_proposal = lambda: proposed_time
    bounce_time = tf.cond(tf.math.greater_equal(self.iteration - 1, self.max_iter),
                         max_reach, valid_proposal)
    print('bounce__time = {}'.format(bounce_time))
    return bounce_time, tf.reshape(ratio, ())


  def initialise_rate_samples(self, target_log_prob_fn, state, velocity,
                              preconditioner, iteration, G, X):
    """Initialise samples for that will form the hull"""
    def for_loop_cond(iteration, G, X):
      """function to tell us to exit this while loop for initialising the
      hulls, which is really just a for loop
      If our iteration value is less than the number of samples within our
      init array, then keep on going
      """
      return tf.math.less(iteration, self.num_init)


    def initialise_rate_loop(target_log_prob_fn, state, velocity,
                             preconditioner, iteration, G, X):
      """Loop function to initialise the samples of the rate that
      we will use to form our Hull
      """
      # get the time we want for this sample
      time = tf.gather(self.init_envelope, iteration)
      print('Time = {}'.format(time))
      # now call the function to evaluate the hull at this loc and add it
      iteration, G, X, _ = self.evaluate_rate(target_log_prob_fn,
                                              state, velocity,
                                              preconditioner, iteration,
                                              G, X, time)
      return iteration, G, X

    with tf.name_scope('initialise_upper_bound'):
      iteration, G, X = tf.while_loop(
        for_loop_cond,
        body=lambda iteration, G, X: initialise_rate_loop(
          target_log_prob_fn, state, velocity, preconditioner, iteration, G, X),
        loop_vars=(iteration, G, X),
        maximum_iterations=self.num_init)
      return iteration, G, X


  def evaluate_rate(self, target_log_prob_fn, state, velocity,
                    preconditioner, iteration, G, X, time):
    print('IN EVAL RATE, iteration = {}'.format(iteration))
    with tf.name_scope('add_sample_upper_bound'):
      #print('eval state = {}'.format(state))
      #print('eval velocity = {}'.format(velocity))
      #print('eval preconditioner = {}'.format(preconditioner))
      #next_state = [s + v * a * time for s, v, a in zip(
      #  state, velocity, preconditioner)]
      next_state = [s + v * time for s, v in zip(state, velocity)]
      grads_target_log_prob = target_log_prob_fn(next_state)
      precond_grads = [tf.math.multiply(g, a) for g, a in zip(grads_target_log_prob, preconditioner)]
      # compute the dot product of this with the velocity
      event_rate = tf.math.maximum(
        compute_dot_prod(velocity, precond_grads), self.gamma)
      G, X, iteration = self.update_envelope_and_time(event_rate, time,
                                                      G, X, iteration)
    return iteration, G, X, event_rate


  def simulate_bounce_loop(self, target_log_prob_fn, state, velocity,
                           preconditioner, proposal_u, ratio, iteration,
                           proposed_time, linear_time, G, X):
    with tf.name_scope('sbps_bounce_loop'):

      beta_mean, beta_cov = self.sbps_beta_posterior(G, X, iteration)
      proposed_time, gamma, _ = self.sbps_propose_time_posterior(X, beta_mean,
                                                              beta_cov, 0)
      # compute the gradient at this proposed time
      iteration, G, X, event_rate = self.evaluate_rate(target_log_prob_fn,
                                                       state,
                                                       velocity,
                                                       preconditioner,
                                                       iteration,
                                                       G, X, proposed_time)
      # now compare true rate with envelope rate
      ratio = event_rate / gamma
      # now make sure that if a Nan or a inf krept in here (very possible)
      # that we attenuate it to zero
      # the is_finite function will find if it is a real number
      ratio = tf.where(~tf.math.is_finite(ratio), tf.zeros_like(ratio), ratio)
      # now draw a uniform sample to see if we accepted this proposal
      proposal_u = self.proposal_uniform.sample(1)
      #print('G_hat = {}'.format(G_hat))
      print('gamma = {}'.format(gamma))
      print('proposal_u = {}'.format(proposal_u))
      print('ratio = {}'.format(ratio))
      return proposal_u, ratio, iteration, tf.reshape(proposed_time, ()), linear_time, G, X



class AdaptiveSBPSampler(SBPSampler):
  """Perform sampling with the adaptive upper bound method proposed by Gilks

  Not for preconditioned models
    #### References:.
  """

  def __init__(self, batch_size=1, data_size=1):
    super().__init__(batch_size=batch_size, data_size=data_size)
    # need initial time points to initialise the envelope
    # self.adjust = tf.convert_to_tensor(1.25, dtype=tf.float32)
    self.init_envelope = np.array([0.0, 1.0, 100.0]).astype(np.float32)
    self.num_init = tf.convert_to_tensor(self.init_envelope.size)
    self.adjust = tf.convert_to_tensor(1.0, dtype=tf.float32)
    self.random_time_dist = tfd.Exponential(1.)
    # self.epsilon = tf.convert_to_tensor(0.000000001, dtype=tf.float32)
    self.gamma = tf.convert_to_tensor(0.1, dtype=tf.float32)
    self.alpha = tf.convert_to_tensor(1.1, dtype=tf.float32)
    # am creating a large time that will force a refresh if needed
    self.force_refresh_time = tf.convert_to_tensor(100000000, dtype=tf.float32)
    print('am in adaptive Poisson Process')



  def simulate_bounce_time(self, target_log_prob_fn, state, velocity):
    """simulate loop for our bounce time

    This is much the same as out PSBPS method, but first we need to iniialise
    our upper bound with a few values
    """
    def accepted_fn(proposal_u, ratio, iteration, proposed_time, G, X):
      """function that just determines if the proposal was accepted or not
      based on the `accepted` arg.
      """
      # return tf.math.logical_or(tf.math.greater(proposal_u, ratio), tf.math.greater(ratio, 2))
      return tf.math.greater(proposal_u, ratio)
    # make sure the iteration index is set to zero
    iteration = tf.convert_to_tensor(0, dtype=tf.int32)
    # initialise linear time and proposed time vars.
    linear_time = tf.convert_to_tensor(0.0, dtype=tf.float32)
    proposed_time = tf.reshape(
      tf.convert_to_tensor(1.0, dtype=tf.float32), ())
    # tensor arrays for holding the gradients and X values used for the linear approx.
    G = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    X = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    # add our initial values to our upper bound envelope
    proposal_u = tf.reshape(tf.convert_to_tensor(1.0, dtype=tf.float32), (1,))
    ratio = tf.reshape(tf.convert_to_tensor(0.0, dtype=tf.float32), (1,))
    # initialise the upper bound
    iteration, G, X = self.initialise_rate_samples(target_log_prob_fn, state,
                                                   velocity, iteration, G, X)
    print('G = {}'.format(G.stack()))
    print('X = {}'.format(X.stack()))
    # now continue sampling until we have a valid proposed time
    # (of have exceeded the maximum number of iterations)
    with tf.name_scope(mcmc_util.make_name('adaptive', 'sbps', 'simulate_bounce')):
      proposal_u, ratio, iteration, proposed_time, G, X = tf.while_loop(
        accepted_fn,
        body=lambda proposal_u, ratio, iteration, proposed_time, G, X: self.simulate_bounce_loop(
          target_log_prob_fn, state, velocity, proposal_u, ratio, iteration, proposed_time, G, X),
        loop_vars=(proposal_u, ratio, iteration, proposed_time, G, X),
        maximum_iterations=self.max_iter)
    # perform some checks on our times to make sure they are within range
    max_reach = lambda: self.max_time
    valid_proposal = lambda: proposed_time
    bounce_time = tf.cond(tf.math.greater_equal(self.iteration - 1, self.max_iter),
                         max_reach, valid_proposal)
    print('Proposed bounce_time = {}'.format(proposed_time))
    print('Proposed ratio = {}'.format(ratio))
    print('proposed__time = {}, bounce__time = {}'.format(proposed_time, bounce_time))
    return bounce_time, tf.reshape(ratio, ())


  def evaluate_rate(self, target_log_prob_fn, state, velocity,
                    iteration, G, X, time):
    print('IN EVAL RATE, iteration = {}'.format(iteration))
    with tf.name_scope('add_sample_upper_bound'):
      #print('eval state = {}'.format(state))
      #print('eval velocity = {}'.format(velocity))
      #print('eval preconditioner = {}'.format(preconditioner))
      next_state = [s + v * time for s, v in zip(state, velocity)]
      grads_target_log_prob = target_log_prob_fn(next_state)
      # compute the dot product of this with the velocity
      event_rate = tf.math.maximum(
        compute_dot_prod(velocity, grads_target_log_prob), self.gamma)
      G = G.write(iteration, event_rate * self.adjust)
      X = X.write(iteration, time)
      iteration = iteration + 1
      print(
        'event_rate = {}, event_time = {}, adjust = {}, adjusted = {}'.format(
          event_rate, time, self.adjust, event_rate * self.adjust))
    return iteration, G, X, event_rate


  def simulate_bounce_loop(self, target_log_prob_fn, state, velocity,
                           proposal_u, ratio, iteration,
                           proposed_time, G, X):
    with tf.name_scope('adaptive_psbps_bounce_loop'):
      print('iteration = {}'.format(iteration))
      # Sample a random time
      with tf.device('/CPU:0'):
        (force_refresh,
         hull_sample_time,
         hull_sample_val) = sample_poisson_thinning(G.stack(),
                                                    X.stack())
      # end running on cpu
      print(f'hull time {hull_sample_time}, hull eval {hull_sample_val}')
      print(f'iteration {iteration}')
      # if the sample was valid, let's go ahead and evaluate it.
      # otherwise, let's avoid computing any more expensive gradients
      # and just tell our kernel that we should do a refresh.
      force_refresh_fn = lambda : self.force_refresh_fn(target_log_prob_fn,
                           hull_sample_val,
                           state,
                           velocity,
                           iteration, G, X,
                           hull_sample_time)
      evaluate_proposal_fn = lambda : self.evaluate_proposal_fn(target_log_prob_fn,
                           hull_sample_val,
                           state,
                           velocity,
                           iteration, G, X,
                           hull_sample_time)
    (proposal_u, ratio,
     iteration, hull_sample_time,
     G, X) = tf.cond(force_refresh, force_refresh_fn, evaluate_proposal_fn)
    return proposal_u, ratio, iteration, hull_sample_time, G, X

  def force_refresh_fn(self, target_log_prob_fn,
                       hull_sample_val,
                       state,
                       velocity,
                       iteration, G, X,
                       hull_sample_time):
    print(f'iteration refresh {iteration}')
    # to force exit the loop, we can return a proposal value
    # and ratio such that
    #   tf.math.greater(proposal_u, ratio)
    # evaluates to false. This will ensure the while loop above that calls
    # this simulate_bounce_loop function will exit.
    proposal_u = tf.reshape(tf.convert_to_tensor(0.0, dtype=tf.float32), (1,))
    print('proposal_u refresh shape = {}'.format(proposal_u.shape))
    ratio = tf.convert_to_tensor(1.0, dtype=tf.float32)
    return proposal_u, tf.reshape(ratio, (1,)), iteration, tf.reshape(self.force_refresh_time, ()), G, X

  def evaluate_proposal_fn(self, target_log_prob_fn,
                           hull_sample_val,
                           state,
                           velocity,
                           iteration, G, X,
                           hull_sample_time):
    # this function should be called if the proposed sample is valid
    # (ie. the event rate is > 0)
    # removing indent, as want this to run on GPU again
    # print(type(state))
    # print(type(G))
    print(iteration)
    iteration, G, X, event_rate = self.evaluate_rate(target_log_prob_fn,
                                                     state,
                                                     velocity,
                                                     iteration, G, X,
                                                     hull_sample_time)
    # compute the ratio between our evaluated rate and the sample from our
    # envelope
    print('eeevent_rate = {}, hull_sample = {}'.format(event_rate, hull_sample_val))
    print('this is event_rate {}, time {}'.format(G.read(iteration - 1),
                                                  hull_sample_time))
    ratio = event_rate / (hull_sample_val + self.gamma)
    # now make sure that if a Nan or a inf krept in here (very possible)
    # that we attenuate it to zero
    # the is_finite function will find if it is a real number
    ratio = tf.where(~tf.math.is_finite(ratio), tf.zeros_like(ratio), ratio)
    # sample our uniform random variable
    proposal_u = self.proposal_uniform.sample(1)
    print('proposal_u shape = {}'.format(proposal_u.shape))
    print('ratio shape = {}'.format(ratio.shape))
    print('hull_sample_val = {}, proposed_val = {}, ratio = {}'.format(hull_sample_val, G.read(iteration -1), ratio))
    return proposal_u, tf.reshape(ratio, (1,)), iteration, tf.reshape(hull_sample_time, ()), G, X



class AdaptivePSBPSampler(PSBPSampler):
  """Perform sampling with the adaptive upper bound method proposed by Gilks

  For preconditioned models
    #### References:.
  """

  def __init__(self, batch_size, data_size):
    super().__init__(batch_size=1, data_size=1)
    # need initial time points to initialise the envelope
    self.adjust = tf.convert_to_tensor(1.00, dtype=tf.float32)
    self.random_time_dist = tfd.Exponential(1.)
    self.gamma = tf.convert_to_tensor(0.000000001, dtype=tf.float32)
    self.force_refresh_time = tf.convert_to_tensor(100000000, dtype=tf.float32)
    self.init_envelope = np.array([0.0, 1.0, 100.0]).astype(np.float32)


  def evaluate_rate(self, target_log_prob_fn, state, velocity,
                    preconditioner, iteration, G, X, time):
    print('IN EVAL RATE, iteration = {}'.format(iteration))
    with tf.name_scope('add_sample_upper_bound'):
      next_state = [s + v * time for s, v in zip(state, velocity)]
      #next_state = [s + v * a * time for s, v, a in zip(
      #  state, velocity, preconditioner)]
      grads_target_log_prob = target_log_prob_fn(next_state)
      # get the preconditioned gradients
      precond_grads = [tf.math.multiply(g, a) for g, a in zip(grads_target_log_prob, preconditioner)]
      # compute the dot product of this with the velocity
      event_rate = tf.math.maximum(
        compute_dot_prod(velocity, precond_grads), self.gamma)
      G = G.write(iteration, event_rate * self.adjust)
      X = X.write(iteration, time)
      iteration = iteration + 1
      return iteration, G, X, event_rate


  def simulate_bounce_time(self, target_log_prob_fn, state, velocity,
                           preconditioner):
    """simulate loop for our bounce time


    This is much the same as out PSBPS method, but first we need to iniialise
    our upper bound with a few values
    """
    def accepted_fn(proposal_u, ratio, iteration, proposed_time, G, X):
      """function that just determines if the proposal was accepted or not
      based on the `accepted` arg.
      """
      #return tf.math.greater(proposal_u, ratio)
      return tf.math.greater(proposal_u, ratio)
      # return tf.math.logical_or(tf.math.greater(proposal_u, ratio), tf.math.greater(ratio, 1.1))
    # make sure the iteration index is set to zero
    iteration = tf.convert_to_tensor(0, dtype=tf.int32)
    # initialise linear time and proposed time vars.
    linear_time = tf.convert_to_tensor(0.0, dtype=tf.float32)
    proposed_time = tf.reshape(
      tf.convert_to_tensor(1.0, dtype=tf.float32), ())
    # tensor arrays for holding the gradients and X values used for the linear approx.
    G = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    X = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    # add our initial values to our upper bound envelope
    proposal_u = tf.reshape(tf.convert_to_tensor(1.0, dtype=tf.float32), (1,))
    ratio = tf.reshape(tf.convert_to_tensor(0.0, dtype=tf.float32), (1,))
    # initialise the upper bound
    iteration, G, X = self.initialise_rate_samples(target_log_prob_fn, state,
                                                   velocity, preconditioner,
                                                   iteration, G, X)
    print('G = {}'.format(G.stack()))
    print('X = {}'.format(X.stack()))
    # now continue sampling until we have a valid proposed time
    # (of have exceeded the maximum number of iterations)
    with tf.name_scope(mcmc_util.make_name('adaptive', 'sbps', 'simulate_bounce')):
      proposal_u, ratio, iteration, proposed_time, G, X = tf.while_loop(
        accepted_fn,
        body=lambda proposal_u, ratio, iteration, proposed_time, G, X: self.simulate_bounce_loop(
          target_log_prob_fn, state, velocity, preconditioner, proposal_u, ratio, iteration, proposed_time, G, X),
        loop_vars=(proposal_u, ratio, iteration, proposed_time, G, X),
        maximum_iterations=self.max_iter)
    # perform some checks on our times to make sure they are within range
    max_reach = lambda: self.max_time
    valid_proposal = lambda: proposed_time
    bounce_time = tf.cond(tf.math.greater_equal(self.iteration - 1, self.max_iter),
                         max_reach, valid_proposal)
    print('Proposed bounce_time = {}'.format(proposed_time))
    print('Proposed ratio = {}'.format(ratio))
    tf.print('bounce__time = {}'.format(bounce_time), output_stream=sys.stdout)
    return bounce_time, tf.reshape(ratio, ())


  def simulate_bounce_loop(self, target_log_prob_fn, state, velocity,
                           preconditioner, proposal_u, ratio, iteration,
                           proposed_time, G, X):
    with tf.name_scope('adaptive_psbps_bounce_loop'):
      print('iteration = {}'.format(iteration))
      # Sample a random time
      with tf.device('/CPU:0'):
        (force_refresh,
         hull_sample_time,
         hull_sample_val) = sample_poisson_thinning(G.stack(),
                                                    X.stack())
      # end running on cpu
      print(f'hull time {hull_sample_time}, hull eval {hull_sample_val}')
      print(f'iteration {iteration}')
      print(f'force refresh  {force_refresh}')
      # if the sample was valid, let's go ahead and evaluate it.
      # otherwise, let's avoid computing any more expensive gradients
      # and just tell our kernel that we should do a refresh.
      force_refresh_fn = lambda : self.force_refresh_fn(
        target_log_prob_fn,
        hull_sample_val,
        state,
        velocity,
        preconditioner,
        iteration, G, X,
        hull_sample_time)
      evaluate_proposal_fn = lambda : self.evaluate_proposal_fn(
        target_log_prob_fn,
        hull_sample_val,
        state,
        velocity,
        preconditioner,
        iteration, G, X,
        hull_sample_time)
    (proposal_u, ratio,
     iteration, hull_sample_time,
     G, X) = tf.cond(force_refresh, force_refresh_fn, evaluate_proposal_fn)
    return proposal_u, ratio, iteration, hull_sample_time, G, X

  def force_refresh_fn(self, target_log_prob_fn,
                       hull_sample_val,
                       state,
                       velocity,
                       preconditioner,
                       iteration, G, X,
                       hull_sample_time):
    print(f'iteration refresh {iteration}')
    # to force exit the loop, we can return a proposal value
    # and ratio such that
    #   tf.math.greater(proposal_u, ratio)
    # evaluates to false. This will ensure the while loop above that calls
    # this simulate_bounce_loop function will exit.
    proposal_u = tf.reshape(tf.convert_to_tensor(0.0, dtype=tf.float32), (1,))
    print('proposal_u refresh shape = {}'.format(proposal_u.shape))
    ratio = tf.convert_to_tensor(1.0, dtype=tf.float32)
    return proposal_u, tf.reshape(ratio, (1,)), iteration, tf.reshape(self.force_refresh_time, ()), G, X

  def evaluate_proposal_fn(self, target_log_prob_fn,
                           hull_sample_val,
                           state,
                           velocity,
                           preconditioner,
                           iteration, G, X,
                           hull_sample_time):
    # this function should be called if the proposed sample is valid
    # (ie. the event rate is > 0)
    # removing indent, as want this to run on GPU again
    # print(type(state))
    # print(type(G))
    print(iteration)
    iteration, G, X, event_rate = self.evaluate_rate(target_log_prob_fn,
                                                     state,
                                                     velocity,
                                                     preconditioner,
                                                     iteration, G, X,
                                                     hull_sample_time)
    # compute the ratio between our evaluated rate and the sample from our
    # envelope
    print('eeevent_rate = {}, hull_sample = {}'.format(event_rate, hull_sample_val))
    print('this is event_rate {}, time {}'.format(G.read(iteration - 1),
                                                  hull_sample_time))
    ratio = event_rate / (hull_sample_val + self.gamma)
    # now make sure that if a Nan or a inf krept in here (very possible)
    # that we attenuate it to zero
    # the is_finite function will find if it is a real number
    ratio = tf.where(~tf.math.is_finite(ratio), tf.zeros_like(ratio), ratio)
    # sample our uniform random variable
    proposal_u = self.proposal_uniform.sample(1)
    print('proposal_u shape = {}'.format(proposal_u.shape))
    print('ratio shape = {}'.format(ratio.shape))
    print('hull_sample_val = {}, proposed_val = {}, ratio = {}'.format(hull_sample_val, G.read(iteration -1), ratio))
    return proposal_u, tf.reshape(ratio, (1,)), iteration, tf.reshape(hull_sample_time, ()), G, X




  # def simulate_bounce_loop(self, target_log_prob_fn, state, velocity,
  #                          preconditioner, proposal_u, ratio, iteration,
  #                          proposed_time, G, X):
  #   with tf.name_scope('adaptive_psbps_bounce_loop'):
  #     # evaluate the rate for this sample
  #     with tf.device('/CPU:0'):
  #       hull_sample_time, hull_sample_val = sample_poisson_thinning(G.stack(),
  #                                                                   X.stack())
  #     # evaluate our rate function now here and add it to our hull as well
  #     iteration, G, X, event_rate = self.evaluate_rate(target_log_prob_fn, state,
  #                                                      velocity, preconditioner,
  #                                                      iteration, G, X,
  #                                                      hull_sample_time)
  #     # compute the ratio between our evaluated rate and the sample from our
  #     # envelope
  #     print('this is event_rate {}, hull_val {}, time {}'.format(G.read(iteration - 1),
  #                                                                hull_sample_val,
  #                                                                hull_sample_time))
  #     ratio = event_rate / hull_sample_val
  #     # now make sure that if a Nan or a inf krept in here (very possible)
  #     # that we attenuate it to zero
  #     # the is_finite function will find if it is a real number
  #     ratio = tf.where(~tf.math.is_finite(ratio), tf.zeros_like(ratio), ratio)
  #     # sample our uniform random variable
  #     proposal_u = self.proposal_uniform.sample(1)
  #     print('proposal_u shape = {}'.format(proposal_u.shape))
  #     print('ratio shape = {}'.format(ratio.shape))
  #     print('hull_sample_val = {}, proposed_val = {}, ratio = {}'.format(hull_sample_val, G.read(iteration -1), ratio))
  #     return proposal_u, tf.reshape(ratio, (1,)), iteration, tf.reshape(hull_sample_time, ()), G, X



class InterpolationSampler(SBPSampler):
  """Piecewise Linear Interpolation for hull formation

  The SBPS looks at picewise linear regression, but I want to look at how the
  piecewise linear method would work for iterpolation. Forming the regression
  can be a bit troublesome, but interpolation is much easier.
  """

  def __init__(self, batch_size=1, data_size=1):
    super().__init__(batch_size, data_size)
    self.init_slope = 0
    self.init_intercept = 0
    self.gamma = tf.convert_to_tensor(1.0, dtype=tf.float32)


  def initialise_hull(self, target_log_prob_fn, state, velocity,
                      G, X, iteration):
    """Initialise samples for that will form the hull"""
    # now call the function to evaluate the hull at this loc
    start_time = tf.convert_to_tensor(0.0, dtype=tf.float32)
    dt = tf.convert_to_tensor(1.0, dtype=tf.float32)
    # iteration, G, X, event_zero = self.evaluate_rate(target_log_prob_fn,
    #                                                  state,
    #                                                  velocity,
    #                                                  iteration,
    #                                                  G, X, start_time)
    # evaluate the hull at the proposed end position
    # event_zero = tf.math.maximum(event_zero, self.gamma)
    # event_zero = 0.0
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


  def evaluate_initialise_hull(self, target_log_prob_fn, state, velocity, time):
    with tf.name_scope('eval_init_upper_bound'):
      next_state = [s + v * time for s, v in zip(state, velocity)]
      print(f'eval_init_hull {next_state[-1]}')
      grads_target_log_prob = target_log_prob_fn(next_state)
      event_rate = compute_dot_prod(velocity, grads_target_log_prob)
      return event_rate




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
    return slope, intercept

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
      In this case, we will update the proposed_hull_eval to minumum
      value of the hull to avoid going negative, and we will return the time such that
      this new linear segment intersects with this mimimum value.

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
    affine_cond = tf.math.greater(tf.math.abs(slope), min_slope)
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
      proposed_time = -U / intercept + last_time
      proposed_hull_eval = intercept
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
    # valid_cond = tf.math.logical_and(valid_cond, tf.math.greater(proposed_hull_eval, 0))
    # valid_cond = tf.math.logical_and(valid_cond, tf.math.greater(proposed_time, 0))

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
      print('updated slope!!!!')
      gamma = tf.cast(self.gamma, tf.float64)
      proposed_time = (gamma - intercept) / slope
      proposed_hull_eval = tf.reshape(gamma,  (1,))
      return proposed_time, proposed_hull_eval

    proposed_time, proposed_hull_eval = tf.cond(valid_cond, valid_proposal, invalid_proposal)
    # now cast returns back to single precision
    proposed_time = tf.cast(proposed_time, dtype=tf.float32)
    proposed_hull_eval = tf.cast(proposed_hull_eval, dtype=tf.float32)
    print(f'proposed time when proposed = {proposed_time}')
    return proposed_time, tf.reshape(proposed_hull_eval, (1,))


  def update_integrated_constant(self,
                                 integrated_constant: tf.Tensor,
                                 slope: tf.Tensor,
                                 intercept: tf.Tensor,
                                 X: tf.TensorArray,
                                 iteration: int,
                                 proposed_time: tf.Tensor) -> tf.Tensor:
    """Add integral of newest hull segment to the integrated constant

    Parameters
    ----------
    integrated_constant : tf.Tensor
        current value for the integrated_constant
    slope : tf.Tensor
        slope/gradient of current linear segment
    intercept : tf.Tensor
        itercept of current linear segment
    X : tf.TensorArray
        Array of sampled times
    iteration : int
        current iteration hull proposal
    proposed_time : tf.Tensor
        newest proposed time that will serve as the end of the current segment

    Returns
    -------
    tf.Tesor
        Updated integrated constant
    """
    # get the starting point of the current hull, which will be the
    # last element in the X array
    start_time = X.read(iteration - 1)[0, 0]
    print(f'start time = {start_time}')
    print(f'proposed time = {proposed_time}')
    integral_upper = 0.5 * slope * tf.square(proposed_time) + intercept * proposed_time
    integral_lower = 0.5 * slope * tf.square(start_time) + intercept * start_time
    integrated_constant = integrated_constant + integral_upper - integral_lower
    return integrated_constant


class InterpolationPSampler(InterpolationSampler):
  """Piecewise Linear Interpolation for hull formation with Preconditioning

  The SBPS looks at picewise linear regression, but I want to look at how the
  piecewise linear method would work for iterpolation. Forming the regression
  can be a bit troublesome, but interpolation is much easier.
  """

  def __init__(self, batch_size=1, data_size=1):
    super().__init__(batch_size, data_size)

  def evaluate_rate(self, target_log_prob_fn, state, velocity,
                    preconditioner, iteration, G, X, time):
    print('IN EVAL RATE, iteration = {}'.format(iteration))
    with tf.name_scope('add_sample_upper_bound'):
      next_state = [s + v * time for s, v in zip(state, velocity)]
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
      next_state = [s + v * time for s, v in zip(state, velocity)]
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


class NaiveThinningSampler(IPPSampler):
  """
  TODO: test and finish implementation
  """
  def __init__(self, exact=True):
    super(*args).__init__(exact=exact)


  def simulate_bounce_time_thinning(self, state, velocity, time):
    """
    """
    # find the current upper bound
    print('state befor fn = {}'.format(state))
    print('here exp')
    exp_d = tfd.Exponential(1.0)
    uni_d = tfd.Uniform()
    accepted = False
    proposed_time = 0.0
    while not accepted:
      _, grads = mcmc_util.maybe_call_fn_and_grads(
        self.target_log_prob_fn, state, name='sbps_simulate_bounce_time')
      upper_bound_rate = self.compute_upper_bound(grads, velocity)
      # now propose time with this rate
      print('proposed time')
      proposed_time += exp_d.sample(1) / upper_bound_rate
      print('proposed time = {}'.format(proposed_time))
      # now get the gradient at this time
      proposed_state = [s + v * proposed_time for s, v in zip(state, velocity)]
      _, grads = mcmc_util.maybe_call_fn_and_grads(
        self.target_log_prob_fn, proposed_state, name='sbps_simulate_bounce_time')
      # get the pointwise rate for this component
      print('getting rate')
      rate = compute_dot_prod(grads, velocity)
      print('rate = {}'.format(rate))
      proposal_u = uni_d.sample(1)
      #new_upper_bound = self.compute_upper_bound(grads, velocity)
      print('orig upper bound = {}'.format(upper_bound_rate))
      #print('new upper bound = {}'.format(new_upper_bound))
      ratio = rate / upper_bound_rate
      accepted_lambda = lambda: True
      rejected_lambda = lambda: False
      #time = proposed_time
      print('proposal_u = {}'.format(proposal_u))
      print('ratio = {}'.format(ratio))
      accepted = tf.case(
        [(tf.math.less(proposal_u, ratio), accepted_lambda)],
        default=rejected_lambda)
    print('state after fn = {}'.format(state))
    if proposed_time <= 0:
      return 10e6
    else:
      print('proposed_time = {}'.format(proposed_time[0]))
      return proposed_time[0]


  def compute_upper_bound(self, grads_target_log_prob, velocity):
    """ Compute upper bound using Cauchy-Schwarz inequality [1]

    Args:
      grads_target_log_prob (list(array)):
        List of arrays for the gradient of each variable.
      velocity (list(array)):
        List of arrays for the velocity of each variable.

     Returns:
       upper bound

    #### References:
    [1] https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality
    """
    # need to find the L2 Norm of all the state elements
    upper = tf.sqrt(compute_l2_norm(grads_target_log_prob))
    # since the norm of the velocity will be one (current assumption)\
    # don't need to worry about computing it, just return the norm for
    # state parts
    return upper
