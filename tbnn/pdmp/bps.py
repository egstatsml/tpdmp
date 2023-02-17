"""
Implements BNN with BPS.

Current implementation will parse the gradient functions,
the model and the data to the ARS module. This is for a first pass
to make sure everythiong works, and then tidy up.

To begin with the main bit of the implementation, start with the
grad_bounce_intensity_fn, and then work back all the way to the
log_likelihood etc.

Focus on methods from [1]


#### References
[1]:  Alexandre Bouchard-Côté∗, Sebastian J. Vollmer†and Arnaud Doucet;
      The Bouncy Particle Sampler: A Non-ReversibleRejection-Free Markov
      Chain Monte Carlo Method. https://arxiv.org/pdf/1510.02451.pdf

"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
import sys
import collections

from tbnn.pdmp.poisson_process import IsotropicGaussianSampler, SBPSampler, PSBPSampler
from tbnn.pdmp import utils
import time as timepy

BPSKernelResults = collections.namedtuple(
  'BPSKernelResults',
  [
    'velocity',               # also used for the next state
    'time',                   # also used for sampling from next state
    'proposed_time',                   # also used for sampling from next state
    'acceptance_ratio'        # ratio for acceptence prob
  ])

PBPSKernelResults = collections.namedtuple(
  'PBPSKernelResults',
  [
    'velocity',               # also used for the next state
    'time',                   # also used for sampling from next state
    'preconditioner',         # important part of preconditioned BPS
    'pre',
    'acceptance_ratio'        # ratio for acceptence prob
  ])



class BPSKernel(tfp.mcmc.TransitionKernel):
  """Transition kernel for BPS within TFP

  Using description for UncalibratedHamiltonianMonteCarlo as the reference
  """
  def __init__(self,
               target_log_prob_fn,
               lambda_ref=1.0,
               std_ref=0.001,
               ipp_sampler=SBPSampler,
               batch_size=1.0,
               data_size=1.0,
               grad_target_log_prob=None,
               bounce_intensity=None,
               grad_bounce_intensity=None,
               state_gradients_are_stopped=False,
               seed=None,
               store_parameters_in_results=False,
               name=None):
    """Initializes this transition kernel.
    Args:

      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      lambda_ref (float):
        reference value for setting refresh rate
      std_ref (float):
        standard deviation for refresh distribution
      state_gradients_are_stopped (bool):
        indicating that the proposed new state be run through
        `tf.stop_gradient`. This is particularly useful when combining
        optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      seed (int):
        Python integer to seed the random number generator.
      store_parameters_in_results (bool):
        If `True`, then `step_size` and `num_leapfrog_steps` are written
        to and read from eponymous fields in the kernel results objects
        returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    if seed is not None and tf.executing_eagerly():
      # TODO(b/68017812): Re-enable once TFE supports `tf.random_shuffle` seed.
      raise NotImplementedError('Specifying a `seed` when running eagerly is '
                                'not currently supported. To run in Eager '
                                'mode with a seed, use `tf.set_random_seed`.')
    if not store_parameters_in_results:
      # this warning is just here for the moment as a placeholder to remain
      # consistent with TFP, in case we need to store any results
      pass
    #self._seed_stream = SeedStream(seed, salt='bps_one_step')
    self._parameters = dict(
      target_log_prob_fn=target_log_prob_fn,
      grad_target_log_prob=grad_target_log_prob,
      lambda_ref=lambda_ref,
      std_ref=std_ref,
      bounce_intensity=bounce_intensity,
      grad_bounce_intensity=grad_bounce_intensity,
      state_gradients_are_stopped=state_gradients_are_stopped,
      seed=seed,
      name=name,
      store_parameters_in_results=store_parameters_in_results)
    self._momentum_dtype = None
    self.ipp_sampler = ipp_sampler(batch_size=batch_size, data_size=data_size)
    self.batch_size = batch_size
    self.data_size = data_size
    self.ar_sampler = []   # is initialised in the bootstrap_results method
    self.ref_dist = tfd.Exponential(self.lambda_ref)

  @property
  def target_log_prob_fn(self):
    return self._parameters['target_log_prob_fn']

  # need to define a setter attribute for the target_log_prob_fn
  @target_log_prob_fn.setter
  def target_log_prob_fn(self, target_log_prob_fn):
    self._parameters['target_log_prob_fn'] = target_log_prob_fn

  @property
  def grad_target_log_prob_fn(self):
    return self._parameters['grad_target_log_prob_fn']

  @property
  def bounce_intensity_fn(self):
    return self._parameters['bounce_intensity_fn']

  @property
  def grad_bounce_intensity_fn(self):
    return self._parameters['grad_bounce_intensity_fn']

  @property
  def state_gradients_are_stopped(self):
    return self._parameters['state_gradients_are_stopped']

  @property
  def lambda_ref(self):
    return self._parameters['lambda_ref']

  @property
  def std_ref(self):
    return self._parameters['std_ref']

  @property
  def seed(self):
    return self._parameters['seed']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  @property
  def is_calibrated(self):
    return False

  @property
  def _store_parameters_in_results(self):
    return self._parameters['store_parameters_in_results']


  def one_step(self, previous_state, previous_kernel_results):
    """ performs update for Bouncy Particle Sampler

    main functionality described in Algorithm 1 of [1]
    For clarity, this implementation will adhere to the order of operations
    stated in Algorithm 1 in [1].

    Args:
      previous_state (tensor):
        previous state of parameters
      previous_kernel_results (`collections.namedtuple` containing `Tensor`s):
        representing values from previous calls to this function (or from the
        `bootstrap_results` function.)

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) after taking exactly one step. Has same type and
        shape as `previous_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.

    #### References
    [1]:  Alexandre Bouchard-Côté∗, Sebastian J. Vollmer†and Arnaud Doucet;
          The Bouncy Particle Sampler: A Non-ReversibleRejection-Free Markov
          Chain Monte Carlo Method. https://arxiv.org/pdf/1510.02451.pdf
    """
    # main implementation of the BPS algorith from alg. 1. of reference [1]
    # step numbers are prepended with the same notation
    # starts at step 2., as assumes x and v have been initialised
    with tf.name_scope(mcmc_util.make_name(self.name, 'bps', 'one_step')):
      # preparing all args first
      # very similar to the HMC module in TFP
      # we are passing the values for target and the gradient from their
      # previous sample to ensure that the maybe_call_fn_and_grads won't
      # compute the target.
      # will be dooing it manually because we want to find the gradient w.r.t
      # parameters and with the time
      tf.print('start one step', output_stream=sys.stdout)
      [
        previous_state_parts,
        previous_velocity_parts,
      ] = self._prepare_args(
        self.target_log_prob_fn,
        previous_state,
        previous_kernel_results.velocity,
        maybe_expand=True,
        state_gradients_are_stopped=self.state_gradients_are_stopped)
      # (a) simulate the first arrival time bounce a IPP
      t_bounce, acceptance_ratio = self.ipp_sampler.simulate_bounce_time(
        self.target_log_prob_fn,
        previous_state_parts,
        previous_velocity_parts)
      # (b) Simulate ref time
      t_ref = tf.reshape(self.ref_dist.sample(1), ())
      tf.print('t_bounce = {}'.format(t_bounce))
      # (c) set the time and update to the next position
      time = tf.math.minimum(t_bounce, t_ref)
      next_state_parts = self.compute_next_step(previous_state_parts,
                                                previous_velocity_parts,
                                                time)
      # (d,e) sample the next velocity
      next_velocity_parts = self.compute_next_velocity(next_state_parts,
                                                       previous_velocity_parts,
                                                       time,
                                                       t_bounce,
                                                       t_ref)
      # now save the next state and velocity in the kernel results
      new_kernel_results = previous_kernel_results._replace(
        velocity=next_velocity_parts,
        proposed_time=t_bounce,
        time=time,
        acceptance_ratio=acceptance_ratio)

      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(previous_state) else x[0]
      return maybe_flatten(next_state_parts), new_kernel_results


  def compute_next_step(self, state, velocity, time):
    """ updates the current state with the velocity and time found"""
    next_step = [u + v * time for u, v in zip(state, velocity)]
    return next_step


  def bounce_intensity(self, time, state_parts, velocity):
    time = tf.Variable(time, dtype=tf.float32)
    state_parts_velocity_time = [
      u + v * time for u,v in zip(state_parts, velocity)]
    target_log_prob, grads_target_log_prob = mcmc_util.maybe_call_fn_and_grads(
      self.target_log_prob_fn, state_parts_velocity_time)
    # apply dot product between dU/dx and velocity
    bounce_intensity = utils.compute_dot_prod(grads_target_log_prob, velocity)
    return bounce_intensity.numpy()


  def grad_bounce_intensity(self, time, state_parts, velocity):
    """want to return lambda and d/dt{lambda}"""
    time = tf.Variable(time, dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape_bounce:
      tape_bounce.watch([time])
      state_parts_velocity_time = [
        u + v * time for u,v in zip(state_parts, velocity)]
      target_log_prob, grads_target_log_prob = mcmc_util.maybe_call_fn_and_grads(
        self.target_log_prob_fn, state_parts_velocity_time)
      # apply dot product between dU/dx and velocity
      bounce_intensity = utils.compute_dot_prod(grads_target_log_prob, velocity)
    # now compute gradient of above w.r.t time
    grad_t = tape_bounce.gradient(bounce_intensity, time)
    # print(grads_target_log_prob)
    # print('bounce_intensity = {}, time = {}, grad_t = {}'.format(bounce_intensity, time, grad_t))
    del tape_bounce
    return grad_t.numpy()


  def simulate_ref_time(self, time):
    return np.random.exponential() / self.lambda_ref
    #return np.random.exponential(self.lambda_ref)


  def compute_next_velocity(self,
                            next_state,
                            current_velocity_parts,
                            time,
                            t_bounce,
                            t_ref):
    """update the velocity based on the current times

    if t == t_bounce:
      update using the gradient of potential
      (Newtonian Elastic Collision)
    else if t == t_ref:
      refresh by sampling from a normal distribution

    Args:
      next_step (List(tf.Tensor)):
        the start of the next state in our updated trajectory
      current_velocity (List(tf.Tensor)):
        velocity just before the new updated trajectory
      time (tf.float):
        the time that will be used for the new trajectory
      t_bounce (tf.float):
        the proposed bounce time in our dynamics
      t_ref (tf.float):
        the sampled reference time

    Returns:
      the updated velocity for the next step
    """
    # if updating using a refresh
    print('t_ref = {}'.format(t_ref))
    refresh = lambda: self.refresh_velocity(current_velocity_parts)
    bounce = lambda: self.collision_velocity(next_state, current_velocity_parts)
    next_velocity = tf.cond(time == t_ref, refresh, bounce)
    return next_velocity


  def refresh_velocity(self, current_velocity_parts):
    """ Use refresh step for updating the velocity component

    This is just sampling from a random normal for each component
    Corresponds to step (d) in Alg 1 of [1]

    Args:
      current_velocity_parts (list(array)):
        parts of the current velocity

    Returns:
      Next velocity sampled from a Normal dist
    """
    tf.print('am refreshing')
    # normal = tfd.Normal(0., 1.0)
    # new_v = [normal.sample(x.shape) for x in current_velocity_parts]
    # new_norm = tf.sqrt(utils.compute_l2_norm(new_v))
    # new_v = [v / new_norm for v in new_v]
    # uniform = tfd.Uniform(low=-1.0, high=1.0)
    # new_v = [uniform.sample(x.shape) for x in current_velocity_parts]
    # new_norm = tf.sqrt(utils.compute_l2_norm(new_v))
    # new_v = [v / new_norm for v in new_v]
    dist = tfd.Normal(loc=0.0, scale=self.std_ref)
    new_v = [dist.sample(x.shape) for x in current_velocity_parts]
    # new_norm = tf.sqrt(utils.compute_l2_norm(new_v))
    # new_v = [v / new_norm for v in new_v]



    return new_v


  def collision_velocity(self, next_state, current_velocity_parts):
    """update the velocity based on simulated collision

    Collision is simulated using Newtonian Elastic Collision, as per
    equation (2) of ref [1].
    This equation is summarised here as.
    v_{i+1} = v - 2 * dot(grads_target, v) / norm(grads_target) * grads_target

    Args:
      next_state (list(array)):
        next position for which we need to compute collision direction
      current_velocity_parts (list(array)):
        parts of the current velocity

    Returns:
      the updated velocity for the next step based on collision dynamics
    """
    tf.print('am bouncing')
    # need to compute the grad for the newest position
    grads_target_log_prob = self.target_log_prob_fn(next_state)
    # TODO: validate design choice here
    # Design choice: Since executing inner product, should we flatten everything
    # to a vector and do it that way, or just do element-wise multiplication and
    # then just sum? One way is mathematically more pretty, other might be
    # easier to write in code?
    # for the moment will do element-wise multiplication
    # print(grads_target_log_prob)
    #
    # Need to find the Norm of the grad component, which requires looking at the
    # all the grad elements in the list.
    # grads_norm = utils.compute_l2_norm(grads_target_log_prob)
    grads_norm = utils.compute_dot_prod(grads_target_log_prob, grads_target_log_prob)
    # now need to compute the inner product of the grads_target and the velocity
    dot_grad_velocity = utils.compute_dot_prod(grads_target_log_prob,
                                              current_velocity_parts)
    # can now compute the new velocity from the simulated collision
    new_v = [v - 2. * u * dot_grad_velocity / grads_norm for u, v in zip(
      grads_target_log_prob, current_velocity_parts)]
    return new_v


  def bootstrap_results(self, init_state):
    """Creates initial `previous_kernel_results` using a supplied `state`.

    Returns an object with the same type as returned by one_step(...)

    Args:
      init_state (Tensor or list(Tensors)):
        initial state for the kernel
    Returns:
        an instance of BPSKernelResults with the initial values set
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'bps', 'bootstrap_results')):
      # print(f'length of the params in the bootstrap {len(init_state)}')
      #init_state, _ = mcmc_util.prepare_state_parts(init_state)
      # init_grads_target_log_prob = self.target_log_prob_fn(init_state)
      # prepare the init velocity
      # get the initial velocity from the refresh fn, by just passing it the
      # state variables which will be used only to get the correct shape of things
      init_velocity = self.refresh_velocity(init_state)
      init_velocity, _ = mcmc_util.prepare_state_parts(init_velocity,
                                                       name='init_velocity')
      # initialise the AR sampler now
      if self._store_parameters_in_results:
        return BPSKernelResults(
          velocity=init_velocity,
          proposed_time=1.0,
          time=1.0,
          acceptance_ratio=0.0)
      else:
        raise NotImplementedError('need to implement for not saving results')

  # @tf.function
  def examine_event_intensity(self, state, velocity, time):
    """method used to examine the event intensity

    This is intended as a diagnositc function; it isn't integral to
    the actual implementation of the BPS, but it is for how the
    intensity function for the IPP that controls the event rate
    behaves.

    Is intended to be run in a loop to see how the intensity changes
    w.r.t. time.

    Args:
      state (Tensor or list(Tensors)):
        state for the kernel
      velocity (Tensor or list(Tensors)):
        current velocity for the kernel
      time (tf.float32):
        the time we are currently looking at

    Returns:
      eval of the IPP rate for the BPS for current state, velocity and time.
    """
    # update the state for givin time and velocity
    updated_state = [s + v * time for s,v in zip(state, velocity)]
    # need to get the gradient of the current state
    grad = self.target_log_prob_fn(updated_state)
    # print('grad = {}'.format(grad))
    ipp_intensity = utils.compute_dot_prod(grad, velocity)
    return ipp_intensity


  def _prepare_args(self, target_log_prob_fn,
                    state,
                    velocity,
                    target_log_prob=None,
                    maybe_expand=False,
                    state_gradients_are_stopped=False):
    """Helper which processes input args to meet list-like assumptions.

    Much of this is directly copied from the HMC module in TFP. Have updated
    for BPS
    """
    state_parts, _ = mcmc_util.prepare_state_parts(state, name='current_state')
    # do similar thing for the velocity components
    velocity_parts, _ = mcmc_util.prepare_state_parts(velocity,
                                                      name='current_velocity')
    def maybe_flatten(x):
      return x if maybe_expand or mcmc_util.is_list_like(state) else x[0]
    # returning the original state, as it is needed for computing the next
    # state/position in step (c) of Alg. 1 from [1]
    return [
      maybe_flatten(state_parts),
      maybe_flatten(velocity_parts),
    ]


class IsotropicGaussianBPSKernel(BPSKernel):
  """Transition kernel for BPS for Isotropic Gaussian Target

  Mostly the same as the BPS, but uses analytic results for sampling using
  the inversion method.
  """
  @mcmc_util.set_doc(BPSKernel.__init__.__doc__)
  def __init__(self,
               target_log_prob_fn,
               lambda_ref=1.0,
               grad_target_log_prob=None,
               bounce_intensity=None,
               grad_bounce_intensity=None,
               state_gradients_are_stopped=False,
               seed=None,
               store_parameters_in_results=False,
               name=None):
    super().__init__(target_log_prob_fn,
                     lambda_ref=lambda_ref,
                     grad_target_log_prob=grad_target_log_prob,
                     bounce_intensity=bounce_intensity,
                     grad_bounce_intensity=grad_bounce_intensity,
                     state_gradients_are_stopped=state_gradients_are_stopped,
                     seed=seed,
                     store_parameters_in_results=store_parameters_in_results,
                     name=name)
    self.ipp_sampler = IsotropicGaussianSampler()


class IterBPSKernel(BPSKernel):
  """Transition kernel for BPS that handles an iter object
  to parse mini batches of data

  Is different to the normal kernels in BPS, in that their is no
  target_log_prob_fn. There is parent function that is called upon each
  iteration, which first iterates over the next batch of data within
  the model, and then returns to local target_log_prob_fn.

  An example would be,

  ```python
  def iter_bnn_neg_joint_log_prob(model, weight_prior_fns, bias_prior_fns, dataset_iter):
    def _fn():
      X, y = dataset_iter.next()
      return bnn_neg_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X, y)
    return _fn

  # create the kernel now
  kernel = IterBPSKernel(iter_bnn_neg_joint_log_prob, ...)
  # include any other args that are parsed to the normal BPSKernel
  ```
  This is different to the BPSKernel (or any other kernel in TFP)
  in that they would just have something like,

  ```python
  # create a callable of the neg joint log prob
  target_log_prob = bnn_neg_joint_log_prob_fn(model, weight_prior_fns,
                                              bias_prior_fns, X, y)
  # create the kernel now
  kernel = BPSKernel(target_log_prob, ...)
  # similarly handle any other args for normal BPS
  ```
  """
  def __init__(self,
               parent_target_log_prob_fn,
               lambda_ref=1.0,
               std_ref=0.001,
               ipp_sampler=SBPSampler,
               batch_size=1.0,
               data_size=1.0,
               grad_target_log_prob=None,
               bounce_intensity=None,
               grad_bounce_intensity=None,
               state_gradients_are_stopped=False,
               seed=None,
               store_parameters_in_results=False,
               name=None):
    """Initializes this transition kernel.
    Args:
      parent_target_log_prob_fn: Python callable returns another Python callable
        which then takes an argument like `current_state`
        (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
        Refer to the class docstring for info and examples about how this
        differs from the normal kernel construction.
      lambda_ref (float):
        reference value for setting refresh rate
      state_gradients_are_stopped (bool):
        indicating that the proposed new state be run through
        `tf.stop_gradient`. This is particularly useful when combining
        optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      seed (int):
        Python integer to seed the random number generator.
      store_parameters_in_results (bool):
        If `True`, then `step_size` and `num_leapfrog_steps` are written
        to and read from eponymous fields in the kernel results objects
        returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    # need to initialise the target_log_prob_fn, so that the
    # bootstrap_results method can be called.
    self.parent_target_log_prob_fn = parent_target_log_prob_fn
    target_log_prob_fn = self.parent_target_log_prob_fn()
    super().__init__(target_log_prob_fn,
                     lambda_ref,
                     std_ref,
                     ipp_sampler,
                     batch_size,
                     data_size,
                     grad_target_log_prob,
                     bounce_intensity,
                     grad_bounce_intensity,
                     state_gradients_are_stopped,
                     seed,
                     store_parameters_in_results,
                     name)


  def one_step(self, previous_state, previous_kernel_results):
    """Will call the parent_target_log_prob_fn, which will get the
    next instance of the target_log_prob fn and set it. Will than call
    the parent class to perform a one step on this iteration.
    """
    # get the new local target_log_prob_fn
    self.target_log_prob_fn = self.parent_target_log_prob_fn()
    print('target_log_prob_fn = {}'.format(self.target_log_prob_fn))
    next_state_parts, next_kernel_results = super().one_step(previous_state, previous_kernel_results)
    return next_state_parts, next_kernel_results


class PBPSKernel(BPSKernel):
  """ Preconditioned BPS kernel from the SBPS paper [1]

  Preconditioned SBPS (pSBPS) uses preconditioning based on the [2]

  Is designed so that classes that implement another form of preconditioning
  (such as the vABPS method) can inherit from here.

  #### Notation and pSBPS

  pSBPS uses method similar to RMSProp, except it normalises it to prevent
  scaling of the gradient.

  In pseudo-code, the pSBPS method for getting the preconditioner looks
  "something" like this,
  ```
  pre_{i} = beta * g^{2} + (1 - beta) * pre_{i-1}
  pre_hat = (1 / d) * sum( 1 / sqrt(pre_{i}) )
  preconditioner = diag(pre_hat / sqrt(pre_{i}))
  ```


  Currently using an admittadly poor choice of var. names for the preconditioner
  components in pSBPS, particularly the `pre_*` variables. Have made an issue
  #43 to change it to something nicer.

  #### References

  [1] Pakman, Ari, et al. "Stochastic bouncy particle sampler."
  International Conference on Machine Learning. 2017.

  [2] Li, Chunyuan, et al. "Preconditioned stochastic gradient Langevin
  dynamics for deep neural networks." arXiv preprint arXiv:1512.07666 (2015).
  """
  def __init__(self,
               target_log_prob_fn,
               beta=0.99,
               epsilon=0.00001,
               lambda_ref=1.0,
               std_ref=0.0001,
               ipp_sampler=PSBPSampler,
               batch_size=1.0,
               data_size=1.0,
               grad_target_log_prob=None,
               bounce_intensity=None,
               grad_bounce_intensity=None,
               state_gradients_are_stopped=False,
               seed=None,
               store_parameters_in_results=False,
               name=None):
    """Initializes this transition kernel.
    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      beta (float):
        Discounting factor (similar to RMSProp)
      epsilon (float):
        small value to make sure we aren't dividing by zero
      lambda_ref (float):
        reference value for setting refresh rate
      std_ref (float):
        std. for reference dist
      state_gradients_are_stopped (bool):
        indicating that the proposed new state be run through
        `tf.stop_gradient`. This is particularly useful when combining
        optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      seed (int):
        Python integer to seed the random number generator.
      store_parameters_in_results (bool):
        If `True`, then `step_size` and `num_leapfrog_steps` are written
        to and read from eponymous fields in the kernel results objects
        returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    super().__init__(target_log_prob_fn,
                     lambda_ref=lambda_ref,
                     std_ref=std_ref,
                     ipp_sampler=ipp_sampler,
                     batch_size=batch_size,
                     data_size=data_size,
                     grad_target_log_prob=grad_target_log_prob,
                     bounce_intensity=bounce_intensity,
                     grad_bounce_intensity=grad_bounce_intensity,
                     state_gradients_are_stopped=state_gradients_are_stopped,
                     seed=seed,
                     store_parameters_in_results=store_parameters_in_results,
                     name=name)
    self.beta = beta
    self.epsilon = 0.001#epsilon


  def one_step(self, previous_state, previous_kernel_results):
    """ performs update for preconditioned Bouncy Particle Sampler

    Very similar to method from the BPSKernel.one_step method,
    though is re-implemented here as there are new params required
    to send through to each of the update methods due to the inclusion
    of the preconditioner.

    For more info on the ins and outs of this method, please refer to
    the BPSKernel.one_step docstring.
    """
    with tf.name_scope(mcmc_util.make_name(self.name, 'pbps', 'one_step')):
      # preparing all args first
      # very similar to the HMC module in TFP
      # we are passing the values for target and the gradient from their
      # previous sample to ensure that the maybe_call_fn_and_grads won't
      # compute the target.
      # will be dooing it manually because we want to find the gradient w.r.t
      # parameters and with the time
      tf.print('start one step', output_stream=sys.stdout)
      [
        previous_state_parts,
        previous_velocity_parts,
        previous_preconditioner_parts,
        previous_pre_parts,
      ] = self._prepare_args(
        self.target_log_prob_fn,
        previous_state,
        previous_kernel_results.velocity,
        previous_kernel_results.preconditioner,
        previous_kernel_results.pre,
        maybe_expand=True,
        state_gradients_are_stopped=self.state_gradients_are_stopped)
      # simulate the arrival of the first event in our IPP
      t_bounce, acceptance_ratio = self.ipp_sampler.simulate_bounce_time(
        self.target_log_prob_fn,
        previous_state_parts,
        previous_velocity_parts,
        previous_preconditioner_parts)
      # simulate from our reference distribution
      t_ref = tf.reshape(self.ref_dist.sample(1), ())
      tf.print('ht_bounce = {}'.format(t_bounce))
      # (c) set the time and update to the next position
      time = tf.math.minimum(t_bounce, t_ref)
      next_state_parts = self.compute_next_step(previous_state_parts,
                                                previous_velocity_parts,
                                                previous_preconditioner_parts,
                                                time)
      # (d,e) sample the next velocity and update the preconditioner
      next_velocity_parts, next_preconditioner_parts, next_pre_parts = self.compute_next_velocity(
        next_state_parts,
        previous_velocity_parts,
        previous_preconditioner_parts,
        previous_pre_parts,
        time,
        t_bounce,
        t_ref)
      # now save the next state and velocity in the kernel results
      new_kernel_results = previous_kernel_results._replace(
        velocity=next_velocity_parts,
        preconditioner=next_preconditioner_parts,
        pre=next_pre_parts,
        time=time,
        acceptance_ratio=acceptance_ratio)

      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(previous_state) else x[0]
      return maybe_flatten(next_state_parts), new_kernel_results


  def compute_next_step(self, state, velocity, preconditioner, time):
    """ updates the current state with the velocity and time found"""
    next_step = [u + a * v * time for u, v, a in zip(state, velocity, preconditioner)]
    return next_step


  def compute_next_velocity(self,
                            next_state,
                            current_velocity_parts,
                            current_preconditioner_parts,
                            current_pre_parts,
                            time,
                            t_bounce,
                            t_ref):
    """update the velocity based on the current times with preconditioner

    if t == t_bounce:
      update using the gradient of potential
      (Newtonian Elastic Collision)
    else if t == t_ref:
      refresh by sampling from a normal distribution

    Args:
      next_step (List(tf.Tensor)):
        the start of the next state in our updated trajectory
      current_velocity (List(tf.Tensor)):
        velocity just before the new updated trajectory
      current_pre_parts (List(tf.Tensor)):
        current "pre" parts for the pSBPS update, refer to class docstring
        for an elaboration on this var.
      current_preconditioner (List(tf.Tensor)):
        current preconditioner matrix
      time (tf.float):
        the time that will be used for the new trajectory
      t_bounce (tf.float):
        the proposed bounce time in our dynamics
      t_ref (tf.float):
        the sampled reference time

    Returns:
      updated velocity for the next step and updated preconditioner components

    """
    t_ref = tf.convert_to_tensor(t_ref)
    print('t_ref = {}, t_bounce = {}, time = {}'.format(t_ref, t_bounce, time))
    t_bounce = tf.convert_to_tensor(t_bounce)
    refresh = lambda: self.refresh_velocity(current_velocity_parts,
                                       current_preconditioner_parts,
                                       current_pre_parts)
    bounce = lambda: self.collision_velocity(next_state, current_velocity_parts,
                                        current_preconditioner_parts,
                                        current_pre_parts)
    next_velocity, next_preconditioner, next_pre = tf.cond(time == t_ref, refresh, bounce)
    return next_velocity, next_preconditioner, next_pre


  def refresh_velocity(self, current_velocity_parts,
                       current_preconditioner_parts, current_pre):
    """ Use refresh step for updating the velocity component

    Pretty much the exact same as the BPSKernel method.
    Am only re-implementing it here as it now takes in a second
    argument for the preconditioner, but it won't update it on
    a refresh step.

    Args:
      current_velocity_parts (list(array)):
        parts of the current velocity
      current_preconditioner_parts (list(array)):
        parts of the current preconditioner
      current_pre_parts (List(tf.Tensor)):
        current "pre" parts for the pSBPS update, refer to class docstring
        for an elaboration on this var.

    Returns:
      Next velocity sampled from a Normal dist and will return the
      input preconditioner (won't update it here)
    """
    tf.print('am refreshing')
    # uniform = tfd.Uniform(low=-1.0, high=1.0)
    # new_v = [uniform.sample(x.shape) for x in current_velocity_parts]
    # new_norm = tf.sqrt(utils.compute_l2_norm(new_v))
    # new_v = [v / new_norm for v in new_v]
    dist = tfd.Normal(loc=0.0, scale=self.std_ref)
    new_v = [dist.sample(x.shape) for x in current_velocity_parts]
    return (new_v, current_preconditioner_parts, current_pre)



  # def collision_velocity(self, next_state, current_velocity_parts,
  #                        current_preconditioner_parts,
  #                        current_pre_parts):
  #   """update the velocity based on simulated collision with preconditioner

  #   Refer to `BPSKernel.collision_velocity` for more info in type of collision.

  #   Is updated here to include preconditioner as done in SBPS paper.

  #   Args:
  #     next_state (list(array)):
  #       next position for which we need to compute collision direction
  #     current_velocity_parts (list(array)):
  #       parts of the current velocity
  #     current_preconditioner_parts (list(array)):
  #       parts of the current preconditioner
  #     current_pre_parts (List(tf.Tensor)):
  #       current "pre" parts for the pSBPS update, refer to class docstring
  #       for an elaboration on this var.

  #   Returns:
  #     the updated velocity for the next step based on collision dynamics with
  #     preconditioner used in SBPS paper.
  #   """
  #   tf.print('am bouncing')
  #   # need to compute the grad for the newest position
  #   _, grads_target_log_prob = mcmc_util.maybe_call_fn_and_grads(
  #     self.target_log_prob_fn, next_state, name='sbps_collision_update')
  #   # update the preconditioner component
  #   pre = [self.beta * tf.square(g) + (1.0 - self.beta) * a for a, g in zip(
  #     current_pre_parts, grads_target_log_prob)]
  #   # make sure there aren't any negative components in here
  #   # there shouldn't be but my paranoia comes into play here
  #   pre = [tf.clip_by_value(a, self.epsilon, 10e10) for a in pre]
  #   # getting the unnormalised preconditioner scaler
  #   # the two `reduce_sum()` ops are needed as will be different number of
  #   # dimensions for the kernel and biases. then divide by the
  #   # number of parameters to normalise it
  #   num_params = tf.cast(
  #     tf.math.reduce_sum([tf.math.reduce_prod(x.shape) for x in pre]),
  #     tf.float32)
  #   pre_hat = tf.math.reduce_sum(
  #     [tf.reduce_sum(tf.math.divide(1.0, (tf.sqrt(a) + self.epsilon))) for a in pre])
  #   pre_hat = tf.divide(pre_hat, num_params)
  #   # can now find the preconditioner values
  #   preconditioner = [tf.divide(pre_hat, tf.sqrt(a) + self.epsilon) for a in pre]
  #   # find the preconditioned gradients now
  #   preconditioned_grads = [tf.math.multiply(a, g) for a, g in zip(preconditioner, grads_target_log_prob)]
  #   grads_norm = utils.compute_l2_norm(preconditioned_grads)
  #   # now need to compute the inner product of the grads_target and the velocity
  #   dot_pre_grad_velocity = utils.compute_dot_prod(preconditioned_grads,
  #                                                  current_velocity_parts)
  #   # can now compute the new velocity from the simulated collision
  #   new_v = [v - 2. * u * dot_pre_grad_velocity / grads_norm for u, v in zip(
  #     preconditioned_grads, current_velocity_parts)]
  #   return (new_v, preconditioner, pre)



  def collision_velocity(self, next_state, current_velocity_parts,
                         current_preconditioner_parts,
                         current_pre_parts):
    """update the velocity based on simulated collision with preconditioner

    Refer to `BPSKernel.collision_velocity` for more info in type of collision.

    Is updated here to include preconditioner as done in SBPS paper.

    Args:
      next_state (list(array)):
        next position for which we need to compute collision direction
      current_velocity_parts (list(array)):
        parts of the current velocity
      current_preconditioner_parts (list(array)):
        parts of the current preconditioner
      current_pre_parts (List(tf.Tensor)):
        current "pre" parts for the pSBPS update, refer to class docstring
        for an elaboration on this var.

    Returns:
      the updated velocity for the next step based on collision dynamics with
      preconditioner used in SBPS paper.
    """
    tf.print('am bouncing')
    # need to compute the grad for the newest position
    grads_target_log_prob =  self.target_log_prob_fn(next_state)
    # update the preconditioner component
    pre = [(1.0 - self.beta) * tf.square(g) + (self.beta * p) for g, p in zip(
      grads_target_log_prob, current_pre_parts)]
    # make sure there aren't any negative components in here
    # there shouldn't be but my paranoia comes into play here
    pre = [tf.clip_by_value(a, self.epsilon, 10e3) for a in pre]
    pre_hat = [tf.math.divide(1.0, (tf.sqrt(a) + self.epsilon)) for a in pre]
    # can now find the preconditioner values
    # finding the mean of the preconditioner
    num_params = tf.cast(
      tf.math.reduce_sum([tf.math.reduce_prod(x.shape) for x in pre]),
      tf.float64)
    pre_hat = tf.math.reduce_sum([tf.reduce_sum(tf.cast(a, tf.float64)) for a in pre_hat])
    pre_hat = tf.cast(pre_hat / num_params, tf.float32)
    preconditioner = [tf.divide(pre_hat, tf.math.sqrt(a + self.epsilon)) for a in pre]
    # find the preconditioned gradients now
    preconditioned_grads = [tf.math.multiply(a, g) for a, g in zip(preconditioner, grads_target_log_prob)]
    # grads_norm = utils.compute_l2_norm(preconditioned_grads)
    grads_norm = utils.compute_dot_prod(preconditioned_grads, preconditioned_grads)
    # now need to compute the inner product of the grads_target and the velocity
    dot_pre_grad_velocity = utils.compute_dot_prod(preconditioned_grads,
                                                   current_velocity_parts)
    # can now compute the new velocity from the simulated collision
    new_v = [v - 2. * u * dot_pre_grad_velocity / grads_norm for u, v in zip(
      preconditioned_grads, current_velocity_parts)]
    return (new_v, preconditioner, pre)



  def bootstrap_results(self, init_state):
    """Creates initial `previous_kernel_results` using a supplied `state`.

    Returns an object with the same type as returned by one_step(...)

    Args:
      init_state (Tensor or list(Tensors)):
        initial state for the kernel
    Returns:
        an instance of BPSKernelResults with the initial values set
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'pbps', 'bootstrap_results')):
      # init_state, _ = mcmc_util.prepare_state_parts(init_state)
      init_grads_target_log_prob = [tf.zeros(x.shape, dtype=tf.float32) for x in init_state]
      # prepare the init velocity and preconditioner
      init_preconditioner = [tf.ones(x.shape, dtype=tf.float32) for x in init_state]
      init_pre = [tf.zeros(x.shape, dtype=tf.float32) for x in init_state]
      # get the initial velocity from the refresh fn, by just passing it the
      # state variables which will be used only to get the correct shape of things
      init_velocity, init_preconditioner, init_pre = self.refresh_velocity(
        init_state, init_preconditioner, init_pre)
      # tf.print('preconditioner = {}'.format(init_preconditioner), output_stream=sys.stdout)
      init_velocity, _ = mcmc_util.prepare_state_parts(init_velocity,
                                                       name='init_velocity')
      init_preconditioner, _ = mcmc_util.prepare_state_parts(init_preconditioner,
                                                       name='init_preconditioner')
      init_pre, _ = mcmc_util.prepare_state_parts(init_pre,
                                                  name='init_pre')

      if self._store_parameters_in_results:
        return PBPSKernelResults(
          velocity=init_velocity,
          preconditioner=init_preconditioner,
          pre=init_pre,
          time=1.0,
          acceptance_ratio=0.0)
      else:
        raise NotImplementedError('need to implement for not saving results')

  @tf.function
  def examine_event_intensity(self, state, velocity, preconditioner, time):
    """method used to examine the event intensity

    This is intended as a diagnositc function; it isn't integral to
    the actual implementation of the BPS, but it is for how the
    intensity function for the IPP that controls the event rate
    behaves.

    Is intended to be run in a loop to see how the intensity changes
    w.r.t. time.

    Args:
      state (Tensor or list(Tensors)):
        state for the kernel
      velocity (Tensor or list(Tensors)):
        current velocity for the kernel
      preconditioner (list(array)):
        parts of the current preconditioner
      time (tf.float32):
        the time we are currently looking at

    Returns:
      eval of the IPP rate for the BPS for current state, velocity and time.
    """
    # update the state for givin time and velocity
    updated_state = [s + a * v * time for s, v, a in zip(state, velocity, preconditioner)]
    # need to get the gradient of the current state
    _, grad = mcmc_util.maybe_call_fn_and_grads(self.target_log_prob_fn,
                                                updated_state)
    ipp_intensity = utils.compute_dot_prod(grad, velocity)
    return ipp_intensity


  def _prepare_args(self, target_log_prob_fn,
                    state,
                    velocity,
                    preconditioner,
                    pre,
                    maybe_expand=False,
                    state_gradients_are_stopped=False):
    """Helper which processes input args to meet list-like assumptions.

    Much of this is directly copied from the HMC module in TFP. Have updated
    for BPS
    """
    state_parts, _ = mcmc_util.prepare_state_parts(state, name='current_state')
    # do similar thing for the velocity components
    velocity_parts, _ = mcmc_util.prepare_state_parts(velocity,
                                                      name='current_velocity')
    preconditioner_parts, _ = mcmc_util.prepare_state_parts(preconditioner,
                                                            name='current_preconditioner')
    pre_parts, _ = mcmc_util.prepare_state_parts(pre,
                                                 name='current_pre')
    def maybe_flatten(x):
      return x if maybe_expand or mcmc_util.is_list_like(state) else x[0]
    # returning the original state, as it is needed for computing the next
    # state/position in step (c) of Alg. 1 from [1]
    return [
      maybe_flatten(state_parts),
      maybe_flatten(velocity_parts),
      maybe_flatten(preconditioner_parts),
      maybe_flatten(pre_parts),
    ]


class IterPBPSKernel(PBPSKernel):
  """Transition kernel for PSBPS preconditioned gradients for
  BPS that handles an iter object to parse mini batches of data

  Is different to the normal kernels in BPS, in that their is no
  target_log_prob_fn. There is parent function that is called upon each
  iteration, which first iterates over the next batch of data within
  the model, and then returns to local target_log_prob_fn.

  An example would be,

  ```python
  def iter_bnn_neg_joint_log_prob(model, weight_prior_fns, bias_prior_fns, dataset_iter):
    def _fn():
      X, y = dataset_iter.next()
      return bnn_neg_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X, y)
    return _fn

  # create the kernel now
  kernel = IterBPSKernel(iter_bnn_neg_joint_log_prob, ...)
  # include any other args that are parsed to the normal BPSKernel
  ```
  This is different to the BPSKernel (or any other kernel in TFP)
  in that they would just have something like,

  ```python
  # create a callable of the neg joint log prob
  target_log_prob = bnn_neg_joint_log_prob_fn(model, weight_prior_fns,
                                              bias_prior_fns, X, y)
  # create the kernel now
  kernel = BPSKernel(target_log_prob, ...)
  # similarly handle any other args for normal BPS
  ```
  """
  def __init__(self,
               parent_target_log_prob_fn,
               beta=0.0,
               epsilon=0.0,
               ipp_sampler=PSBPSampler,
               lambda_ref=1.0,
               std_ref=1.0,
               batch_size=1.0,
               data_size=1.0,
               grad_target_log_prob=None,
               bounce_intensity=None,
               grad_bounce_intensity=None,
               state_gradients_are_stopped=False,
               seed=None,
               store_parameters_in_results=False,
               name=None):
    """Initializes this transition kernel.
    Args:
      parent_target_log_prob_fn: Python callable returns another Python callable
        which then takes an argument like `current_state`
        (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
        Refer to the class docstring for info and examples about how this
        differs from the normal kernel construction.
      lambda_ref (float):
        reference value for setting refresh rate
      state_gradients_are_stopped (bool):
        indicating that the proposed new state be run through
        `tf.stop_gradient`. This is particularly useful when combining
        optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      seed (int):
        Python integer to seed the random number generator.
      store_parameters_in_results (bool):
        If `True`, then `step_size` and `num_leapfrog_steps` are written
        to and read from eponymous fields in the kernel results objects
        returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    # need to initialise the target_log_prob_fn, so that the
    # bootstrap_results method can be called.
    self.parent_target_log_prob_fn = parent_target_log_prob_fn
    target_log_prob_fn = self.parent_target_log_prob_fn()
    super().__init__(target_log_prob_fn,
                     0.0,  # beta set too zero
                     0.0,  # epsilon set too zero
                     lambda_ref,
                     std_ref,
                     ipp_sampler,
                     batch_size,
                     data_size,
                     grad_target_log_prob,
                     bounce_intensity,
                     grad_bounce_intensity,
                     state_gradients_are_stopped,
                     seed,
                     store_parameters_in_results,
                     name)


  def one_step(self, previous_state, previous_kernel_results):
    """Will call the parent_target_log_prob_fn, which will get the
    next instance of the target_log_prob fn and set it. Will than call
    the parent class to perform a one step on this iteration.
    """
    # get the new local target_log_prob_fn
    self.target_log_prob_fn = self.parent_target_log_prob_fn()
    print('target_log_prob_fn = {}'.format(self.target_log_prob_fn))
    next_state_parts, next_kernel_results = super().one_step(previous_state,
                                                             previous_kernel_results)
    return next_state_parts, next_kernel_results


class CovPBPSKernel(PBPSKernel):
  """ Preconditioned BPS kernel with diagonal covariance matrix
  """
  def __init__(self,
               target_log_prob_fn,
               beta=0.01,
               epsilon=1e-5,
               ipp_sampler=PSBPSampler,
               lambda_ref=1.0,
               std_ref=0.0001,
               batch_size=1.0,
               data_size=1.0,
               grad_target_log_prob=None,
               bounce_intensity=None,
               grad_bounce_intensity=None,
               state_gradients_are_stopped=False,
               seed=None,
               store_parameters_in_results=False,
               name=None):
    """Initializes this transition kernel.
    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      beta (float):
        Discounting factor (similar to RMSProp)
      epsilon (float):
        small value to make sure we aren't dividing by zero
      lambda_ref (float):
        reference value for setting refresh rate
      state_gradients_are_stopped (bool):
        indicating that the proposed new state be run through
        `tf.stop_gradient`. This is particularly useful when combining
        optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      seed (int):
        Python integer to seed the random number generator.
      store_parameters_in_results (bool):
        If `True`, then `step_size` and `num_leapfrog_steps` are written
        to and read from eponymous fields in the kernel results objects
        returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    super().__init__(target_log_prob_fn,
                     0.0,  # beta set too zero
                     0.0,  # epsilon set too zero
                     lambda_ref,
                     std_ref,
                     ipp_sampler,
                     batch_size,
                     data_size,
                     grad_target_log_prob,
                     bounce_intensity,
                     grad_bounce_intensity,
                     state_gradients_are_stopped,
                     seed,
                     store_parameters_in_results,
                     name)

  def collision_velocity(self, next_state, current_velocity_parts,
                         preconditioner,
                         not_used):
    """update the velocity based on simulated collision with preconditioner


    Refer to `BPSKernel.collision_velocity` for more info in type of collision.

    Is updated here to include preconditioner as similar to what is done in the
    SBPS paper, but am using the covariance as the scaler for it.

    TODO: Refer to issue 39, which explins more about why it needs to be updated.
    Args:
      next_state (list(array)):
        next position for which we need to compute collision direction
      current_velocity_parts (list(array)):
        parts of the current velocity
      preconditioner (list(array)):
        parts of the preconditioner
    Returns:
      the updated velocity for the next step based on collision dynamics with
      preconditioner used in SBPS paper.
    """
    tf.print('am bouncing')
    # need to compute the grad for the newest position
    grads_target_log_prob = self.target_log_prob_fn(next_state)
    preconditioned_grads = [tf.math.multiply(a, g) for a, g in zip(preconditioner, grads_target_log_prob)]
    grads_norm = utils.compute_l2_norm(preconditioned_grads)
    # now need to compute the inner product of the grads_target and the velocity
    dot_pre_grad_velocity = utils.compute_dot_prod(preconditioned_grads,
                                                   current_velocity_parts)
    # can now compute the new velocity from the simulated collision
    new_v = [v - 2. * u * dot_pre_grad_velocity / grads_norm for u, v in zip(
      preconditioned_grads, current_velocity_parts)]
    return (new_v, preconditioner, not_used)


class IterCovPBPSKernel(CovPBPSKernel):
  """Transition kernel for covariance preconditioned gradients for
  BPS that handles an iter object to parse mini batches of data

  Is different to the normal kernels in BPS, in that their is no
  target_log_prob_fn. There is parent function that is called upon each
  iteration, which first iterates over the next batch of data within
  the model, and then returns to local target_log_prob_fn.

  An example would be,

  ```python
  def iter_bnn_neg_joint_log_prob(model, weight_prior_fns, bias_prior_fns, dataset_iter):
    def _fn():
      X, y = dataset_iter.next()
      return bnn_neg_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X, y)
    return _fn

  # create the kernel now
  kernel = IterBPSKernel(iter_bnn_neg_joint_log_prob, ...)
  # include any other args that are parsed to the normal BPSKernel
  ```
  This is different to the BPSKernel (or any other kernel in TFP)
  in that they would just have something like,

  ```python
  # create a callable of the neg joint log prob
  target_log_prob = bnn_neg_joint_log_prob_fn(model, weight_prior_fns,
                                              bias_prior_fns, X, y)
  # create the kernel now
  kernel = BPSKernel(target_log_prob, ...)
  # similarly handle any other args for normal BPS
  ```
  """
  def __init__(self,
               parent_target_log_prob_fn,
               beta=0.0,
               epsilon=0.0,
               lambda_ref=1.0,
               std_ref=0.0001,
               ipp_sampler=PSBPSampler,
               batch_size=1.0,
               data_size=1.0,
               grad_target_log_prob=None,
               bounce_intensity=None,
               grad_bounce_intensity=None,
               state_gradients_are_stopped=False,
               seed=None,
               store_parameters_in_results=False,
               name=None):
    """Initializes this transition kernel.
    Args:
      parent_target_log_prob_fn: Python callable returns another Python callable
        which then takes an argument like `current_state`
        (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
        Refer to the class docstring for info and examples about how this
        differs from the normal kernel construction.
      lambda_ref (float):
        reference value for setting refresh rate
      state_gradients_are_stopped (bool):
        indicating that the proposed new state be run through
        `tf.stop_gradient`. This is particularly useful when combining
        optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      seed (int):
        Python integer to seed the random number generator.
      store_parameters_in_results (bool):
        If `True`, then `step_size` and `num_leapfrog_steps` are written
        to and read from eponymous fields in the kernel results objects
        returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    # need to initialise the target_log_prob_fn, so that the
    # bootstrap_results method can be called.
    self.parent_target_log_prob_fn = parent_target_log_prob_fn
    target_log_prob_fn = self.parent_target_log_prob_fn()
    super().__init__(target_log_prob_fn,
                     beta=beta,  # beta set too zero
                     epsilon=epsilon,  # epsilon set too zero
                     lambda_ref=lambda_ref,
                     std_ref=std_ref,
                     ipp_sampler=ipp_sampler,
                     batch_size=batch_size,
                     data_size=data_size,
                     grad_target_log_prob=grad_target_log_prob,
                     bounce_intensity=bounce_intensity,
                     grad_bounce_intensity=grad_bounce_intensity,
                     state_gradients_are_stopped=state_gradients_are_stopped,
                     seed=seed,
                     store_parameters_in_results=store_parameters_in_results,
                     name=name)


  def one_step(self, previous_state, previous_kernel_results):
    """Will call the parent_target_log_prob_fn, which will get the
    next instance of the target_log_prob fn and set it. Will than call
    the parent class to perform a one step on this iteration.
    """
    # get the new local target_log_prob_fn
    self.target_log_prob_fn = self.parent_target_log_prob_fn()
    print('target_log_prob_fn = {}'.format(self.target_log_prob_fn))
    next_state_parts, next_kernel_results = super().one_step(previous_state,
                                                             previous_kernel_results)
    return next_state_parts, next_kernel_results



class BoomerangKernel(BPSKernel):
  """ boomerang sampler
  """
  def __init__(self,
               target_log_prob_fn,
               preconditioner,
               mean,
               epsilon=0.00001,
               lambda_ref=1.0,
               std_ref=0.0001,
               ipp_sampler=SBPSampler,
               batch_size=1.0,
               data_size=1.0,
               grad_target_log_prob=None,
               bounce_intensity=None,
               grad_bounce_intensity=None,
               state_gradients_are_stopped=False,
               seed=None,
               store_parameters_in_results=False,
               name=None):
    """Initializes this transition kernel.
    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      preconditioner (list(tf.tensor)):
        diagonal approx of covariance used for preconditioning
      mean (list(tf.tensor)):
        mean of the preconditioner (or the MAP estimate)
      epsilon (float):
        small value to make sure we aren't dividing by zero
      lambda_ref (float):
        reference value for setting refresh rate
      std_ref (float):
        std. for reference dist
      state_gradients_are_stopped (bool):
        indicating that the proposed new state be run through
        `tf.stop_gradient`. This is particularly useful when combining
        optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      seed (int):
        Python integer to seed the random number generator.
      store_parameters_in_results (bool):
        If `True`, then `step_size` and `num_leapfrog_steps` are written
        to and read from eponymous fields in the kernel results objects
        returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
    super().__init__(target_log_prob_fn,
                     lambda_ref=lambda_ref,
                     std_ref=std_ref,
                     ipp_sampler=ipp_sampler,
                     batch_size=batch_size,
                     data_size=data_size,
                     grad_target_log_prob=grad_target_log_prob,
                     bounce_intensity=bounce_intensity,
                     grad_bounce_intensity=grad_bounce_intensity,
                     state_gradients_are_stopped=state_gradients_are_stopped,
                     seed=seed,
                     store_parameters_in_results=store_parameters_in_results,
                     name=name)
    self.epsilon = epsilon
    self.preconditioner = preconditioner
    self.preconditioner_sqrt = [tf.math.sqrt(x) for x in self.preconditioner]
    self.mean = mean


  def one_step(self, previous_state, previous_kernel_results):
    """ performs update for preconditioned Bouncy Particle Sampler

    Very similar to method from the BPSKernel.one_step method,
    though is re-implemented here as there are new params required
    to send through to each of the update methods due to the inclusion
    of the preconditioner.

    For more info on the ins and outs of this method, please refer to
    the BPSKernel.one_step docstring.
    """
    with tf.name_scope(mcmc_util.make_name(self.name, 'pbps', 'one_step')):
      # preparing all args first
      # very similar to the HMC module in TFP
      # we are passing the values for target and the gradient from their
      # previous sample to ensure that the maybe_call_fn_and_grads won't
      # compute the target.
      # will be dooing it manually because we want to find the gradient w.r.t
      # parameters and with the time
      tf.print('start one step boomerang', output_stream=sys.stdout)
      [
        previous_state_parts,
        previous_velocity_parts,
      ] = self._prepare_args(
        self.target_log_prob_fn,
        previous_state,
        previous_kernel_results.velocity,
        maybe_expand=True,
        state_gradients_are_stopped=self.state_gradients_are_stopped)
      # simulate the arrival of the first event in our IPP
      t_bounce, acceptance_ratio = self.ipp_sampler.simulate_bounce_time(
        self.target_log_prob_fn,
        previous_state_parts,
        previous_velocity_parts)
      # simulate from our reference distribution
      t_ref = tf.reshape(self.ref_dist.sample(1), ())
      tf.print('ht_bounce = {}'.format(t_bounce))
      # (c) set the time and update to the next position
      time = tf.math.minimum(t_bounce, t_ref)
      # get the next state and the end velocity at the time of the update
      next_state_parts, end_velocity = self.compute_next_step_and_end_velocity(
        previous_state_parts,
        previous_velocity_parts,
        time)
      # now need to update the velocity, using either the value for the velocity and
      # state, or use refreshment.
      next_velocity_parts = self.compute_next_velocity(
        next_state_parts,
        end_velocity,
        time,
        t_bounce,
        t_ref)
      # now save the next state and velocity in the kernel results
      new_kernel_results = previous_kernel_results._replace(
        velocity=next_velocity_parts,
        time=time,
        proposed_time=t_bounce,
        acceptance_ratio=acceptance_ratio)

      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(previous_state) else x[0]
      return maybe_flatten(next_state_parts), new_kernel_results


  def compute_next_step_and_end_velocity(self, state, velocity, time):
    """ updates the current state with the velocity and time found"""
    next_step = [mu +  (s - mu) * tf.cos(time)  + v * tf.sin(time)
                 for mu, s, v in zip(self.mean, state, velocity)]
    end_velocity = [-(s - mu) * tf.sin(time)  + v * tf.cos(time)
                 for mu, s, v in zip(self.mean, state, velocity)]
    return next_step, end_velocity


  def compute_next_velocity(self,
                            next_state,
                            end_velocity,
                            time,
                            t_bounce,
                            t_ref):
    """update the velocity based on the current times with preconditioner

    if t == t_bounce:
      update using the gradient of potential
      (Newtonian Elastic Collision)
    else if t == t_ref:
      refresh by sampling from a normal distribution

    Args:
      next_step (List(tf.Tensor)):
        the start of the next state in our updated trajectory
      end_velocity (List(tf.Tensor)):
        velocity just before the new updated trajectory
      current_pre_parts (List(tf.Tensor)):
        current "pre" parts for the pSBPS update, refer to class docstring
        for an elaboration on this var.
      current_preconditioner (List(tf.Tensor)):
        current preconditioner matrix
      time (tf.float):
        the time that will be used for the new trajectory
      t_bounce (tf.float):
        the proposed bounce time in our dynamics
      t_ref (tf.float):
        the sampled reference time

    Returns:
      updated velocity for the next step and updated preconditioner components

    """
    print('t_ref = {}, t_bounce = {}, time = {}'.format(t_ref, t_bounce, time))
    refresh = lambda: self.refresh_velocity(end_velocity)
    bounce = lambda: self.collision_velocity(next_state, end_velocity)
    next_velocity = tf.cond(time == t_ref, refresh, bounce)
    return next_velocity


  def refresh_velocity(self, end_velocity):
    """ Use refresh step for updating the velocity component

    Pretty much the exact same as the BPSKernel method.
    Am only re-implementing it here as it now takes in a second
    argument for the preconditioner, but it won't update it on
    a refresh step.

    Args:
      end_velocity (list(array)):
        parts of the current velocity at the time of proposed update
    Returns:
      Next velocity sampled from a Normal dist and will return the
      input preconditioner (won't update it here)
    """
    tf.print('am refreshing')
    # uniform = tfd.Uniform(low=-1.0, high=1.0)
    # new_v = [uniform.sample(x.shape) for x in end_velocity]
    # new_norm = tf.sqrt(utils.compute_l2_norm(new_v))
    # new_v = [v / new_norm for v in new_v]

    # uniform = tfd.Normal(loc=0.0, scale=self.std_ref)
    # new_v = [uniform.sample(x.shape) for x in end_velocity]
    dist_list = [tfd.Normal(loc=0.0, scale=tf.math.sqrt(v))
                 for v in self.preconditioner]
    new_v = [d.sample() for d in dist_list]
    return (new_v)


  def collision_velocity(self, next_state, end_velocity):
    """update the velocity based on simulated collision with preconditioner

    Within the boomerang sampler, the preconditioner is defined by the covariance
    of the reference measure.
    Here we are just using the diagonal of it (variance), and is stored as a
    class variable as does not change.

    Follows from the format used in the boomerang sampler paper.
    https://github.com/jbierkens/ICML-boomerang

    Args:
      next_state (list(array)):
        next position for which we need to compute collision direction
      end_velocity (list(array)):
        parts of the current velocity at the time right before the update time

    Returns:
      the updated velocity for the next step based on collision dynamics with
      preconditioner used in boomerang paper.
    """
    tf.print('am boomeranging')
    # need to compute the grad for the newest position
    grads_target_log_prob = self.target_log_prob_fn(next_state)
    # going to follow naming convention similar to boomerang code
    # switch_rate = <grad_U, v>
    # implementing dot product with a bunch of sums
    switch_rate = tf.math.reduce_sum(
      [tf.math.reduce_sum(tf.math.multiply(g, v))
       for g, v in zip(grads_target_log_prob, end_velocity)])
    # skewed grads = <Sigma^{1/2}, grad_U>
    # here, preconditioner = Sigma
    skewed_grads = [tf.math.multiply(s, g)
                    for s, g in zip(self.preconditioner_sqrt,
                                    grads_target_log_prob)]
    # implements |Sigma^(1/2) grad_U|^2
    # now computing  |Sigma^(1/2) grad_U|^2
    # or written as |skewed_grads|^2
    denominator = tf.math.reduce_sum(
      [tf.math.reduce_sum(tf.math.multiply(sg, sg)) for sg in skewed_grads])
    # putting it all together
    new_v = [v - (2.0 * switch_rate / denominator) * (pre_sqrt * g)
             for v, pre_sqrt, g in zip(
                 end_velocity,
                 self.preconditioner_sqrt,
                 skewed_grads)]
                 # grads_target_log_prob)]
    return new_v



class BoomerangIterKernel(BoomerangKernel):
  """ boomerang sampler
  """
  def __init__(self,
               parent_target_log_prob_fn,
               preconditioner,
               mean,
               epsilon=0.00001,
               lambda_ref=1.0,
               std_ref=0.001,
               ipp_sampler=SBPSampler,
               batch_size=1.0,
               data_size=1.0,
               grad_target_log_prob=None,
               bounce_intensity=None,
               grad_bounce_intensity=None,
               state_gradients_are_stopped=False,
               seed=None,
               store_parameters_in_results=False,
               name=None):
   """Initializes this transition kernel.
    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      preconditioner (list(tf.tensor)):
        diagonal approx of covariance used for preconditioning
      mean (list(tf.tensor)):
        mean of the preconditioner (or the MAP estimate)
      epsilon (float):
        small value to make sure we aren't dividing by zero
      lambda_ref (float):
        reference value for setting refresh rate
      state_gradients_are_stopped (bool):
        indicating that the proposed new state be run through
        `tf.stop_gradient`. This is particularly useful when combining
        optimization over samples from the HMC chain.
        Default value: `False` (i.e., do not apply `stop_gradient`).
      seed (int):
        Python integer to seed the random number generator.
      store_parameters_in_results (bool):
        If `True`, then `step_size` and `num_leapfrog_steps` are written
        to and read from eponymous fields in the kernel results objects
        returned from `one_step` and
        `bootstrap_results`. This allows wrapper kernels to adjust those
        parameters on the fly.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'hmc_kernel').
    """
   # bootstrap_results method can be called.
   self.parent_target_log_prob_fn = parent_target_log_prob_fn
   target_log_prob_fn = self.parent_target_log_prob_fn()
   super().__init__(target_log_prob_fn,
                    preconditioner,
                    mean,
                    lambda_ref=lambda_ref,
                    std_ref=std_ref,
                    ipp_sampler=ipp_sampler,
                    batch_size=batch_size,
                    data_size=data_size,
                    grad_target_log_prob=grad_target_log_prob,
                    bounce_intensity=bounce_intensity,
                    grad_bounce_intensity=grad_bounce_intensity,
                    state_gradients_are_stopped=state_gradients_are_stopped,
                    seed=seed,
                    store_parameters_in_results=store_parameters_in_results,
                    name=name)

  def one_step(self, previous_state, previous_kernel_results):
    """Will call the parent_target_log_prob_fn, which will get the
    next instance of the target_log_prob fn and set it. Will than call
    the parent class to perform a one step on this iteration.
    """
    # get the new local target_log_prob_fn
    self.target_log_prob_fn = self.parent_target_log_prob_fn()
    print('target_log_prob_fn = {}'.format(self.target_log_prob_fn))
    next_state_parts, next_kernel_results = super().one_step(previous_state,
                                                             previous_kernel_results)
    return next_state_parts, next_kernel_results
