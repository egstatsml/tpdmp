"""Fit conv. nets with PDMDs."""
import contextlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
from sklearn.metrics import accuracy_score
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import tensorflow_probability as tfp
import tensorflow_addons as tfa
from tbnn.pdmp.bps import IterBPSKernel

from tbnn.pdmp.poisson_process import (SBPSampler, PSBPSampler, AdaptivePSBPSampler, AdaptiveSBPSampler, InterpolationSampler)
from tbnn.pdmp.networks import get_network
from tbnn.pdmp import networks

from tbnn.pdmp import agc
import tensorflow_models as tfm

from tbnn.pdmp.model import (
    get_map_test, plot_density, get_model_state, pred_forward_pass, get_map,
    get_mle, trace_fn, graph_hmc, nest_concat, set_model_params, build_network, set_model_params_hmc,
    bps_iter_main, hmc_main, nuts_main, boomerang_iter_main, pbps_iter_main,
    get_map_iter, sgld_iter_main, save_map_weights,
    cov_pbps_test_iter_main, boomerang_test_iter_main, variational_iter_main, hmc_iter_main)
# import some helper functions
#
from tbnn.pdmp.nfnets import ScaledStandardizedConv2D
from tbnn.utils import utils
from tbnn.pdmp.utils import compute_dot_prod

import tensorflow_addons as tfa

import argparse
import os
import neptune
import sys
import time
import pickle

tfd = tfp.distributions

VALID_OPTIMIZER_STRS = ['sgd', 'adam', 'rmsprop']


class NoDistribute():
  """Implements a scope method that does nothing

  Is usefule for when working with HMC from tfp, which doesn't support
  strategies.
  """
  scope = contextlib.nullcontext

  def experimental_distribute_dataset(self, dataset):
    return dataset

class MyLR(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self,
               init_lr,
               warm_up_steps=2500.0,
               decay_steps=140625.0,
               name=None):
    super().__init__()
    self.init_lr = tf.convert_to_tensor(init_lr)
    self.warm_up_steps = tf.convert_to_tensor(warm_up_steps)
    self.decay_steps = tf.convert_to_tensor(decay_steps)

  def __call__(self, step):
    if step < self.warm_up_steps:
      lr = self.init_lr * step / self.warm_up_steps
    else:
      step = min(step, self.decay_steps)
      cosine_decay = 0.5 * (
          1 + tf.math.cos(np.pi *
                          (step - self.warm_up_steps) / self.decay_steps))
      lr = (self.init_lr) * cosine_decay  #+ self.init_lr
    return lr


def get_data(data, batch_size, strategy):
  dataset = utils.load_dataset(data, batch_size)
  # create distributed version of it as required by the
  # distribution strategy
  training_data = strategy.experimental_distribute_dataset(dataset.train)
  # now create an iter object of it

  training_iter = iter(training_data)
  # training_iter = training_data
  return (training_iter, training_data, dataset.test, dataset.test_orig,
          dataset.dimension_dict, dataset.label_dict)


def classification_accuracy(model, dataset):
  """get classification accuracy
    Args:
      model (keras.Model):
        neural network model to evaluate
      dataset (tf.Dataset):
        data to evaluate on
    Returns:
      classification accuracy
    """
  classification_array, correct_labels = classify(model, dataset)
  # now find the accuracy
  accuracy = accuracy_score(correct_labels,
                            classification_array,
                            normalize=True)
  # want to do some testing on the way to handle it all with tensorflow datasets.
  # want to know if I can zip datasets together to let me know which one's I classified
  # correctly.

  # first need to unbatch the dataset.
  dataset_unbatched = dataset.unbatch()
  # now want to create a dataset that has the predicted labels.
  dataset_pred = tf.data.Dataset.from_tensor_slices(classification_array)
  # now want to zip them together
  dataset_combined = tf.data.Dataset.zip((dataset_unbatched, dataset_pred))
  # let's get the cardinality of this first so we know how many samples there are
  num_samples = dataset_combined.cardinality()
  # now I want to filter them based on the one's that were incorectly classified.
  # tmp =  dataset_combined.take(1)
  # for tmp_samp in tmp.as_numpy_iterator():
  #   print(f"lookin at single sample {tmp_samp[0]}")
  #   print(f"lookin at single pred {tmp_samp[1]}")
  dataset_incorrect = dataset_combined.filter(
      lambda x_and_y, pred: tf.logical_not(
          tf.math.equal(tf.argmax(x_and_y[1], axis=0), pred)))
  num_incorrect = 0
  for x, y in dataset_incorrect.as_numpy_iterator():
    num_incorrect += 1
    # let's see how many of these were included, or how many where incorrectly classified
    # num_incorrect = dataset_incorrect.cardinality()
  print(num_incorrect, num_samples)
  print(f'dataset acc = {num_incorrect/num_samples}')
  return accuracy


def classify(model, dataset):
  """Classify all the examples

    Args:
    input_ (np.array):
      input data to evaluate
    labels (np.array):
      true labels corresponding to the inputs

    Returns
    -------
    classification_array (np.array):
      list of what the classifier actually predicted
    correct_prediction_eval (np.array):
      list of what the correct prediction labels should be
    """
  classification_list = []
  label_list = []
  # classify each batch and store the results
  for input_batch, labels_batch in dataset:
    output_ = model(input_batch)
    #print('output_ = {}'.format(output_))
    classification_list.extend(np.argmax(output_, axis=1))
    label_list.extend(np.argmax(labels_batch, axis=1))
    classification_array = np.array(classification_list)
    correct_prediction = np.array(label_list)

  return classification_array, correct_prediction


# @tf.function
#


def examine_sample_graph(kernel, init_state):
  bps_results = tfp.mcmc.sample_chain(num_results=1,
                                      current_state=init_state,
                                      return_final_kernel_results=True,
                                      kernel=kernel)
  samples = bps_results.all_states
  # initialise stafe for next iter
  init_state = [x[-1, ...] for x in samples]
  # final kernel results used to initialise next call of loop
  kernel_results = bps_results.final_kernel_results
  velocity = kernel_results.velocity
  # now iterate over the time steps to evaluate the
  #
  i = tf.constant(0, dtype=tf.float32)
  time_dt = tf.constant(0.2, dtype=tf.float32)
  time = tf.Variable(0.0, dtype=tf.float32)
  cond = lambda G, i: tf.less(i, 10)
  G = tf.TensorArray(tf.float32,
                     size=0,
                     dynamic_size=True,
                     clear_after_read=False)

  def examine_loop(state, velocity, time, time_dt, G, i):
    t = time + time_dt * tf.cast(i, tf.float32)
    test = kernel.examine_event_intensity(state, velocity, t)
    G = G.write(tf.cast(i, tf.int32), test)
    i = i + 1
    return G, i

  G, i = tf.while_loop(cond,
                       lambda G_l, i_l: examine_loop(init_state, velocity, time,
                                                     time_dt, G_l, i_l),
                       loop_vars=(G, i))
  return G


@tf.function
def examine_grad_fn(param_list, model, likelihood_fn, X, y, strategy):
  per_replica_gradient = strategy.run(gradient_step,
                                      args=(
                                          model,
                                          likelihood_fn,
                                          X,
                                          y,
                                          param_list,
                                      ))
  return strategy.reduce(tf.distribute.ReduceOp.SUM,
                         per_replica_gradient,
                         axis=None)


def examine_rate(model,
                 parent_bnn_neg_joint_log_prob,
                 training_iter,
                 likelihood_fn,
                 strategy,
                 state,
                 out_dir,
                 num_samp=200):
  kernel = IterBPSKernel(
      # ipp_sampler=SBPSampler,
      ipp_sampler=InterpolationSampler,
      parent_target_log_prob_fn=parent_bnn_neg_joint_log_prob,
      store_parameters_in_results=True,
      lambda_ref=1000.0)
  init_state = [tf.convert_to_tensor(x) for x in state]
  trace_fn = lambda _, pkr: (pkr.acceptance_ratio, pkr.time, pkr.proposed_time)
  kernel_previous_results = kernel.bootstrap_results(init_state)
  for test_iter in range(0, 10):
    print('eval loop {}'.format(test_iter))
    bps_results = graph_hmc(num_results=1,
                            current_state=init_state,
                            kernel=kernel,
                            previous_kernel_results=kernel_previous_results,
                            return_final_kernel_results=True,
                            num_steps_between_results=1,
                            trace_fn=trace_fn)
    # bps_results = examine_sample_graph(kernel, init_state)
    print('examined graph')
    # timepy.sleep(10)
    samples = bps_results.all_states
    traced = bps_results.trace
    # initialise stafe for next iter
    init_state = [x[-1, ...] for x in samples]
    # final kernel results used to initialise next call of loop
    kernel_results = bps_results.final_kernel_results
    velocity = kernel_results.velocity
    # now iterate over the time steps to evaluate the
    time_dt = tf.constant(0.02, dtype=tf.float32)
    time = tf.Variable(0.0, dtype=tf.float32)
    test = np.zeros(num_samp)
    X, y = next(training_iter)
    # grad_fn = examine_grad_fn(model, likelihood_fn, X, y, strategy)
    for i in range(0, num_samp):
      time = time + time_dt
      updated_state = [s + v * time for s, v in zip(state, velocity)]
      grad = examine_grad_fn(updated_state, model, likelihood_fn, X, y,
                             strategy)
      ipp_intensity = compute_dot_prod(grad, velocity).numpy()
      test[i] = ipp_intensity
      time = time + time_dt
      time_arr = np.linspace(0, time_dt.numpy() * num_samp, num_samp)
    plt.figure()
    plt.plot(time_arr, test)
    plt.xlabel('time')
    plt.ylabel('IPP intensity')
    plt.savefig(os.path.join(out_dir, 'ipp_test_{}.png'.format(test_iter)))
    np.save(os.path.join(out_dir, 'conv_time_array_{}.npy'.format(test_iter)),
            time_arr)
    np.save(os.path.join(out_dir, 'conv_test_array_{}.npy'.format(test_iter)),
            test)


def neg_log_likelihood(model, likelihood_fn, X, y, strategy):
  """Compute the neg log likelihood of a model.

    This function expects that all model parameters have already
    been set.

    Parameters
    ----------
    model : keras.Moodel
        model we are evaluating.
    likelihood_fn : tfp.Distribution
        Distribution that defines the log likelihood of our model.
    X : tf.Tensor
        Our input data.
    y : tf.Tensor
        output label's.

    Returns
    -------
    Neg log likelihood.
    """
  logits = model(X)
  # add the log likelihood now
  log_likelihood_dist = likelihood_fn(logits)
  lp = tf.reduce_sum(log_likelihood_dist.log_prob(y))
  return -1.0 * lp


def bnn_neg_joint_log_prob_fn(model, likelihood_fn, X, y):
  """Compute neg joint log prob for model.

    Neg joint log prob defined as

    Parameters
    ----------
    model : keras.Model
        Model that helps define likelihood.
    likelihood_fn : tfp.Distribution
        Distribution with the log_prob method.
    X : tf.Tensor
        input array
    y : tf.Tensor
        output label's.

    Returns
    -------
    callable that will compute the neg joint log prob, that requires model parameters to be supplied as an argument.
    """

  def _fn(*param_list):
    with tf.name_scope('bnn_joint_log_prob_fn'):
      # set the model params
      m = set_model_params(model, param_list)
      # print('current  model params ttttt= {}'.format(model.layers[1].kernel))
      # neg log likelihood of predicted labels
      neg_ll = neg_log_likelihood(m, likelihood_fn, X, y)
      # now get the losses from the prior (negative log prior)
      # these are stored within the models `losses` variable
      neg_lp = tf.reduce_sum(m.losses)
      # add them together for the total loss
      return neg_ll + neg_lp

  return _fn


def bnn_hmc_neg_joint_log_prob_fn(model, likelihood_fn, X, y):
  """Compute neg joint log prob for model for HMC

    Is essentially the same as the neg joint log prob used
    for pdmp methods, but uses a different method to set model params.


    Parameters
    ----------
    model : keras.Model
        Model that helps define likelihood.
    likelihood_fn : tfp.Distribution
        Distribution with the log_prob method.
    X : tf.Tensor
        input array
    y : tf.Tensor
        output label's.

    Returns
    -------
    callable that will compute the neg joint log prob, that requires model parameters to be supplied as an argument.
    """

  def _fn(*param_list):
    with tf.name_scope('bnn_joint_log_prob_fn'):
      # set the model params
      m = set_model_params(model, param_list)
      # print('current  model params ttttt= {}'.format(model.layers[1].kernel))
      # neg log likelihood of predicted labels
      neg_ll = neg_log_likelihood(m, likelihood_fn, X, y)
      # now get the losses from the prior (negative log prior)
      # these are stored within the models `losses` variable
      neg_lp = tf.reduce_sum(m.losses)
      # add them together for the total loss
      return neg_ll + neg_lp

  return _fn




def loss_object_opt(model, likelihood_fn, X, y):
  with tf.name_scope('loss_object_opt'):
    return neg_joint(model, likelihood_fn, X, y)
  #pred = model(X)
  #loss = likelihood_fn(y, pred)
  #return loss


def neg_joint(model, likelihood_fn, X, y):
  with tf.name_scope('neg_joint'):
    # neg log likelihood of predicted labels
    #neg_ll = neg_log_likelihood(model, likelihood_fn, X, y)
    pred = model(X)
    neg_ll = likelihood_fn(y, pred)
    # now get the losses from the prior (negative log prior)
    # these are stored within the models `losses` variable
    neg_lp = tf.nn.scale_regularization_loss(tf.reduce_sum(model.losses))
    # add them together for the total loss
    return neg_ll + neg_lp


def loss_object_sampling(model, likelihood_fn, X, y, param_list):
  with tf.name_scope('loss_object_sampling'):
    # first set the model params
    model = set_model_params(model, param_list)
    return neg_joint(model, likelihood_fn, X, y)


def loss_object(model, likelihood_fn, X, y):
  with tf.name_scope('bnn_joint_log_prob_fn'):
    # neg log likelihood of predicted labels
    neg_ll = likelihood_fn(y, pred)
    # now get the losses from the prior (negative log prior)
    # these are stored within the models `losses` variable
    neg_lp = tf.reduce_sum(model.losses)
    # add them together for the total loss
    return neg_ll + neg_lp


def iter_bnn_neg_joint_log_prob(model, likelihood_fn, dataset_iter):

  def _fn():
    X, y = dataset_iter.next()
    return bnn_neg_joint_log_prob_fn(model, likelihood_fn, X, y)

  return _fn


def gradient_step(model, likelihood_fn, X, y, param_list):
  # set the model params
  print(f'grad_step param list shape={len(param_list)}')
  print(f'grad_step : {param_list[-1]}')
  # m = model
  model = set_model_params(model, param_list)
  # print(param_list)
  with tf.GradientTape() as tape:
    # for i in range(0, len(param_list)):
    #   tape.watch(param_list[i])
    neg_log_prob = neg_joint(model, likelihood_fn, X, y)
    gradients = tape.gradient(neg_log_prob,  model.trainable_variables)
  return gradients



def hmc_step(model, likelihood_fn, X, y, param_list):
  """For HMC, just want to compute the neg joint log prob"""
  # set the model params
  print(f'grad_step param list shape={len(param_list)}')
  print(f'grad_step : {param_list[-1]}')
  model = set_model_params_hmc(model, param_list)
  neg_log_prob = neg_joint(model, likelihood_fn, X, y)
  # need to remove the negative for HMC with tfp
  return -1.0 * neg_log_prob


def opt_step(compute_loss, model, X, y, optimizer):
  # set the model params
  with tf.GradientTape() as tape:
    neg_log_prob = compute_loss(X, y)
    # predictions = model(X, training=True)
    # neg_ll = compute_loss(y, predictions)
  gradients = tape.gradient(neg_log_prob, model.trainable_variables)
  # grad_mean = [tf.reduce_mean(x).numpy() for x in gradients]
  # grad_min = [tf.reduce_min(x).numpy() for x in gradients]
  # grad_max = [tf.reduce_max(x).numpy() for x in gradients]
  # print('grad max')
  # print(grad_max)
  # print('grad mean')
  # print(grad_mean)
  # print('grad min')
  # print(grad_min)



  # agc_gradients = agc.adaptive_clip_grad(model.trainable_variables,
  #                                        gradients,
  #                                        clip_factor=0.01,
  #                                        eps=1e-3)
  # optimizer.apply_gradients(zip(agc_gradients, model.trainable_variables))
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  # print(model.layers[2].kernel[:, :, 0, 0])
  # print(gradients[0][:, :, 0, 0])
  return neg_log_prob



def sgld_step(compute_loss, model, X, y, optimizer):
  # set the model
  with tf.GradientTape() as tape:
    neg_log_prob = compute_loss(X, y)
    # predictions = model(X, training=True)
    # neg_ll = compute_loss(y, predictions)
  gradients = tape.gradient(neg_log_prob, model.trainable_variables)
  # add some Gaussian noise to the gradients, where the variance is
  # 2 times the learning rate for the optimizer
  dist = tfd.Normal(loc=0.0,
                    scale=tf.math.sqrt(optimizer.learning_rate * 2))

  grad_noise = [dist.sample(x.shape) for x in gradients]
  noisy_grads = [g + n for g, n in zip(gradients, grad_noise)]
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return neg_log_prob



def distributed_gradient_step(model, likelihood_fn, X, y, strategy):

  def _fn(param_list):
    per_replica_gradient = strategy.run(gradient_step,
                                        args=(
                                            model,
                                            likelihood_fn,
                                            X,
                                            y,
                                            param_list,
                                        ))
    return strategy.reduce(tf.distribute.ReduceOp.SUM,
                           per_replica_gradient,
                           axis=None)
  return _fn


def distributed_hmc_step(model, likelihood_fn, X, y, strategy):

  def _fn(param_list):
    per_replica_gradient = strategy.run(hmc_step,
                                        args=(
                                            model,
                                            likelihood_fn,
                                            X,
                                            y,
                                            param_list,
                                        ))
    return strategy.reduce(tf.distribute.ReduceOp.SUM,
                           per_replica_gradient,
                           axis=None)
  return _fn



def hmc_step_wrapper(model, likelihood_fn, X, y, strategy):

  def _fn(*param_list):
    joint_log_prob = hmc_step(model,
                              likelihood_fn,
                              X,
                              y,
                              param_list)
    return joint_log_prob
  return _fn




@tf.function
def distributed_opt_step(compute_loss, model, X, y, strategy, optimizer):
  per_replica_loss = strategy.run(opt_step,
                                  args=(
                                      compute_loss,
                                      model,
                                      X,
                                      y,
                                      optimizer,
                                  ))
  return strategy.reduce(tf.distribute.ReduceOp.SUM,
                         per_replica_loss,
                         axis=None)



@tf.function
def distributed_sgld_step(compute_loss, model, X, y, strategy, optimizer):
  per_replica_loss = strategy.run(sgld_step,
                                  args=(
                                      compute_loss,
                                      model,
                                      X,
                                      y,
                                      optimizer,
                                  ))
  return strategy.reduce(tf.distribute.ReduceOp.SUM,
                         per_replica_loss,
                         axis=None)



def iter_grad_fn(model, likelihood_fn, dataset_iter, strategy):

  def _fn():
    X, y = dataset_iter.next()
    return distributed_gradient_step(model, likelihood_fn, X, y, strategy)

  return _fn


def iter_opt_fn(model, loss_fn, dataset_iter, strategy, optimizer):

  def _fn():
    X, y = next(dataset_iter)  #.next()
    return distributed_opt_step(loss_fn, model, X, y, strategy, optimizer)

  return _fn


def iter_sgld_fn(loss_fn, model, dataset_iter, strategy, optimizer):

  def _fn():
    X, y = dataset_iter.next()
    return distributed_sgld_step(loss_fn, model, X, y, strategy, optimizer)

  return _fn

def iter_hmc_fn(model, likelihood_fn, dataset_iter, strategy):

  def _fn():
    X, y = dataset_iter.next()
    return hmc_step_wrapper(model, likelihood_fn, X, y, strategy)
  return _fn



def hessian_diag_step(model, likelihood_fn, X, y):
  with tf.GradientTape() as tape_hessian:
    with tf.GradientTape() as tape_grad:
      neg_log_prob = neg_joint(model, likelihood_fn, X, y)
    gradients = tape_grad.gradient(neg_log_prob, model.trainable_variables)
  hessian_diag = tape_hessian.gradient(gradients, model.trainable_variables)
  return hessian_diag


@tf.function
def distributed_hessian_diag_step(model, likelihood_fn, X, y, strategy):
  per_replica_gradient = strategy.run(hessian_diag_step,
                                      args=(
                                          model,
                                          likelihood_fn,
                                          X,
                                          y,
                                      ))
  return strategy.reduce(tf.distribute.ReduceOp.SUM,
                         per_replica_gradient,
                         axis=None)


def iter_hessian_diag_fn(model, likelihood_fn, dataset_iter, strategy):

  def _fn():
    X, y = dataset_iter.next()
    return distributed_hessian_diag_step(model, likelihood_fn, X, y, strategy)

  return _fn


def create_loss_fn(model, likelihood_fn, args, strategy):
  with strategy.scope():
    denominator = args.batch_size if args.opt_divide_batch else 1
    def compute_loss(X, y):
      pred = model(X, training=True)
      neg_ll = likelihood_fn(y, pred)
      # print(f'neg_ll = {tf.reduce_sum(neg_ll)}')
      # now get the losses from the prior (negative log prior)
      # these are stored within the models `losses` variable
      neg_lp = tf.reduce_sum(model.losses)
      # now scale the neg ll and the neg log prior based on the distribution strategy
      per_example_loss = tf.nn.compute_average_loss(
          neg_ll, global_batch_size=denominator
      ) + tf.nn.scale_regularization_loss(neg_lp)
      return per_example_loss

  return compute_loss

def create_sgld_loss_fn(model, likelihood_fn, args, data_dimension_dict, strategy):
  with strategy.scope():
    dataset_size = tf.convert_to_tensor(data_dimension_dict['dataset_size'],
                                        dtype=tf.float32)
    batch_size = tf.convert_to_tensor(args.batch_size, dtype=tf.int32)
    def compute_loss(X, y):
      pred = model(X, training=True)
      neg_ll = likelihood_fn(y, pred)
      # now scale the negative log likelihood by the number of data points
      # don't divide by the batch size here, that will be done in the
      # compute_average_loss fn below, which is strategy aware
      neg_ll_scaled = dataset_size * neg_ll
      # now get the losses from the prior (negative log prior)
      # these are stored within the models `losses` variable
      neg_lp = tf.reduce_sum(model.losses)
      # now scale the neg ll and the neg log prior based on the distribution strategy
      per_example_loss = tf.nn.compute_average_loss(
          neg_ll, global_batch_size=batch_size
      ) + tf.nn.scale_regularization_loss(neg_lp)
      return per_example_loss
  return compute_loss



def get_distribution_strategy(gpus, hmc=False):
  if len(gpus) == 1:
    print(gpus)
    if hmc:
      return NoDistribute()
    else:
      return tf.distribute.OneDeviceStrategy(gpus[0])
  else:
    return tf.distribute.MirroredStrategy(gpus)


def get_optimizer(optimizer_str,
                  initial_lr,
                  strategy,
                  momentum=0.0,
                  schedule=None,
                  t_mul=None,
                  m_mul=None,
                  piecewise_decay_steps=None,
                  decay_rate=0.1,
                  num_iters=0):
  """Get right optimi_zer for the job.

  Different models work better with different optimizers. Adam has reported to
  work pretty we'll and converge reasonably quickly for many tasks.

  In general we will probably want to stick with Adam, but for some of the larger
  models like ResNet50 trained on imagenet, we might want to use a different
  optimizer. For these models, people traditionally just use some SGD with momentum
  and a decaying learning rate.

  This function will just make it easier for us to specify an optimizer,
  and maybe a learning rate schedule to use.

  Parameters
  ----------
  optimizer_str : str
      optimizer to use.
  initial_lr : float
      initial learning rate
      Used by: All
  strategy : tf.distribute.Stategy
      Distribution stragegy for training.
  momentum: float (default 0.0)
      momentum to be used.
      Used by: All schedules with SGD
  schedule: str (default None)
      Learning rate schedule to use.
  t_mul: int (default None)
      Factor to run cosine restart decay for after first iter
      Used by: CosineDecayRestarts
  m_mul: int (default None)
      Factor to multiply initial learning rate by after first iteration
      of the cosine warm restart (generally good to just set to one)
      Used by: CosineDecayRestarts
  piecewise_decay_steps: list(int) (default None)
      Steps to apply the learning rate schedule.
      Used by:  PiecewiseDecay, CosineDecayRestarts, PolynomialDecay for SGLD
  decay_rate: float (default 0.1)
      If piecewise  decay is specified, this is the rate at which lr will be
      attenuated
  num_iters: int (default 0)
      Number of training iters. Is used if cosine schedule applied.
      Used by: CosineDecay

  Returns
  -------
      Keras Optimizer.

  Raises
  ------
     ValueError
         if invalid optimizer supplied or scheduler.
     ValueError
         if decay steps is supplied (not None), but the
         scheduler was not supplied (is None). Doesn't make sense to
         have decay steps with no scheduler, so will assume it is an error.
  """
  with strategy.scope():
    # check if a scheduler was supplied
    if schedule is None:
      lr = initial_lr
    elif schedule.lower() == 'piecewise':
      # create the steps needed
      lr_values = [
          initial_lr * decay_rate**(i) for i in range(len(piecewise_decay_steps) + 1)
      ]
      print('here')
      print(piecewise_decay_steps)
      print(lr_values)
      lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
          piecewise_decay_steps, lr_values)
    elif schedule.lower() == 'cosine':
      lr = tf.keras.optimizers.schedules.CosineDecay(initial_lr, num_iters)
      print('cosine lr')
    elif schedule.lower() == 'cosine_restart':
      # first check if m_mult was supplied. If it wasn't and is set to
      # None still, lets just set it to one.
      if m_mul == None:
        m_mul = 1.0
        print(f'setting m_mult to {m_mul}')
      lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
          initial_lr, piecewise_decay_steps[0], t_mul=t_mul, m_mul=m_mul)
    elif schedule.lower() == 'linear':
      lr = tf.keras.optimizers.schedules.PolynomialDecay(
          initial_lr, piecewise_decay_steps, end_learning_rate=0.0, power=1)
    elif schedule.lower() == 'warmup':
      lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
          [1000, 2000, 5000], [initial_lr * 0.001, initial_lr * 0.01, initial_lr * 0.1, initial_lr])

      # lr = tfm.optimization.LinearWarmup(
      #   after_lr,
      #   1000,
      #   0.0000001
      # )
    elif schedule.lower() == 'mine':
      lr = MyLR(lr)
      print('getting my warmup lr')
      time.sleep(10)
    else:
      raise ValueError((f'Currently only support Piecewise constant decay'
                        'or no learning rate schedule.'))
    # now create the optimizer
    if optimizer_str == "adam":
      opt = tf.keras.optimizers.Adam(lr)
      # opt = tfa.optimizers.AdamW(0.02, learning_rate=lr)
    elif optimizer_str == "sgd":
      opt = tf.optimizers.SGD(learning_rate=lr,
                              momentum=momentum,
                              nesterov=False)  #,  global_clipnorm=1.0)
    elif optimizer_str == "rmsprop":
      opt = tf.keras.optimizers.RMSProp(lr, momentum=momentum)
    else:
      raise ValueError((f'Invalid optimizer string. got {args.opt},'
                        'expect one of {VALID_OPTIMIZER_STRS}'))
    print('opt', opt)
  return opt


def load_map_weights(model, args):
  """Get map weights ready for sampling.

  For most network's, this will just load in the weights normally. For nfnets
  where we did fine tuning on the final layer, we will load in the full keras
  weights to load in, then we will call the get weights function which will get
  the weights we are training from.

  Parameters
  ----------
  model: keras.Model
      keras model, needed for nfnets type model
  args : cmdline ars
      Command line args with map weights path and the network type

  Returns
  -------
      list of tensors holding the MAP weights.

  """
  if args.network == 'nfnet':
    # load in the weights saved for the entire network.
    model.load_weights(args.map_path)
    # need to set the right weights to trainable
    for layer in model.layers:
      if isinstance(layer, (tf.keras.layers.Dense, ScaledStandardizedConv2D)):
        # need to add loss to the layer kernel and bias
        layer.trainable = True
        # if is nf base layers
      elif "Functional" == layer.__class__.__name__:
        for _layer in layer.layers:
          if isinstance(_layer,
                        (tf.keras.layers.Dense, ScaledStandardizedConv2D)):
            _layer.trainable = True
          else:
            _layer.trainable = False
            # if is any other kind of layer set it to be false
      else:
        layer.trainable = False
        # now get the map weights for trainable variables we will be
        # sampling from
    print('checking model shapes')
    map_initial_state = get_model_state(model)
  elif args.network == 'resnet50_keras':
    print(model.summary())
    return model.trainable_variables
  else:
    with open(args.map_path, 'rb') as f:
      map_initial_state = pickle.load(f)
  return map_initial_state


def normal_likelihood():

  def _fn(y, pred):
    y = tf.cast(y, tf.float32)
    dist = tfd.Normal(loc=pred, scale=1.0)
    return -1.0 * dist.log_prob(y)

  return _fn


def main(args):
  """main function to fit the models"""
  # get the distribution strategy.
  strategy = get_distribution_strategy(args.gpus, args.hmc)
  # getting the data
  (training_iter, training_ds, test_ds, orig_test_ds, data_dimension_dict,
   label_dict) = get_data(args.data, args.batch_size, strategy)
  # if is regression model
  if args.data in ['toy_a', 'toy_b', 'toy_c', 'moons', 'boston']:
    input_dims = [data_dimension_dict['in_dim']]
  else:
    input_dims = [
        data_dimension_dict['in_height'], data_dimension_dict['in_width'],
        data_dimension_dict['in_channels']
    ]
  # get single sample to build the model
  print(
      f"memory usage {tf.config.experimental.get_memory_info('GPU:0')['current'] / 10 ** 9} GB"
  )
  model = get_network(args.network,
                      strategy,
                      input_dims,
                      data_dimension_dict['out_dim'],
                      prior=args.prior,
                      vi=args.vi,
                      map_training=True)  #args.map_path == None)

  if args.likelihood == 'categorical':
    with strategy.scope():
      # likelihood_fn = networks.get_likelihood_fn(args.likelihood)
      likelihood_fn = tf.keras.losses.CategoricalCrossentropy(
          from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
      likelihood_fn_map = tf.keras.losses.CategoricalCrossentropy(
          from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
  elif args.likelihood == 'normal':
    # likelihood_fn = tf.keras.losses.MeanSquaredError(
    #   reduction=tf.keras.losses.Reduction.NONE)
    # likelihood_fn_map = tf.keras.losses.MeanSquaredError(
    #   reduction=tf.keras.losses.Reduction.NONE)
    likelihood_fn = normal_likelihood()
    likelihood_fn_map = normal_likelihood()
  else:
    # is bernoulli
    likelihood_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    likelihood_fn_map = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

  loss_fn = create_loss_fn(model, likelihood_fn_map, args, strategy)

  print('likelihood_fn = {}'.format(likelihood_fn))
  print(model)
  print(model.summary())
  print('model type = {}'.format(model))
  bnn_neg_joint_log_prob = iter_grad_fn(model, likelihood_fn, training_iter,
                                        strategy)
  hessian_fn = iter_hessian_diag_fn(model, likelihood_fn, training_iter,
                                    strategy)

  # get the initial state for obtaining MAP estimate.
  # This can just be the getting initial values from the model we defined
  initial_state = get_model_state(model)
  #print('initial_state = {}'.format(initial_state))
  print('Dataset = {}'.format(training_iter))
  if args.map_path == None:
    optimizer = get_optimizer(args.opt,
                              args.lr,
                              strategy,
                              schedule=args.scheduler,
                              momentum=args.momentum,
                              t_mul=args.t_mul,
                              m_mul=args.m_mul,
                              piecewise_decay_steps=args.decay_steps,
                              decay_rate=args.decay_rate,
                              num_iters=args.map_iters)
    dist_opt_step = iter_opt_fn(model, loss_fn, training_iter, strategy,
                                optimizer)
    map_start = time.time()
    # model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #               optimizer=optimizer)
    # model.fit(training_iter, steps_per_epoch=100, epochs=5, validation_data=test_ds)
    map_initial_state = get_map_test(dist_opt_step,
                                     loss_fn,
                                     training_iter,
                                     model,
                                     likelihood_fn_map,
                                     optimizer,
                                     strategy,
                                     num_iters=args.map_iters,
                                     test_dataset=test_ds)
    # with strategy.scope():
    #   model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #                 optimizer="adam",
    #                 metrics=["accuracy"])
    #   model.fit(training_iter, epochs=100, steps_per_epoch=500,
    #             validation_data=test_ds)

    map_end = time.time()
    print('time to find MAP estimate = {}'.format(map_end - map_start))
    # get accuracy for classification model
    if args.likelihood == 'categorical':
      accuracy = classification_accuracy(model, test_ds)
      print('Test accuracy from MAP = {}'.format(accuracy))
    # plot response for a regression model
    elif args.likelihood == 'normal':
      if args.data in ['boston']:
        # no plots for the UCI data
        pass
      else:
        X_train, y_train = next(training_iter)
        model = set_model_params(model, map_initial_state)
        pred = pred_forward_pass(model, map_initial_state,
                                 X_train.numpy().astype(np.float32))
        plt.plot(X_train, pred, color='k')
        print(X_train.shape)
        print(y_train.shape)
        plt.scatter(X_train, y_train, color='b', alpha=0.25)
        plt.savefig(os.path.join(args.out_dir, 'pred_map.png'))
    # plot map response for logistic regression model
    else:
      X_train, y_train = next(training_iter)
      pred = pred_forward_pass(model, map_initial_state,
                               X_train.numpy().astype(np.float32))
      pred = np.round(tf.keras.activations.sigmoid(pred)).astype(np.int64)
      plt.scatter(X_train.numpy()[pred.ravel() == 0, 0],
                  X_train.numpy()[pred.ravel() == 0, 1],
                  color='b')
      plt.scatter(X_train.numpy()[pred.ravel() == 1, 0],
                  X_train.numpy()[pred.ravel() == 1, 1],
                  color='r')
      #plt.scatter(X_test, y_test, color='b', alpha=0.5)
      plt.savefig(os.path.join(args.out_dir, 'pred_logistic_map.png'))

    # save the MAP weights
    # if is nfnet, save all the weights from the fine tuning
    # else, save weights in format suitable for sampling
    if args.network == 'nfnet':
      model.save_weights(os.path.join(args.out_dir, 'map_weights.h5'))
    else:
      save_map_weights(map_initial_state, args.out_dir)
  else:
    print('loading map')
    map_initial_state = load_map_weights(model, args)
    print('done loading map')
  # examine_rate(model, bnn_neg_joint_log_prob, training_iter, likelihood_fn, strategy, map_initial_state, args.out_dir)
  # now train MCMC method if specified
  # number of samples available for training
  if args.bps:
    bps_iter_main(model,
                  args.ipp_sampler,
                  args.ref,
                  args.std_ref,
                  args.num_results,
                  args.num_burnin,
                  args.out_dir,
                  args.num_loops,
                  bnn_neg_joint_log_prob,
                  map_initial_state,
                  training_iter,
                  test_ds,
                  orig_test_ds,
                  args.likelihood,
                  args.batch_size,
                  args.batch_size,
                  data_dimension_dict,
                  # plot_results=~args.no_plot,
                  plot_results=False,
                  run_eval=False,
                  num_steps_between_results=args.steps_between)
  if args.boomerang:
    boomerang_test_iter_main(model,
                             args.ipp_sampler,
                             args.ref,
                             args.std_ref,
                             args.num_results,
                             args.num_burnin,
                             args.out_dir,
                             args.num_loops,
                             hessian_fn,
                             bnn_neg_joint_log_prob,
                             strategy,
                             likelihood_fn,
                             map_initial_state,
                             training_iter,
                             test_ds,
                             orig_test_ds,
                             args.likelihood,
                             args.batch_size,
                             args.batch_size,
                             data_dimension_dict,
                             # plot_results=~args.no_plot,
                             plot_results=False,
                             run_eval=False,
                             num_steps_between_results=args.steps_between)
  if args.cov_pbps:
    cov_pbps_test_iter_main(model,
                            args.ipp_sampler,
                            args.ref,
                            args.std_ref,
                            args.num_results,
                            args.num_burnin,
                            args.out_dir,
                            args.num_loops,
                            bnn_neg_joint_log_prob,
                            map_initial_state,
                            training_iter,
                            test_ds,
                            orig_test_ds,
                            args.likelihood,
                            args.batch_size,
                            args.batch_size,
                            data_dimension_dict,
                            plot_results=False,
                            run_eval=False,
                            # plot_results=~args.no_plot,
                            num_steps_between_results=args.steps_between)

  if args.pbps:
    pbps_iter_main(model,
                   args.ipp_sampler,
                   args.ref,
                   args.std_ref,
                   args.num_results,
                   args.num_burnin,
                   args.out_dir,
                   args.num_loops,
                   bnn_neg_joint_log_prob,
                   map_initial_state,
                   training_iter,
                   test_ds,
                   orig_test_ds,
                   args.likelihood,
                   args.batch_size,
                   args.batch_size,
                   data_dimension_dict,
                   # plot_results=~args.no_plot,
                   plot_results=False,
                   run_eval=False,
                   num_steps_between_results=args.steps_between)
  if args.vi:
    # create a likelihood that will use sum reduction
    with strategy.scope():
      likelihood_vi_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

    optimizer = get_optimizer(args.opt,
                              args.lr,
                              strategy,
                              schedule=args.scheduler,
                              momentum=args.momentum,
                              t_mul=args.t_mul,
                              m_mul=args.m_mul,
                              piecewise_decay_steps=args.decay_steps,
                              num_iters=args.map_iters)
    variational_iter_main(model,
                          args.num_results,
                          args.out_dir,
                        likelihood_vi_fn,
                          optimizer,
                        map_initial_state,
                        training_ds,
                        test_ds,
                        orig_test_ds,
                        args.likelihood,
                        args.batch_size,
                        args.batch_size,
                        data_dimension_dict,
                        plot_results=~args.no_plot,
                        run_eval=True)



  if args.sgld:
    optimizer = get_optimizer('sgd',
                              args.lr,
                              strategy,
                              schedule='linear',
                              # schedule= None,
                              momentum=0.0,
                              t_mul=0.0,
                              m_mul=0.0,
                              piecewise_decay_steps=args.num_results,
                              num_iters=None)
    # create the loss fn needed for sgld
    loss_fn = create_sgld_loss_fn(model, likelihood_fn, args, data_dimension_dict, strategy)
    # get the distributed step for sgld
    sgld_opt_fn = iter_sgld_fn(loss_fn,
                               model,
                               training_iter,
                               strategy,
                               optimizer)

    sgld_iter_main(model,
                   args.num_results,
                   args.out_dir,
                   args.num_loops,
                   sgld_opt_fn,
                   map_initial_state,
                   training_iter,
                   test_ds,
                   orig_test_ds,
                   args.likelihood,
                   data_dimension_dict,
                   # plot_results=~args.no_plot,
                   # run_eval=True,
                   plot_results=False,
                   run_eval=False,
                   num_steps_between_results=args.steps_between)
  if args.hmc:
    # get the normal log prob (not the negative)
    bnn_joint_log_prob = iter_hmc_fn(model,
                                     likelihood_fn,
                                     training_iter,
                                     strategy)

    hmc_iter_main(model,
                  args.num_results,
                  args.num_burnin,
                  args.out_dir,
                  args.step_size,
                  args.leapfrog_steps,
                  bnn_joint_log_prob,
                  map_initial_state,
                  training_ds,
                  test_ds,
                  orig_test_ds,
                  args.likelihood,
                  data_dimension_dict,
                  plot_results=False,#~args.no_plot,
                  run_eval=False,
                  num_steps_between_results=args.steps_between)

def check_cmd_args(args):
  """check all the commandline arguments are valid"""
  # check network argument
  networks.check_network_arg(args.network)
  # now the likelihood
  networks.check_likelihood_arg(args.likelihood)
  # check prior is ok, and format if needed
  args.prior = networks.check_format_prior_arg(args.prior)
  # lets check that the directories suplied exist, and if they don't,
  # lets make them
  utils.check_or_mkdir(args.out_dir)
  # let's also make sure that the gpu specification is valid
  args.gpus = check_format_gpus(args.gpus)
  # let's check some of the arguments for the optimizer as we'll
  (args.opt, args.map_iters, args.lr, args.decay_steps,
   args.decay_rate) = check_format_optimizer_args(args)
  return args


def check_format_optimizer_args(args):
  """Validate args for optimizer if specified.

  Want to make sure the arguments specified are correct and valid.

  Parameters
  ----------
  args : Argparse object
      cmdline arguments for this program.

  Returns
  -------
  opt, map_iters, lr, decay_steps, decay_rate arguments
  and formats them accordingly.

  Raises
  ------
      ValueError if any arguments are misspecified.
  """
  # check the optimizer string
  args.opt = args.opt.lower()
  if args.opt.lower() not in VALID_OPTIMIZER_STRS:
    raise ValueError((f'Invalid optimizer string. got {args.opt},'
                      'expect one of {VALID_OPTIMIZER_STRS}'))
  if (args.map_iters < 0):
    raise ValueError((f'Invalid map_iter arg. expect positive int'
                      ', got {args.map_iters}'))
  if args.lr < 0:
    raise ValueError((f'Invalid learning rate arg. expect positive float'
                      ', got {args.lr}'))
  # need to first check if it is not None
  if args.decay_rate != None:
    if args.decay_rate < 0:
      raise ValueError((f'Invalid decay rate arg. expect positive float'
                        ', got {args.decay_rate}'))
  # now will try and format the decay steps to see if is valid format
  # it should be as a list of integers separated by commas
  try:
    # if it is just set to None than don't need to worry.
    if args.decay_steps == None:
      decay_step_list = None
    else:
      decay_step_list = args.decay_steps.split(',')
      decay_step_list = [int(i) for i in decay_step_list]
  except Exception as e:
    raise ValueError((f'Invalid format for decay steps. Expected list '
                      'of integers seperated by commas with no spacing '
                      'for example: 10,20,30 '
                      'but got {args.decay_steps}'))

  return (args.opt.lower(), args.map_iters, args.lr, decay_step_list,
          args.decay_rate)


def check_format_gpus(gpus):
  """Validate cmdline arg for gpus.

    GPU arg is interpreted and saved as as string, but it can also be an integer
    or a list of integers, eg. 0,1,2,3 etc. This fn will check the arguments
    supplied from commandline to ensure that they are valid, and will throw an
    exception upfront if any funny bussiness is encountered.

    Parameters
    ----------
    gpus : str
        gpu spec for program. Can be one of, "all", to use all gpus, a string
        with a single integer "0", or string with a list of valid gpu ids
        "0,1,2,3".

    Returns
    -------
    gpu arg formatted

    Raises
    ------
    value error if incorrect format.
    """
  # check if is just "all" to use all gpus
  all_gpus = tf.config.list_physical_devices('GPU')
  if gpus == 'all':
    gpus_to_use = [f'/gpu:{x}' for x in range(0, len(all_gpus))]
  # check if is a single integer
  elif gpus.isdigit():
    gpus_to_use = [f'/gpu:{gpus}']
  # only other valid format is integers separated by a comma
  # with no spaces.
  elif gpus.replace(',', '').isdigit():
    gpus_to_use = [f'/gpu:{x}' for x in gpus.split(',')]
  # if have made it to here, no valid format found
  else:
    raise ValueError("Invalid gpus spec provided.")
  print(f'gpus to be used = {gpus_to_use}')
  return gpus_to_use


if __name__ == '__main__':

  parser = argparse.ArgumentParser(
      prog='test_conv_new',
      epilog=main.__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('network', type=str, help='network type')
  parser.add_argument('likelihood',
                      type=str,
                      help='type of likelihood function to use')
  parser.add_argument('prior', type=str, help='prior to be used')
  parser.add_argument('--out_dir',
                      type=str,
                      default='./out',
                      help='out directory where data is saved')
  parser.add_argument('--gpus',
                      type=str,
                      default='all',
                      help='gpus to use (default to all)')
  parser.add_argument('--opt',
                      type=str,
                      default='adam',
                      help='optimizer to be used')
  parser.add_argument('--lr',
                      type=float,
                      default=0.001,
                      help='initial learning rate for MAP')
  parser.add_argument('--momentum',
                      type=float,
                      default=0.0,
                      help='Momentum parameter to use for MAP optimizer')
  parser.add_argument('--scheduler',
                      type=str,
                      default=None,
                      help='Learning rate scheduler for MAP optimizer')
  parser.add_argument('--decay_rate',
                      type=float,
                      default=None,
                      help='Decay rate for piecewise decay for map optimizer')
  parser.add_argument('--decay_steps',
                      type=str,
                      default=None,
                      help=('comma seperated step values to apply piecewise'
                            'scheduler decay rate.'))
  parser.add_argument('--batch_size',
                      type=int,
                      default=100,
                      help='Number of samples per batch')
  parser.add_argument('--data',
                      type=str,
                      default='mnist_im',
                      help='data set to use')
  parser.add_argument('--t_mul',
                      type=float,
                      default=2,
                      help='Exponential range scaling for CosineRestart schedule')
  parser.add_argument('--m_mul',
                      type=float,
                      default=1.0,
                      help='Exponential range scaling for CosineRestart schedule')
  parser.add_argument('--ref',
                      type=float,
                      default=1.0,
                      help='lambda for refresh poisson process')
  parser.add_argument('--std_ref',
                      type=float,
                      default=0.001,
                      help='Variance for refresh distribution')
  parser.add_argument('--map_iters',
                      type=int,
                      default=20000,
                      help='number of iterations for map estimate')
  parser.add_argument('--opt_divide_batch',
                      type=bool,
                      default=False,
                      nargs='?',
                      const=True,
                      help='whether to divide optimizer loss by batch size')
  parser.add_argument('--map_path',
                      type=str,
                      default=None,
                      help='path to load map weights')
  parser.add_argument('--bps',
                      type=bool,
                      default=False,
                      nargs='?',
                      const=True,
                      help='whether to run BPS')
  parser.add_argument('--boomerang',
                      type=bool,
                      default=False,
                      nargs='?',
                      const=True,
                      help='whether to run Boomerang Sampler')
  parser.add_argument('--cov_pbps',
                      type=bool,
                      default=False,
                      nargs='?',
                      const=True,
                      help='whether to run Cov precond. BPS')
  parser.add_argument('--pbps',
                      type=bool,
                      default=False,
                      nargs='?',
                      const=True,
                      help='sbps')
  parser.add_argument('--sgld',
                      type=bool,
                      default=False,
                      nargs='?',
                      const=True,
                      help='whether to run sgld')
  parser.add_argument('--vi',
                      type=bool,
                      default=False,
                      nargs='?',
                      const=True,
                      help='whether to run variational inference')
  parser.add_argument('--ipp_sampler',
                      type=str,
                      default='adaptive',
                      nargs='?',
                      help='type of sampling scheme for event IPP')
  parser.add_argument('--hmc',
                      type=bool,
                      default=False,
                      nargs='?',
                      const=True,
                      help='whether to run HMC')
  parser.add_argument('--step_size',
                      type=float,
                      default=0.0001,
                      nargs='?',
                      help='HMC step size')
  parser.add_argument('--leapfrog_steps',
                      type=int,
                      default=10,
                      nargs='?',
                      help='Number HMC leapfrog steps')
  parser.add_argument('--num_results',
                      type=int,
                      default=100,
                      help='number of sample results')
  parser.add_argument('--num_loops',
                      type=int,
                      default=1,
                      help='number of loops needed to get all results')
  parser.add_argument('--steps_between',
                      type=int,
                      default=1,
                      help='number samples between results')
  parser.add_argument('--num_burnin',
                      type=int,
                      default=100,
                      help='number of burnin samples')
  description_help_str = ('experiment description'
                          '(place within single quotes \'\'')
  parser.add_argument('--description',
                      type=str,
                      default='test-logistic',
                      nargs='?',
                      help=description_help_str)
  parser.add_argument('--exp_name',
                      type=str,
                      default='test-conv',
                      nargs='?',
                      help='name of experiment (usually don\'t have to change)')
  parser.add_argument('--no_log',
                      type=bool,
                      default=False,
                      nargs='?',
                      const=True,
                      help='whether should skip logging to neptune or not')
  parser.add_argument(
      '--no_plot',
      type=bool,
      default=False,
      nargs='?',
      const=True,
      help='whether should skip plotting or getting pred metrics')
  args = parser.parse_args(sys.argv[1:])
  # now lets check all the arguments
  args = check_cmd_args(args)
  # if args.bps == False and args.hmc == False and args.nuts == False:
  #   raise ValueError('Either arg for BPS, HMC or NUTS must be supplied')
  # if we are logging info to neptune, initialise the logger now
  exp_params = vars(args)
  print('args = {}'.format(args))
  exp_tags = [
      key for key, x in exp_params.items()
      if (isinstance(x, bool) and (x == True))
  ]
  if (not args.no_log):
    print('logging to neptune')
    neptune.init('ethangoan/{}'.format(args.exp_name))
    neptune.create_experiment(
        name='test_conv',
        params=exp_params,
        tags=exp_tags,
        upload_source_files=[args.config, './test_conv.py'])

  main(args)
