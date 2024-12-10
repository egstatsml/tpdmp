"""model.py.

handles implementation of running different samplers for
the different models.

One main component is how the memory of GPU devices available
is handled. For larger models (or even moderately sized models),
storing all samples on the GPU is not feasible (and not necessary).
It might not even be feasible to store them within CPU RAM, so we
save them to file.

Example: say we need to run 1000 samples for our model, but we
can only fit 100 samples in the GPU. To get the full number of samples,
we run the chain to fill up our GPU memopry, quickly exit it, and then
run the chain again (will do this 10 times).

```python

for i in range(0, 10):
  samples = run_chain(num_samples=100)
  # save the current samples to file
  save_samples(samples, i)
```


##### Examples


Getting Concrete function:

Just use the `get_concrete_function()` method attached to any tf function.
```python
concrete = graph_hmc.get_concrete_function(
  num_results=num_results,
  current_state=map_initial_state,
  kernel=kernel,
  trace_fn=None)
```

Saving model graph:

```python
# create a writer
writer = tf.summary.create_file_writer('./conv_debug')
tf.summary.trace_on(graph=True, profiler=True)
####### CALL tf.function HERE #######
with writer.as_default():
  tf.summary.trace_export(
    name='my_func_trace',
    step=0,
    profiler_outdir=out_dir)
```



"""
import time
import time as timepy
from abc import ABCMeta
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.layers import (Dense, Conv2D, MaxPool2D, Flatten,
                                     GlobalAveragePooling2D, BatchNormalization,
                                     Layer, Add)
# from tensorflow_probability.layers import (DenseReparameterization,
#                                            Convolutional2DReparameterization)

from tbnn.pdmp.nfnets import ScaledStandardizedConv2D
from tbnn.pdmp.pdmp import (BPSKernel, IterBPSKernel, CovPBPSKernel,
                           IterCovPBPSKernel, PBPSKernel, IterPBPSKernel,
                           BoomerangKernel, BoomerangIterKernel)
from tbnn.pdmp.poisson_process import (InterpolationSampler, InterpolationBoomSampler,
                                       InterpolationPSampler)
from tbnn.pdmp.hmc import IterHMCKernel

import tensorflow as tf
from tbnn.nn.mlp import MLP
from tbnn.nn.conv import Conv
from tbnn.utils import utils, display, reliability_diagrams

import os
import sys
from glob import glob
import json
from sklearn.metrics import accuracy_score
from tbnn.pdmp.resnet import ResNetLayer
# from tbnn.pdmp.resnet50 import ResNet50Block, ResNetShortcut
from tbnn.pdmp.frn import FRN
from tbnn.pdmp.train_utils import *

tfd = tfp.distributions


class InvalidMCMCLikelihood(ValueError):
  """Exception for invalid likelihood to be raised."""

  def __init__(self, arg):
    print('Invalid likelihood function specified in config file')
    print('Supplied {}, available loss functions are {}'.format(
        arg, ['normal', 'bernoulli', 'categorical']))
    ValueError.__init__(self)


class MCMCModelUtils(metaclass=ABCMeta):
  """Mixin class to be used for MCMC models.

  This class should only implement a few basic functionalities,
  such as setting joint distributions, setting likelihoods,
  setting parameters and running predictive forward passes.

  This class won't implement a lot of the core functionality,
  as it is inteded as a mixin class to be shared across varying
  model types, such as Dense or Conv nets. These model defs
  should implement a lot of the main low lovel functionality.

  This is an abstract class as it should never be used in isolation,
  only as a mixin class.
  """

  def __init__(self, *args, **kwargs):
    # not initialising anything yet
    pass


class MCMCMLP(MLP):

  def __init__(self,
               prior,
               posterior,
               json_path=None,
               dimension_dict=None,
               name='MCMC_MLP'):
    # explicitly adding likelihoof fn needed for TFP
    super().__init__(prior, posterior, json_path, dimension_dict, name)

  def _get_likelihood(self, arg):
    '''get likelihood for the current model

    For the MCMC models it should be the name of a TFP distribution.
    This method checks that the likelihood supplied is valid, and then
    sets the self.likelihood_fn attribute to the corresponding dist.

    If the supplied likelihood is not valid, than raise an error

    Args:
      args (str):
        name of likelihood fn supplied in the config file

    Returns:
      the input argument if it is of a valid form

    Raises:
      InvalidMCMCLikelihood() if incorrect likelihood supplied
    '''
    arg = arg.lower()
    if (arg == 'normal'):
      self.likelihood_fn = tfd.Normal
    elif (arg == 'bernoulli'):
      self.likelihood_fn = tfd.Bernoulli
    elif (arg == 'categorical'):
      self.likelihood_fn = tfd.OneHotCategorical
    else:
      raise InvalidMCMCLikelihood(arg)
    return arg


class MCMCConv(Conv):

  def __init__(self,
               prior,
               posterior,
               json_path=None,
               dimension_dict=None,
               name='MCMC_MLP'):
    # explicitly adding likelihoof fn needed for TFP
    super().__init__(prior, posterior, json_path, dimension_dict, name)

  def _get_likelihood(self, arg):
    '''get likelihood for the current model

    For the MCMC models it should be the name of a TFP distribution.
    This method checks that the likelihood supplied is valid, and then
    sets the self.likelihood_fn attribute to the corresponding dist.

    If the supplied likelihood is not valid, than raise an error

    Args:
      args (str):
        name of likelihood fn supplied in the config file

    Returns:
      the input argument if it is of a valid form

    Raises:
      InvalidMCMCLikelihood() if incorrect likelihood supplied
    '''
    arg = arg.lower()
    if (arg == 'normal'):
      self.likelihood_fn = tfd.Normal
    elif (arg == 'bernoulli'):
      self.likelihood_fn = tfd.Bernoulli
    elif (arg == 'categorical'):
      self.likelihood_fn = tfd.OneHotCategorical
    else:
      raise InvalidMCMCLikelihood(arg)
    return arg


def plot_density(trace, num_chains, num_samples, figsize):
  """Plots traces for individual parameters."""
  print(trace.shape)
  # if our variables we are plotting the trace for can be represented as a matrix
  if (len(trace.shape) == 3):
    n_rows = trace.shape[1]
    n_cols = trace.shape[2]
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    axes = axes.reshape(n_rows, n_cols)
    print(axes.shape)
    print(type(axes))
    for row in range(0, n_rows):
      for col in range(0, n_cols):
        # reshape the trace data to seperate for individual chains
        for chain in range(0, num_chains):
          sns.kdeplot(trace[chain:num_samples * (chain + 1), row, col],
                      ax=axes[row, col])
        #axes[row, col].set_xlabel('w_{}_{}'.format(row, col), fontsize=6)
        axes[row, col].set_axis_off()
  # otherwise the trace variables can be represented as a vector
  else:
    #
    n_rows = trace.shape[1]
    fig, axes = plt.subplots(nrows=n_rows, figsize=figsize)
    for row in range(0, n_rows):
      # reshape the trace data to seperate for individual chains
      for chain in range(0, num_chains):
        sns.kdeplot(trace[chain:num_samples * (chain + 1), row], ax=axes[row])
      axes[row].set_xlabel('w_{}'.format(row), fontsize=6)
      axes[row].set_axis_off()
  return fig, axes


def get_model_state(model):
  """return list of model parameters.

  Is just a small helper function that finds the trainable parameters that we
  are sampling within the model.

  Parameters
  ----------
  model : keras.model
      network we are sampling from.

  Returns
  --------
  list(tf.Ternsor)
      list of model trainable parameters
  """
  return model.trainable_variables


def set_model_params(model, param_list):
  """Apply new params to a model.

  When sampling, will need to apply previous params of a model, or sometimes
  maybe to apply the initial state of model params. This function will take in a
  list of parameters, parameters of the same format as that returned from
  `get_model_state`, and update the mode to use this new params.

  Parameters
  ----------
  model : keras.Model
      model/network we are sampling from.
  param_list : list(tf.Tensor)
      list of trainable parameters we need to update, basically the one's we are
      sampling from or optimizing.

  Returns
  -------
  keras.Model
    model with params from `param_list` applied.
  """
  # print('look at param list')
  # print(type(param_list))
  # for p in param_list:
  #   print(type(p))
  param_idx = 0
  for layer_idx in range(0, len(model.layers)):
    # if we are using nfnets model, we will need to access the base
    # sub network which was created with the functional method.
    if "Functional" == model.layers[layer_idx].__class__.__name__:
      param_idx = set_nfnet_base_params(model.layers[layer_idx], param_list,
                                        param_idx)
    # set model params, but make sure this current layer isn't a flatten or
    # pooling layer with no params to set
    elif (isinstance(model.layers[layer_idx], Dense) or
          isinstance(model.layers[layer_idx], Conv2D) or
          isinstance(model.layers[layer_idx], ScaledStandardizedConv2D)):
      model, param_idx = set_dense_or_conv_params(model, layer_idx, param_list,
                                                  param_idx)
    elif isinstance(model.layers[layer_idx], ResNetLayer):
      model, param_idx = set_resnet_layer_params(model, layer_idx, param_list,
                                                 param_idx)
    elif isinstance(model.layers[layer_idx], FRN):
      model, param_idx = set_frn_layer_params(model, layer_idx, param_list,
                                              param_idx)
    else:
      pass
      # print('LAYER TYPE = {}, class = {}'.format(
      #     type(model.layers[layer_idx]), model.layers[layer_idx].__class__))
    #print('bias after= {}'.format(model.layers[i].bias))
  return model


def set_nfnet_base_params(nf_base, param_list, param_idx):
  """ set base params for nfnet.

  For our nfnet, we used the base pretrained network and then fine tuned def __iter__(self):
  to run at a smaller resolution input, which meant changing the final layer. When we
  did this we created a Functional model, where our model was defined such as:
     nfnet_base:
     Global pool:
     Dense

  This function will set the params in the nf_base part of the network, which keras
  has defined as a single layer of our full network.

  Parameters
  ----------
  nf_base : keras.Model (functional)
      base network of the nfnet we fine tuned.

  param_list : list(tf.Tensor)
      list of trainable parameters we need to update, basically the one's we are
      sampling from or optimizing.

  Returns
  -------
      updated param idx
  """
  for layer_idx in range(0, len(nf_base.layers)):
    # set model params, but make sure this current layer isn't a flatten or
    # pooling layer with no params to set
    if (isinstance(nf_base.layers[layer_idx], Dense) or
        isinstance(nf_base.layers[layer_idx], ScaledStandardizedConv2D)):
      _, param_idx = set_dense_or_conv_params(nf_base, layer_idx, param_list,
                                              param_idx)
    else:
      print('LAYER TYPE = {}, class = {}, name= {}'.format(
          type(nf_base.layers[layer_idx]), nf_base.layers[layer_idx].__class__,
          nf_base.layers[layer_idx].name))
  return param_idx


def set_dense_or_conv_params(model, layer_idx, param_list, param_idx):
  """Set params for dense or conv net.

  Only one function for both conv and dense layers, as they have the same
  number of parameters and parameter names (kernel and bias).

  Params:
    model (keras.Model):
      keras model of network
    params (list(tf.Varaibles)):
      list of variables for the model
    layer_idx (int):
      index for the current layer in the model
    param_idx (int):
      index to tell us the starting point in the list of model parameters to
    look at.

  Returns
  -------
    updated Model with params set, incremented param_idx
  """
  # print(f'layer index =  {layer_idx}, param_idx = {param_idx}')
  # print(
  #     f'param shape = {model.layers[layer_idx].kernel.shape}, param_shape =   {param_list[param_idx].shape}'
  # )
  #print(param_list[param_idx])
  model.layers[layer_idx].kernel.assign(param_list[param_idx])
  # model.layers[layer_idx].kernel = param_list[param_idx]
  param_idx += 1
  print(model.layers[layer_idx].use_bias)
  if (model.layers[layer_idx].use_bias):
    model.layers[layer_idx].bias.assign(param_list[param_idx])
    # model.layers[layer_idx].bias = param_list[param_idx]
    param_idx += 1
  return model, param_idx


def set_resnet_layer_params(model, layer_idx, param_list, param_idx):
  """Set params for resnet layers.

  Params:
    model (keras.Model):
      keras model of network
    params (list(tf.Varaibles)):
      list of variables for the model
    layer_idx (int):
      index for the current layer in the model
    param_idx (int):
      index to tell us the starting point in the list of model parameters to
    look at.

  Returns
  -------
    updated Model with params set, incremented param_idx.
  """
  # update conv params
  # model, param_idx = set_dense_or_conv_params( model, layer_idx,
  #                                             param_list, param_idx)
  #
  model.layers[layer_idx].conv.kernel.assign(param_list[param_idx])
  param_idx += 1
  if (model.layers[layer_idx].conv.use_bias):
    model.layers[layer_idx].conv.bias.assign(param_list[param_idx])
    param_idx += 1
  # now need to update normalisation layer params
  model.layers[layer_idx].norm.tau.assign(param_list[param_idx])
  model.layers[layer_idx].norm.gamma.assign(param_list[param_idx + 1])
  model.layers[layer_idx].norm.beta.assign(param_list[param_idx + 2])
  # increment param_idx by three
  param_idx += 3
  # if(isinstance(model.layers[layer_idx], FRN)):
  #   model, param_idx = set_frn_layer_params(model, layer_idx,
  #                                           param_list, param_idx)
  # else:
  #   raise NotImplementedError('Currently only supporting FRN normalisation')
  #
  return model, param_idx


def set_resnet50_block_params(model, layer_idx, param_list, param_idx):
  """Set params for resnet 50 blocks.

  The Resnet50 blocks are set such that is;
    shortcut = self.shortcut(input)
    x = self.conv1(input)
    x = self.norm1(x)
    x = self.activation1(x)
    x = self.conv2(x)
    x = self.norm2(x)
    x = self.conv3(x)
    x = self.conv3(x)
    x = self.norm3(x)
    x = layers.Add(name=name + '_add')([shortcut, x])
    x = self.activation2(x)
    return x

  where activation will typically be a swish (no params)
  and the normalisation will be a Frn layer.

  This function will handle setting all of these, by calling
  the specific functions that update these parameters, and then
  increment the parameter index in tern.

  Params:
    model (keras.Model):
      keras model of network
    params (list(tf.Varaibles)):
      list of variables for the model
    layer_idx (int):
      index for the current layer in the model
    param_idx (int):
      index to tell us the starting point in the list of model parameters to
    look at.

  Returns
  -------
    updated Model with params set, incremented param_idx.
  """
  # update shortcut params
  model, param_idx = set_shortcut_params(model, layer_idx, param_list,
                                         param_idx)
  # update conv1
  model, param_idx = set_dense_or_conv_params(model, layer_idx, param_list,
                                              param_idx)
  # norm 1
  model, param_idx = set_frn_layer_params(model, layer_idx, param_list,
                                          param_idx)
  # conv 2
  model, param_idx = set_dense_or_conv_params(model, layer_idx, param_list,
                                              param_idx)
  # norm 2
  model, param_idx = set_frn_layer_params(model, layer_idx, param_list,
                                          param_idx)
  # conv 3
  model, param_idx = set_dense_or_conv_params(model, layer_idx, param_list,
                                              param_idx)
  # norm 3
  model, param_idx = set_frn_layer_params(model, layer_idx, param_list,
                                          param_idx)
  return model, param_idx


def set_shortcut_params(model, layer_idx, param_list, param_idx):
  """Set params for shortcut layer if needed.

  Within resnet50, some blocks will include a shortcut layer with convolutional
  params and some normalisation. This fn will update and set those params if
  needed. If no conv block is used, an Identity layer is used which contains no
  parameters. In this case, the function won't set anything and will simply
  return the original model.

  Parameters
  ----------
  model : keras.Model
      model or network we are sampling from.
  layer_idx : int
      index for the current layer
  param_list : list(tf.Tensor)
      list of parameters for the whole network
  param_idx : int
      the index for the starting parameter that needs to be set.

  Returns
  -------
    updated Model with params set, incremented param_idx.

  """
  if (isinstance(model.layers[layer_idx], ResNetShortcut)):
    # then update the shortcut parameters, which are a conv block followed
    # by a normalisation
    model, param_idx = set_dense_or_conv_params(model, layer_idx, param_list,
                                                param_idx)
    model, param_idx = set_frn_layer_params(model, layer_idx, param_list,
                                            param_idx)
  return model, param_idx


def set_frn_layer_params(model, layer_idx, param_list, param_idx):
  """Set params for FRN layer.

  sets tau, gamma and beta parameters

  Params:
    model (keras.Model):
      keras model of network
    params (list(tf.Varaibles)):
      list of variables for the model
    layer_idx (int):
      index for the current layer in the model
    param_idx (int):
      index to tell us the starting point in the list of model parameters to
    look at.

  Returns
  -------
    updated Model with params set, incremented param_idx by thre.
  """
  model.layers[layer_idx].tau.assign(param_list[param_idx])
  model.layers[layer_idx].gamma.assign(param_list[param_idx + 1])
  model.layers[layer_idx].beta.assign(param_list[param_idx + 2])
  # increment param_idx by three
  param_idx += 3
  return model, param_idx


def set_variational_model_params(model, param_list):
  """Apply params from a point estimate network to a variational network.

  Will set the point estimate values to the means of the new models.

  Parameters
  ----------
  model : keras.Model
      model/network we are sampling from.
  param_list : list(tf.Tensor)
      list of trainable parameters we need to update, basically the one's we are
      sampling from or optimizing.

  Returns
  -------
  keras.Model
    model with params from `param_list` applied.
  """
  # print('look at param list')
  # print(type(param_list))
  # for p in param_list:
  #   print(type(p))
  param_idx = 0
  for layer_idx in range(0, len(model.layers)):
    # if we are using nfnets model, we will need to access the base
    # sub network which was created with the functional method.

    # set model params, but make sure this current layer isn't a flatten or
    # pooling layer with no params to set
    if (isinstance(
        model.layers[layer_idx], (tfp.layers.DenseReparameterization,
                                  tfp.layers.Convolution2DReparameterization))):
      model, param_idx = set_variational_dense_or_conv_params(model,
                                                              layer_idx,
                                                              param_list,
                                                              param_idx)
    else:
      print('LAYER TYPE = {}, class = {}'.format(
          type(model.layers[layer_idx]), model.layers[layer_idx].__class__))
    #print('bias after= {}'.format(model.layers[i].bias))
  return model


def set_variational_dense_or_conv_params(model, layer_idx, param_list, param_idx):
  """Set mean variational params for dense or conv layers

  Only one function for both conv and dense layers, as they have the same
  number of parameters and parameter names (kernel and bias).

  Params:
    model (keras.Model):
      keras model of network
    params (list(tf.Varaibles)):
      list of variables for the model
    layer_idx (int):
      index for the current layer in the model
    param_idx (int):
      index to tell us the starting point in the list of model parameters to
    look at.

  Returns
  -------
    updated Model with params set, incremented param_idx
  """
  model.layers[layer_idx].kernel_posterior.distribution._loc.assign(param_list[param_idx])
  param_idx += 1
  print(model.layers[layer_idx].bias_posterior)
  if (model.layers[layer_idx].bias_posterior is not None):
    model.layers[layer_idx].bias_posterior.distribution._loc.assign(param_list[param_idx])
    param_idx += 1
  return model, param_idx


def set_model_params_hmc(model, param_list):
  """Apply new params to a model suitable for HMC

  Works similar to methods used for pdmp samplers, except will
  replace the params instead of just assigning them. This will allow it to
  work with tensorflow probabilities hmc method.

  When sampling, will need to apply previous params of a model, or sometimes
  maybe to apply the initial state of model params. This function will take in a
  list of parameters, parameters of the same format as that returned from
  `get_model_state`, and update the mode to use this new params.

  Parameters
  ----------
  model : keras.Model
      model/network we are sampling from.
  param_list : list(tf.Tensor)
      list of trainable parameters we need to update, basically the one's we are
      sampling from or optimizing.

  Returns
  -------
  keras.Model
    model with params from `param_list` applied.
  """
  # print('look at param list')
  # print(type(param_list))
  # for p in param_list:
  #   print(type(p))
  param_idx = 0
  for layer_idx in range(0, len(model.layers)):
    # if we are using nfnets model, we will need to access the base
    # sub network which was created with the functional method.
    if "Functional" == model.layers[layer_idx].__class__.__name__:
      param_idx = set_nfnet_base_params(model.layers[layer_idx], param_list,
                                        param_idx)
    # set model params, but make sure this current layer isn't a flatten or
    # pooling layer with no params to set
    elif (isinstance(model.layers[layer_idx], Dense) or
          isinstance(model.layers[layer_idx], Conv2D) or
          isinstance(model.layers[layer_idx], ScaledStandardizedConv2D)):
      model, param_idx = set_dense_or_conv_params_hmc(model, layer_idx, param_list,
                                                      param_idx)
    elif isinstance(model.layers[layer_idx], ResNetLayer):
      model, param_idx = set_resnet_layer_params_hmc(model, layer_idx, param_list,
                                                     param_idx)
    elif isinstance(model.layers[layer_idx], FRN):
      model, param_idx = set_frn_layer_params_hmc(model, layer_idx, param_list,
                                                  param_idx)
    else:
      pass
      # print('LAYER TYPE = {}, class = {}'.format(
      #     type(model.layers[layer_idx]), model.layers[layer_idx].__class__))
    #print('bias after= {}'.format(model.layers[i].bias))
  return model


def set_dense_or_conv_params_hmc(model, layer_idx, param_list, param_idx):
  """Set params for dense or conv net but with HMC compatability

  Only one function for both conv and dense layers, as they have the same
  number of parameters and parameter names (kernel and bias).

  Params:
    model (keras.Model):
      keras model of network
    params (list(tf.Varaibles)):
      list of variables for the model
    layer_idx (int):
      index for the current layer in the model
    param_idx (int):
      index to tell us the starting point in the list of model parameters to
    look at.

  Returns
  -------
    updated Model with params set, incremented param_idx
  """
  # print(f'layer index =  {layer_idx}, param_idx = {param_idx}')
  # print(
  #     f'param shape = {model.layers[layer_idx].kernel.shape}, param_shape =   {param_list[param_idx].shape}'
  # )
  print(f'param_idx  = {param_idx}')
  print(f'len param_list  = {len(param_list)}')
  if len(param_list) == 1:
    print(param_list[0])
  model.layers[layer_idx].kernel = param_list[param_idx]
  # model.layers[layer_idx].kernel = param_list[param_idx]
  param_idx += 1
  print(model.layers[layer_idx].use_bias)
  if (model.layers[layer_idx].use_bias):
    model.layers[layer_idx].bias = param_list[param_idx]
    # model.layers[layer_idx].bias = param_list[param_idx]
    param_idx += 1
  return model, param_idx


def set_resnet_layer_params_hmc(model, layer_idx, param_list, param_idx):
  """Set params for resnet layers.

  Params:
    model (keras.Model):
      keras model of network
    params (list(tf.Varaibles)):
      list of variables for the model
    layer_idx (int):
      index for the current layer in the model
    param_idx (int):
      index to tell us the starting point in the list of model parameters to
    look at.

  Returns
  -------
    updated Model with params set, incremented param_idx.
  """
  # update conv params
  # model, param_idx = set_dense_or_conv_params( model, layer_idx,
  #                                             param_list, param_idx)
  #
  model.layers[layer_idx].conv.kernel = param_list[param_idx]
  param_idx += 1
  if (model.layers[layer_idx].conv.use_bias):
    model.layers[layer_idx].conv.bias = param_list[param_idx]
    param_idx += 1
  # now need to update normalisation layer params
  model.layers[layer_idx].norm.tau = param_list[param_idx]
  model.layers[layer_idx].norm.gamma = param_list[param_idx + 1]
  model.layers[layer_idx].norm.beta = param_list[param_idx + 2]
  # increment param_idx by three
  param_idx += 3
  # if(isinstance(model.layers[layer_idx], FRN)):
  #   model, param_idx = set_frn_layer_params(model, layer_idx,
  #                                           param_list, param_idx)
  # else:
  #   raise NotImplementedError('Currently only supporting FRN normalisation')
  #
  return model, param_idx




@tf.function
def pred_forward_pass(model, param_list, x):
  model = set_model_params(model, param_list)
  out = model(x)
  print(out.shape)
  return out

@tf.function
def pred_forward_pass_hmc(model, param_list, x):
  model = set_model_params_hmc(model, param_list)
  out = model(x)
  return out



def bnn_log_likelihood(model):

  def log_likelihood_eval(x):
    pred = model(x)
    #print('pred shape = {}'.format(pred.shape))
    #print(model.likelihood_fn)
    #print(model.summary())
    log_likelihood_fn = model.likelihood_fn(pred)
    return log_likelihood_fn

  return log_likelihood_eval


def bnn_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X, y):

  def _fn(*args):
    with tf.name_scope('bnn_joint_log_prob_fn'):
      print(args)
      weights_list = args[::2]
      biases_list = args[1::2]
      # set the model weights and bias params
      m = set_model_params(model, weights_list, biases_list)
      # likelihood of predicted labels
      log_likelihood_fn = bnn_log_likelihood(m)
      print(log_likelihood_fn)
      log_likelihood_dist = log_likelihood_fn(X)
      # add the log likelihood now
      lp = tf.reduce_sum(log_likelihood_dist.log_prob(y))
      return lp

  return _fn


def bnn_neg_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X, y):
  joint_log_prob = bnn_joint_log_prob_fn(model, weight_prior_fns,
                                         bias_prior_fns, X, y)
  return lambda *x: -1.0 * joint_log_prob(*x)


def bnn_map_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X, y):

  def _fn():
    with tf.name_scope('bnn_joint_log_prob_fn'):
      print(args)
      weights_list = args[::2]
      biases_list = args[1::2]
      # adding prior component from the joint dist.
      lp = sum([
          tf.reduce_sum(fn.log_prob(w))
          for fn, w in zip(weight_prior_fns, weights_list)
      ])
      lp += sum([
          tf.reduce_sum(fn.log_prob(b))
          for fn, b in zip(bias_prior_fns, biases_list)
      ])
      # set the model weights and bias params
      m = set_model_params(model, weights_list, biases_list)
      # likelihood of predicted labels
      log_likelihood_fn = bnn_log_likelihood(m)
      print(log_likelihood_fn)
      log_likelihood_dist = log_likelihood_fn(X)
      # add the log likelihood now
      lp += tf.reduce_sum(log_likelihood_dist.log_prob(y))
      return lp

  return _fn


def bnn_map_neg_joint_log_prob_fn(model, weight_prior_fns, bias_prior_fns, X,
                                  y):
  joint_log_prob = bnn_joint_log_prob_fn(model, weight_prior_fns,
                                         bias_prior_fns, X, y)
  return lambda *x: -1.0 * joint_log_prob(*x)


def get_map(target_log_prob_fn,
            state,
            model,
            num_iters=100000,
            save_every=100,
            initial_lr=0.001,
            decay_rate=0.1,
            decay_steps=10000):
  """obtain a MAP estimate"""
  num_steps = num_iters // decay_steps
  boundaries = np.linspace(decay_steps, num_iters, num_steps)
  values = [initial_lr
           ] + [(initial_lr * decay_rate**i) for i in range(num_steps)]
  print('lr boundaries = {}'.format(boundaries))
  print('lr values = {}'.format(values))
  # Set up M-step (gradient descent).
  learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries, values)
  opt = tf.keras.optimizers.Adam(learning_rate=0.001)  #learning_rate)
  state_vars = state  #[tf.Variable(s) for s in state]

  def map_loss():
    return target_log_prob_fn(*state_vars)

  @tf.function
  def minimize():
    #print('state_vars = {}'.format(state_vars))
    with tf.GradientTape() as tape:
      loss = map_loss()
      grads = tape.gradient(loss, state_vars)
    opt.apply_gradients(zip(grads, state_vars))  #trainable_variables))

  for _ in range(num_iters):
    minimize()
  keras.backend.clear_session()
  # return the state of the model now
  return get_model_state(model)


def get_mle(model, x, y, num_iters=1000, save_every=100):
  """obtain a MAP estimate"""

  #@tf.function
  def _fn():
    model.train_on_batch(x, y)

  for _ in range(0, num_iters):
    _fn()

  return get_model_state(model)


def get_map_iter(iter_target_log_prob_fn,
                 state,
                 model,
                 likelihood_fn,
                 num_iters=2000,
                 save_every=100,
                 decay_steps=50000,
                 eval_step=5000,
                 initial_lr=1e3,
                 decay_rate=0.9,
                 dataset=None,
                 opt_str='adam',
                 X_test=None,
                 y_test=None,
                 X_train=None,
                 y_train=None):
  """obtain a MAP estimate for method with iteraor over data"""
  if opt_str == 'sgd':
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
      boundaries = [5000, 10000, 15000, 20000]
      values = [
          initial_lr, initial_lr * 0.1, initial_lr * 0.01, initial_lr * 0.001,
          initial_lr * 0.0001
      ]
      scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
          boundaries, values, name=None)
      opt = tf.optimizers.SGD(learning_rate=scheduler, momentum=0.9)
  else:
    opt = tf.optimizers.Adam(learning_rate=0.001)
  state_vars = state

  def map_loss():
    target_log_prob_fn = iter_target_log_prob_fn()
    return target_log_prob_fn(*state_vars)

  def accuracy_loss_fn(model, X, y):
    """get some loss metrics to see how we are going.

    want to get accuracy, neg log likelihood, and neg prior to track.

    Need to have the loss specified anew, since the target log joint prob
    is specified to be handled with the training data.
    """
    logits = model.predict(X)
    print('l shape = {}'.format(logits.shape))
    likelihood_dist = likelihood_fn(logits)
    neg_log_likelihood = -1.0 * tf.reduce_sum(likelihood_dist.log_prob(y))
    # combine the prior list into a single array, as
    neg_log_prior = tf.reduce_sum(model.losses)
    loss = neg_log_likelihood + neg_log_prior
    accuracy = accuracy_score(tf.argmax(y, axis=1), tf.argmax(logits, axis=1))
    return loss, neg_log_likelihood, neg_log_prior, accuracy

  # history = model.fit(X_train, y_train, batch_size=80,
  #                     epochs=2000, verbose=True,
  #                     validation_data=(X_test, y_test))
  def eval_test(X, y, i):
    loss_train, neg_log_likelihood_train, neg_log_prior_train, accuracy_train = accuracy_loss_fn(
        model, X_train, y_train)
    print(
        'Training Data: loss = {}, Neg_ll = {}, Neg_prior = {}, Accuracy = {}'.
        format(loss_train, neg_log_likelihood_train, neg_log_prior_train,
               accuracy_train))
    loss_test, neg_log_likelihood_test, neg_log_prior_test, accuracy_test = accuracy_loss_fn(
        model, X, y)
    print('Testing Data: loss = {}, Neg_ll = {}, Neg_prior = {}, Accuracy = {}'.
          format(loss_test, neg_log_likelihood_test, neg_log_prior_test,
                 accuracy_test))
    print('', flush=True)


  @tf.function
  def minimize():
    #print('state_vars = {}'.format(state_vars))
    #print('bias before = {}'.format(model.layers[-1].bias))
    with tf.GradientTape() as tape:
      loss = map_loss()
      grads = tape.gradient(loss, state_vars)
    #print('grads = {}'.format(grads))
    opt.apply_gradients(zip(grads, state_vars))  #trainable_variables))
    #print('bias after = {}'.format(model.layers[-1].bias))

  if (X_test is not None) and (y_test is not None):
    eval_fn = eval_test
  else:
    eval_fn = lambda X, y, i: None

  for i in tqdm(range(num_iters)):
    minimize()
    if i % eval_step == 0:
      eval_fn(X_test, y_test, i)

  keras.backend.clear_session()
  # return the state of the model now
  return get_model_state(model)


@tf.function
def model_fwd(model, data):
  return model(data, training=False)


# @tf.function
def distributed_eval_step(model, data, labels, likelihood_fn, test_loss,
                          test_accuracy, strategy):
  replica_results = strategy.run(eval_step,
                                 args=(
                                     model,
                                     data,
                                     labels,
                                     likelihood_fn,
                                     test_loss,
                                     test_accuracy,
                                 ))
  # return strategy.gather(replica_results, 0)
  return strategy.experimental_local_results(replica_results)


def eval_step(model, data, labels, likelihood_fn, loss_metric, accuracy_metric):
  logits = model.call(data, training=False)
  print(model.trainable_variables)
  likelihood = likelihood_fn(labels, logits)
  loss_metric.update_state(likelihood)
  accuracy_metric.update_state(labels, logits)
  return logits


def accuracy_loss_fn(model, ds, likelihood_fn, test_loss, test_accuracy, strategy, num_train_eval=1):
  """Get some evaluation metrics from training.

  Want to be able to get some summary stats from my model performance as I go.
  This function will allow me to do this for both the train and
  validation/test sets. The train and test/val data sets are represented
  differently. The training set is an iterator that will run on forever, while
  the val/test set will be a tf.Dataset with fixed length and no shuffling
  etc. This function will do a check to see if the Dataset supplied is of
  either type and handle it accordingly. If dataset is an iterator, will sample
  data from `num_train_eval` batches for evaluation from our training set. If is
  from our validation set, it will run over the entire dataset to generate metrics.

  Parameters
  ----------
  model : keras.Model
      Our model to evaluate.
  ds : tf.data.Dataset or iterator
      dataset to evaluate on.
  likelihood_fn: callable
      likelihood function.
  strategy : tf.distribute.Strategy
      Distribution strategy used for training
  num_train_eval : int
      number of batches to evaluate on if the dataset is an iterator for the
      training set. If the dataset supplied is a tf.Dataset object for the
      val/test set, then this value is ignored.
  Returns
  -------
    loss: float
      The total loss over the dataset (neg log likelihood + neg log prior)
     neg_log_likelihood: float
       neg log likelihood over the evaluated data
     neg_log_priot: float
       neg_log_prior over params
     accuracy:
       accuracy over evaluated data
   """
  # creating list that will hold the logits from each batch
  logits_list = []
  # same thing for the labels
  labels_list = []
  # if the dataset is an iterator, which will be the case for the training data,
  # we will just run a certain number of iterations and not necessarily go over the
  # entire dataset.
  # If the dataset is listed is of type tf.Dataset, then we can iterate over the
  # entire thing
  if isinstance(ds, (tf.data.Dataset, tf.distribute.DistributedDataset)):
    # is our testing set, so let's iterate over it all.
    for inputs_batch, labels_batch in ds:
      print(inputs_batch.shape)
      logits = distributed_eval_step(model, inputs_batch, labels_batch, likelihood_fn,
                                     test_loss, test_accuracy,
                                     strategy)
  else:
    # is our training set, and thus is an iterator, so we can't use the above logic
    # since out method for training and sampling uses an iterator made over a dataset
    # that will repeat indefinitely.
    print('in eval for training')
    for _ in range(0, num_train_eval):
      inputs_batch, labels_batch = ds.next()
      # print(inputs_batch.shape)
      logits = distributed_eval_step(model, inputs_batch, labels_batch, likelihood_fn, test_loss, test_accuracy,
                                     strategy)
      logits_list.extend(logits)
      labels = strategy.gather(labels_batch, axis=0)
      labels_list.extend(labels)
  # # now concatenate the list of logits and labels into a single array
  # logits = tf.stack(logits_list, axis=0)
  # labels = tf.stack(labels_list, axis=0)
  # neg_log_likelihood = tf.reduce_sum(likelihood_fn(labels, logits))
  # # combine the prior list into a single array, as
  # neg_log_prior = tf.reduce_sum(model.losses)
  # loss = neg_log_likelihood + neg_log_prior
  # accuracy = accuracy_score(tf.argmax(labels, axis=1), tf.argmax(logits,
  #                                                                axis=1))
  # return loss, neg_log_likelihood, neg_log_prior, accuracy


def get_map_test(distributed_opt_step,
                 compute_loss,
                 training_iter,
                 model,
                 likelihood_fn,
                 optimizer,
                 strategy,
                 test_dataset=None,
                 num_iters=2000,
                 eval_step=4999,
                 tboard_iter=20):
  """obtain a MAP estimate for method with iteraor over data.

  Takes input the optimizers step for the model we are looking at, and will
  perform optimization to get the MAP estimate. This function assumes the
  `distributed_op_step` is defined for the negative joint log probability of the
  model of interest, but in reality it could be any function that explicitly
  calls the gradient.tape() methods and performs updates of parameters.

  Parameters
  ----------
  distributed_opt_step : callable
      function we are optimizing. This function should be iterating over the
      training data we need, and be handling explicit calls the gradient.tape()
      and the update of parameters given the optimizer we have chosen.
      This function should take as arguments:
      `(model, likelihood_fn, X, y, strategy, optimizer)`
  training_iter : iterable
      iterable of the training data. The `distributed_op_step` should be
      iterating over this for training purposes. This is only included to allow
      us to test perform some tests during the training process.
  model : keras.Model
      the model we are looking at.
  likelihood_fn : callable
      Defines the negative
  optimizer : keras.optimizer.Optimizer
      optimizer that is used during training.
  strategy : tf.distributed.Strategy
      Distribution strategy used for this mode.
  test_dataset : tf.data.Dataset
      Dataset object used so we can evaluate performance during training. NOTE:
      this is tf.data.Dataset, whilst the training data is an iterable. This is
      because we generally want to have a bit more control over when a new batch
      is generated during training.
  num_iters : int
      number of training iterations.
  eval_step : int
      step to perorm model evaluation

  Returns
  -------
  list of tensors for the model parameters at the MAP.

  """
  # creating metrics for evaluation steps
  with strategy.scope():
    test_loss = tf.keras.metrics.Sum(name='test_loss')
    test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')
    train_loss = tf.keras.metrics.Sum(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

  # create a distributed version of the testing data set
  test_dataset_distributed = strategy.experimental_distribute_dataset(test_dataset)

  def eval_fn():
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    accuracy_loss_fn(
      model, training_iter, likelihood_fn, train_loss, train_accuracy, strategy)
    print('Training Data:  Neg_ll = {}, Accuracy = {}'.format(train_loss.result(),
                                                              train_accuracy.result()))
    accuracy_loss_fn(
      model, test_dataset_distributed, likelihood_fn, test_loss, test_accuracy, strategy)
    print('Testing Data:  Neg_ll = {}, Accuracy = {}'.format(test_loss.result(),
                                                             test_accuracy.result()))
    print('', flush=True)

  summary_writer = tf.summary.create_file_writer('logs/')
  for train_step_count in tqdm(range(num_iters)):
    distributed_opt_step()
    if(train_step_count % tboard_iter == 0):
      with summary_writer.as_default():
        tf.summary.scalar(
          'learning_rate',
          optimizer.learning_rate,
          step=train_step_count)
        tf.summary.scalar(
          'prior',
          tf.reduce_sum(model.losses),
          step=train_step_count)
        tf.summary.scalar(
          'test_neg_ll',
          test_loss.result(),
          step=train_step_count)



    if train_step_count % eval_step == 0 and train_step_count > 0:
      eval_fn()
  return get_model_state(model)


def trace_fn(current_state, results, summary_freq=100):
  return results


@tf.function
def graph_hmc(*args, **kwargs):
  """Compile static graph for tfp.mcmc.sample_chain.
    Since this is bulk of the computation, using @tf.function here
    signifcantly improves performance (empirically about ~5x).
    """
  return tfp.mcmc.sample_chain(*args, **kwargs)


def nest_concat(*args):
  return tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=0), *args)


def build_prior(layers):
  print('building prior = {}'.format(layers))
  weights_prior = []
  bias_prior = []
  prior_units = [2, *layers]
  for units in prior_units[:-1]:
    p_scale = tf.sqrt(1.0 / tf.cast(units, dtype=tf.float32))
    weights_prior.append(tfd.Normal(loc=0., scale=p_scale))
    bias_prior.append(tfd.Normal(loc=0., scale=p_scale))
  return weights_prior, bias_prior


def find_num_iters_and_loops(num_params,
                             num_samples,
                             gpu_memory=6e9,
                             gpu_ratio=0.05,
                             precision='single'):
  """ Find number of loops needed to get the specified number of samples

  Refer to the docstring of this module for more of a description about why
  this is needed.


  The amount of memory we need is defined by the memory needed for
  all paramaters of a single loop, + the past samples.


  The memory needed for a single sample will be the number of parameters
  times the number of samples desired (plus kernel results from trace_fn).
  Will also need aditional memory loop will be the memory for parameters needed
  for a single iteration, which will be the amount of memory needed for a single
  sample + the extra params (so for BPS will be sample state, gradients and
  velocity, so will be the memory of a single sample time 3).

  Args:
    num_params (int):
      number of WEIGHTS in the model.
    num_samples (int):
      total number of samples needed
    gpu_memory (int):
      GPU memory available in Bytes.
    gpu_ratio (float):
      ratio of memory to allocate for. Should be in (0, 1)
    precision (str):
      if we are using 'single', 'double' or even 'half' floating point precision

  returns:
    number of iters per loop, and the number of loops needed.
  """
  num_samples = np.int32(num_samples)  #.astype(np.int32)
  # find the total amount of memory per weight (in bytes).
  if (precision == 'single'):
    bytes_per_weight = 4
  elif (precision == 'double'):
    bytes_per_weight = 8
  elif (precision == 'half'):
    bytes_per_weight = 2
  else:
    raise ValueError('Incorrect value specified for precision arg.')
  # memory of a single sample will be the number of bytes per weight time
  # the number of samples
  single_sample_memory = np.int(num_params * bytes_per_weight)
  # memory needed for a single iteration is equal to the amount of
  # memory needed for a single sample times three, since
  # need to store the params, velocity and gradients
  single_iter_memory = single_sample_memory * 3
  # now find the total amounbt of memory needed
  total_memory = single_iter_memory + single_sample_memory * num_samples
  print('Total memory = {}'.format(total_memory))
  print('GPU memory = {}'.format(gpu_memory))
  print('float loops = {}'.format(gpu_ratio * gpu_memory / total_memory))
  # now see how many times we will need to loop in order to get this
  # number of samples
  num_samples_per_loop = np.min([
      np.ceil(total_memory / (gpu_ratio * gpu_memory)).astype(np.int32),
      num_samples
  ])
  num_loops = np.min([
      np.ceil(num_samples / num_samples_per_loop).astype(np.int32), num_samples
  ])
  return num_samples, np.int32(1)

def boomerang_gaussian_measure(state, preconditioner, mean):
  # get the inverse of the preconditioner
  # now the state component
  state_part = tf.reduce_sum([
      tf.reduce_sum(tf.square(s - m) / (2.0 * p))
      for s, m, p in zip(state, mean, preconditioner)
  ])
  return state_part


def boomerang_neg_log_likelihood(model, likelihood_fn, X, y):
  logits = model(X)
  print('likelihood_fn is = {}'.format(likelihood_fn))
  log_likelihood_dist = likelihood_fn(logits)  #, scale=0.05)
  # add the log likelihood now
  lp = tf.reduce_sum(log_likelihood_dist.log_prob(y))
  return lp


def boomerang_bnn_neg_joint_log_prob_fn(model, likelihood_fn, X, y,
                                        preconditioner, mean):

  def _fn(*param_list):
    with tf.name_scope('boomerang_bnn_joint_log_prob_fn'):
      # set the model params
      m = set_model_params(model, param_list)
      # print('current  model params ttttt= {}'.format(model.layers[1].kernel))
      # neg log likelihood of predicted labels
      neg_ll = boomerang_neg_log_likelihood(model, likelihood_fn, X, y)
      # now get the losses from the prior (negative log prior)
      # these are stored within the models `losses` variable
      gaussian_measure = boomerang_gaussian_measure(param_list, preconditioner,
                                                    mean)

      neg_lp = tf.reduce_sum(m.losses)
      # add them together for the total loss
      return neg_ll + gaussian_measure + neg_lp

  return _fn


# @tf.function
def boomerang_one_step(state, kernel, prev_kernel_results):
  next_state, next_kernel_results = kernel.one_step(state, prev_kernel_results)
  return next_state, next_kernel_results


def neg_boomerang_joint(model, likelihood_fn, X, y,
                        dataset_size, batch_size,
                        param_list, preconditioner,
                        mean):
  # neg log likelihood of predicted labels
  pred = model(X)
  neg_ll = dataset_size / batch_size * likelihood_fn(y, pred)
  # these are stored within the models `losses` variable
  gaussian_measure = boomerang_gaussian_measure(param_list, preconditioner,
                                                mean)
  # add them together for the total loss
  return neg_ll + gaussian_measure


def gradient_boomerang_step(model, likelihood_fn, X, y,
                            dataset_size, batch_size,
                            param_list,
                            preconditioner, mean):
  # set the model params
  m = set_model_params(model, param_list)
  with tf.GradientTape() as tape:
    neg_log_prob = neg_boomerang_joint(m, likelihood_fn, X, y,
                                       dataset_size, batch_size,
                                       param_list,
                                       preconditioner, mean)
  gradients = tape.gradient(neg_log_prob, m.trainable_variables)
  return gradients


def distributed_gradient_boomerang_step(model, likelihood_fn, X, y,
                                        dataset_size, batch_size,
                                        preconditioner, mean, strategy):

  def _fn(param_list):
    per_replica_gradient = strategy.run(gradient_boomerang_step,
                                        args=(
                                            model,
                                            likelihood_fn,
                                            X,
                                            y,
                                            dataset_size,
                                            batch_size,
                                            param_list,
                                            preconditioner,
                                            mean,
                                        ))
    return strategy.reduce(tf.distribute.ReduceOp.SUM,
                           per_replica_gradient,
                           axis=None)

  return _fn


def iter_grad_boomerang_fn(model, likelihood_fn, dataset_iter,
                           data_size, batch_size,
                           preconditioner,
                           mean, strategy):

  dataset_size = tf.convert_to_tensor(data_size, dtype=tf.float32)
  batch_size = tf.convert_to_tensor(batch_size, dtype=tf.float32)
  def _fn():
    X, y = dataset_iter.next()
    return distributed_gradient_boomerang_step(model, likelihood_fn, X, y,
                                               dataset_size, batch_size,
                                               preconditioner, mean, strategy)

  return _fn


def bps_iter_main(model,
                  ipp_sampler_str,
                  lambda_ref,
                  std_ref,
                  num_results,
                  num_burnin_steps,
                  out_dir,
                  num_loops,
                  bnn_neg_joint_log_prob,
                  map_initial_state,
                  training_iter,
                  test_ds,
                  test_orig_ds,
                  likelihood_str,
                  batch_size,
                  data_size,
                  data_dimension_dict,
                  plot_results=True,
                  run_eval=True,
                  num_steps_between_results=0):
  """main method for running BPS on model"""
  print('running bps')
  start_time = time.time()
  # finding the number of samples to perform for each iteration
  ipp_sampler = InterpolationSampler
  # print('map_initial_state bps = {}'.format(map_initial_state))
  num_params = np.sum([x.numpy().size for x in map_initial_state])
  print('num_params = {}'.format(num_params))
  num_samples_per_loop = np.int32(num_results / num_loops)
  # create the kernel
  kernel = IterBPSKernel(parent_target_log_prob_fn=bnn_neg_joint_log_prob,
                         store_parameters_in_results=True,
                         ipp_sampler=ipp_sampler,
                         batch_size=batch_size,
                         data_size=data_size,
                         lambda_ref=lambda_ref,
                         std_ref=std_ref)
  # convert the init state into a tensor first
  init_state = [tf.convert_to_tensor(x) for x in map_initial_state]
  # BURNIN PHASE
  # start sampling for burnin, and then discard these samples
  # we may not be able to fit all the burnin samples in a single loop,
  # so we will loop over a few times if we need to
  print('num_samples_per_loop = {}'.format(num_samples_per_loop))
  print('num_loops = {}'.format(num_loops))
  num_burnin_iters = np.ceil(num_burnin_steps / num_samples_per_loop).astype(
      np.int)
  # create the trace function
  trace_fn = lambda _, pkr: (pkr.acceptance_ratio, pkr.time, pkr.proposed_time)
  # run bootstrap here to get the initial state for burnin
  # this allows us to start sampling at each loop exactly where we left off
  kernel_previous_results = kernel.bootstrap_results(init_state)
  # now run the burnin phase
  for burnin_iter in range(0, num_burnin_iters):
    print('burnin iter = {}'.format(burnin_iter))
    bps_results = graph_hmc(num_results=num_samples_per_loop,
                            current_state=init_state,
                            kernel=kernel,
                            previous_kernel_results=kernel_previous_results,
                            return_final_kernel_results=True,
                            trace_fn=trace_fn)
    # extract the chain and the final kernel results
    bps_chain = bps_results.all_states
    # final kernel results used to initialise next call of loop
    kernel_previous_results = bps_results.final_kernel_results
    # get the final state of the chain from the previous burnin iter
    init_state = [x[-1, ...] for x in bps_chain]
  end_warmup_time = time.time()
  print('time warmup = {}'.format(end_warmup_time - start_time))
  # SAMPLING PHASE
  # now loop over the actual samples from (hopefully) the posterior
  acceptance_list = []
  time_list = []
  proposed_time_list = []
  for loop_iter in range(0, num_loops):
    print('loop iter = {}'.format(loop_iter))
    bps_results = graph_hmc(num_results=num_samples_per_loop,
                            current_state=init_state,
                            kernel=kernel,
                            previous_kernel_results=kernel_previous_results,
                            return_final_kernel_results=True,
                            num_steps_between_results=num_steps_between_results,
                            trace_fn=trace_fn)

    # extract the chain and the final kernel results
    bps_chain = bps_results.all_states
    # add the acceptance ratios to the list
    acceptance_list.append(bps_results.trace[0].numpy())
    time_list.append(bps_results.trace[1].numpy())
    proposed_time_list.append(bps_results.trace[2].numpy())
    # final kernel results used to initialise next call of loop
    kernel_previous_results = bps_results.final_kernel_results
    # save these samples to file
    save_chain(bps_chain, out_dir, loop_iter)
    # get the final state of the chain from the previous loop iter
    init_state = [x[-1] for x in bps_chain]
  print('finished sampling')
  end_sampling_time = time.time()
  print('total sampling time = {}'.format(end_sampling_time - start_time))
  if len(acceptance_list) > 1:
    acceptance_ratio = np.concatenate(acceptance_list)
    result_time = np.concatenate(time_list)
    proposed_time = np.concatenate(proposed_time_list)
  else:
    acceptance_ratio = acceptance_list[0]
    result_time = time_list[0]
    proposed_time = proposed_time_list[0]
  print('acceptance_ratio = {}'.format(acceptance_ratio))
  print(type(acceptance_ratio))
  print('num acceptance_ratio > 1 = {}'.format(np.sum(
      (acceptance_ratio > 1.0))))
  print(f'result time = {result_time}')
  print(f'proposed time = {proposed_time}')
  np.save(os.path.join(out_dir, 'acceptance_ratio.npy'), acceptance_ratio)
  np.save(os.path.join(out_dir, 'proposed_times.npy'), proposed_time)
  np.save(os.path.join(out_dir, 'result_times.npy'), result_time)
  # plot the results if specified to
  if (plot_results or run_eval):
    print('plotting pred posterior and test accuracy')
    # if the output is one dimensional, will be regression task so plot using
    # regression methods
    if (data_dimension_dict['out_dim'] == 1):
      plot_pred_posterior(model, bps_chain, likelihood_str, num_results,
                          training_iter, test_ds, out_dir, 'bps')
      # plot_pred_posterior(model, chain, likelihood_str, num_results, training_iter,
      # test_ds, out_dir, name):

    # otherwise will be classification task, so plot using those methods
    else:
      if data_dimension_dict['in_channels'] == 1:
        plt_dims = [
            data_dimension_dict['in_height'], data_dimension_dict['in_width']
        ]
      else:
        plt_dims = [
            data_dimension_dict['in_height'], data_dimension_dict['in_width'],
            data_dimension_dict['in_channels']
        ]
      eval_plot_image_iter_pred_posterior(model, out_dir, training_iter,
                                          test_ds, test_orig_ds,
                                          data_dimension_dict['out_dim'],
                                          data_dimension_dict['out_dim'],
                                          plot_results, plt_dims)

@tf.function
def hmc_one_step(state, kernel, prev_kernel_results):
  next_state, next_kernel_results = kernel.one_step(state, prev_kernel_results)
  return next_state, next_kernel_results



def hmc_iter_main(model,
                  num_results,
                  num_burnin_steps,
                  out_dir,
                  step_size,
                  leapfrog_steps,
                  bnn_joint_log_prob,
                  map_initial_state,
                  training_iter,
                  test_ds,
                  test_orig_ds,
                  likelihood_str,
                  data_dimension_dict,
                  plot_results=True,
                  run_eval=True,
                  num_steps_between_results=1):
  """main method for running HMC on model"""
  print('running hmc')
  start_time = time.time()
  num_params = np.sum([x.numpy().size for x in map_initial_state])
  print('num_params = {}'.format(num_params))
  kernel = IterHMCKernel(parent_target_log_prob_fn=bnn_joint_log_prob,
                         step_size=step_size,
                         num_leapfrog_steps=leapfrog_steps,
                         store_parameters_in_results=True)
  # convert the init state into a tensor first
  next_state = [tf.convert_to_tensor(x) for x in map_initial_state]
  # initialise the chain for hmc
  # for now will just be empty list of lists
  hmc_chain = [[] for i in range(0, len(next_state))]
  print(hmc_chain)
  trace_fn = lambda _, pkr: pkr.log_acceptance_correction
  # run bootstrap here to get the initial state for burnin
  # this allows us to start sampling at each loop exactly where we left off
  kernel_results = kernel.bootstrap_results(next_state)
  start_time = time.time()
  # warmup
  for i in range(0, num_burnin_steps):
    next_state, kernel_results = hmc_one_step(next_state, kernel,
                                              kernel_results)

  print('end warmup')
  end_warmup_time = time.time()
  print(f'Warmup time = {end_warmup_time - start_time}')
  # sampling
  for i in range(0, int(num_results * num_steps_between_results)):
    next_state, kernel_results = hmc_one_step(next_state, kernel,
                                              kernel_results)
    if i % num_steps_between_results == 0:
      # add the found state to the hmc chain
      for i in range(0, len(next_state)):
        hmc_chain[i].append(next_state[i])

  end_sampling_time = time.time()
  print(f'total sampling time = {end_sampling_time - start_time}')
  # extract the chain and the final kernel results
  # want to convert the lists of lists to an array
  hmc_chain = [tf.stack(x, axis=0) for x in hmc_chain]
  # save these samples to file
  save_chain(hmc_chain, out_dir, 0)
  # get the final state of the chain from the previous loop iter
  init_state = [x[-1] for x in hmc_chain]
  print('finished sampling')
  end_sampling_time = time.time()
  print('total sampling time = {}'.format(end_sampling_time - start_time))
  if (plot_results or run_eval):
    print('plotting pred posterior and test accuracy')
    # if the output is one dimensional, will be regression task so plot using
    # regression methods
    if (data_dimension_dict['out_dim'] == 1):
      plot_pred_posterior(model, hmc_chain, likelihood_str, num_results,
                          training_iter, test_ds, out_dir, 'hmc')
      # plot_pred_posterior(model, chain, likelihood_str, num_results, training_iter,
      # test_ds, out_dir, name):

    # otherwise will be classification task, so plot using those methods
    else:
      if data_dimension_dict['in_channels'] == 1:
        plt_dims = [
            data_dimension_dict['in_height'], data_dimension_dict['in_width']
        ]
      else:
        plt_dims = [
            data_dimension_dict['in_height'], data_dimension_dict['in_width'],
            data_dimension_dict['in_channels']
        ]
      eval_plot_image_iter_pred_posterior(model, out_dir, training_iter,
                                          test_ds, test_orig_ds,
                                          data_dimension_dict['out_dim'],
                                          data_dimension_dict['out_dim'],
                                          plot_results, plt_dims,
                                          name='hmc')

def sgld_iter_main(model,
                   num_results,
                   out_dir,
                   num_loops,
                   sgld_opt_step,
                   map_initial_state,
                   training_iter,
                   test_ds,
                   test_orig_ds,
                   likelihood_str,
                   data_dimension_dict,
                   plot_results=True,
                   run_eval=True,
                   num_steps_between_results=0):
  """main method for running sgld on model"""
  print('running sgld')
  start_time = time.time()
  num_params = np.sum([x.numpy().size for x in map_initial_state])
  print('num_params = {}'.format(num_params))
  num_samples_per_loop = np.int32(num_results / num_loops)
  # convert the init state into a tensor first
  init_state = [tf.convert_to_tensor(x) for x in map_initial_state]
  model = set_model_params(model, init_state)
  for loop_iter in range(0, num_loops):
    sgld_chain = [[] for i in range(len(model.trainable_variables))]
    print(sgld_chain)
    print('loop iter = {}'.format(loop_iter))
    for i in range(0, num_samples_per_loop * (num_steps_between_results + 1)):
      # run sgld opt step
      sgld_opt_step()
      if i % (num_steps_between_results + 1) == 0:
        # add the updated model params to the list
        for param_idx in range(0, len(sgld_chain)):
          sgld_chain[param_idx].append(model.trainable_variables[param_idx].read_value())

    # stack the list of variables so params have dimensions of [n_samples, *param_dims]
    print(sgld_chain[-1])
    sgld_chain = [tf.stack(x, axis=0) for x in sgld_chain]
    # save these samples to file
    save_chain(sgld_chain, out_dir, loop_iter)
    # get the final state of the chain from the previous loop iter
  print('finished sampling')
  end_sampling_time = time.time()
  print('total sampling time = {}'.format(end_sampling_time - start_time))
  if (plot_results or run_eval):
    print('plotting pred posterior and test accuracy')
    # if the output is one dimensional, will be regression task so plot using
    # regression methods
    if (data_dimension_dict['out_dim'] == 1):
      plot_pred_posterior(model, sgld_chain, likelihood_str, num_results,
                          training_iter, test_ds, out_dir, 'sgld')
    # otherwise will be classification task, so plot using those methods
    else:
      if data_dimension_dict['in_channels'] == 1:
        plt_dims = [
            data_dimension_dict['in_height'], data_dimension_dict['in_width']
        ]
      else:
        plt_dims = [
            data_dimension_dict['in_height'], data_dimension_dict['in_width'],
            data_dimension_dict['in_channels']
        ]
      eval_plot_image_iter_pred_posterior(model, out_dir, training_iter,
                                          test_ds, test_orig_ds,
                                          data_dimension_dict['out_dim'],
                                          data_dimension_dict['out_dim'],
                                          plot_results, plt_dims)




def variational_iter_main(model,
                          num_results,
                          out_dir,
                          likelihood_fn,
                          optimizer,
                          map_initial_state,
                          training_ds,
                          test_ds,
                          test_orig_ds,
                          likelihood_str,
                          batch_size,
                          data_size,
                          data_dimension_dict,
                          plot_results=True,
                          run_eval=True):
  """main method for running variational inference"""
  print('running variational inference')
  init_state = [tf.convert_to_tensor(x) for x in map_initial_state]
  # set model parameters
  model = set_variational_model_params(model, init_state)
  # compile the model
  # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  model.compile(loss=likelihood_fn, optimizer=optimizer)
  # model.compile(loss=loss, optimizer=optimizer)
  sample_x, sample_y = next(iter(training_ds))
  print(model(sample_x).shape)
  print(sample_y.shape)
  start_inference_time = time.time()
  # start inference
  model.fit(training_ds, epochs=1, steps_per_epoch=num_results)
  end_inference_time = time.time()
  print(f'Inference time = {end_inference_time - start_inference_time}')
  # plot the results if specified to
  if (plot_results or run_eval):
    print('plotting pred posterior and test accuracy')
    # if the output is one dimensional, will be regression task so plot using
    # regression methods
    # get the raw data from the tf.Dataframes
    X_train, y_train = next(iter(training_ds))
    X_test, y_test = next(iter(test_ds))
    # make them all numpy arrays
    X_train = X_train.numpy()
    y_train = y_train.numpy()
    X_test = X_test.numpy()
    y_test = y_test.numpy()
    # otherwise will be classification task, so plot using those methods
    # perform prediction for each iteration
    num_plot = 100
    pred = np.zeros([num_plot, y_test.size])
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    plt.figure()
    plt.scatter(X_train, y_train, color='b', alpha=0.15)
    pred_idx = 0
    for sample_idx in range(0, num_plot):
      pred[pred_idx, :] = model(X_test).numpy().ravel()
      plt.plot(X_test, pred[pred_idx, :], alpha=0.05, color='k')
      pred_idx += 1
    # plt.axis('off')
    print('saving images to {}'.format(out_dir))
    print(os.path.join(out_dir, 'pred_vi.png'))
    plt.savefig(os.path.join(out_dir, 'pred_vi.png'))
    plt.savefig(os.path.join(out_dir, 'pred_vi.pdf'), bbox_inches='tight')


@tf.function
def cov_pbps_one_step(state, kernel, prev_kernel_results):
  next_state, next_kernel_results = kernel.one_step(state, prev_kernel_results)
  return next_state, next_kernel_results


def cov_pbps_test_iter_main(model,
                            ipp_sampler_str,
                            lambda_ref,
                            cov_ref,
                            std_ref,
                            num_results,
                            num_burnin_steps,
                            out_dir,
                            num_loops,
                            bnn_neg_joint_log_prob,
                            map_initial_state,
                            training_iter,
                            test_ds,
                            test_orig_ds,
                            likelihood_str,
                            batch_size,
                            data_size,
                            data_dimension_dict,
                            plot_results=True,
                            run_eval=True,
                            num_steps_between_results=0):
  """main method for running BPS on model with diagonal covariance
  preconditioned gradients.
  """
  print('running covariance preconditioned bps')
  print('ipp_sampler_str = {}'.format(ipp_sampler_str))
  start_time = time.time()
  # finding the number of samples to perform for each iteration
  print('map_initial_state bps = {}'.format(map_initial_state))
  num_params = np.sum([x.numpy().size for x in map_initial_state])
  print('num_params = {}'.format(num_params))
  num_samples_per_loop = np.int32(num_results / num_loops)
  # num_samples_per_loop, num_loops = find_num_iters_and_loops(
  # num_params, num_results)
  ipp_sampler = InterpolationPSampler
  bps_ipp_sampler = InterpolationSampler
  print('ipp_sampler = {}'.format(ipp_sampler))
  # create the kernel
  kernel = IterBPSKernel(parent_target_log_prob_fn=bnn_neg_joint_log_prob,
                         store_parameters_in_results=True,
                         ipp_sampler=bps_ipp_sampler,
                         batch_size=batch_size,
                         data_size=data_size,
                         lambda_ref=lambda_ref,
                         std_ref=std_ref)
  # convert the init state into a tensor first
  init_state = [tf.convert_to_tensor(x) for x in map_initial_state]
  # BURNIN PHASE
  # start sampling for burnin, and then discard these samples
  # we may not be able to fit all the burnin samples in a single loop,
  # so we will loop over a few times if we need to
  print('num_samples_per_loop = {}'.format(num_samples_per_loop))
  print('num_loops = {}'.format(num_loops))
  # run bootstrap here to get the initial state for burnin
  # this allows us to start sampling at each loop exactly where we left off
  next_state = init_state
  kernel_results = kernel.bootstrap_results(init_state)
  # create a current estimate for the running mean and variance
  running_mean = [tf.zeros_like(x) for x in init_state]
  delta1 = [tf.zeros_like(x) for x in init_state]
  delta2 = [tf.zeros_like(x) for x in init_state]
  M2 = [tf.ones_like(x) for x in init_state]
  print('num_burnin = {}'.format(num_burnin_steps))
  for i in range(0, num_burnin_steps):
    next_state, kernel_results = cov_pbps_one_step(next_state, kernel,
                                                   kernel_results)
    # now get running variance appro  x
    delta1 = [x - m for x, m in zip(next_state, running_mean)]
    # first need running mean approx
    # the + 1 in the denominator is because the loop starts at zero
    running_mean = [m + d / (i + 1) for m, d in zip(running_mean, delta1)]
    # recompute second delta with updated mean, needed for M2 for variance later on
    delta2 = [x - m for x, m in zip(next_state, running_mean)]
    M2 = [M + d1 * d2 for M, d1, d2 in zip(M2, delta1, delta2)]
  # can now get the variance calc
  std_preconditioner = [tf.math.sqrt(M / num_burnin_steps) for M in M2]
  # std_preconditioner = [tf.math.sqrt(M / num_burnin_steps) for M in M2]
  end_warmup_time = time.time()
  print('time warmup = {}'.format(end_warmup_time - start_time))
  # for i in range(0, len(online_var)):
  #   print('{} var shape = {}, param shape = {}'.format(i, online_var[i].shape,
  #                                                      init_state[i].shape))
  kernel = IterCovPBPSKernel(parent_target_log_prob_fn=bnn_neg_joint_log_prob,
                             store_parameters_in_results=True,
                             ipp_sampler=ipp_sampler,
                             batch_size=batch_size,
                             data_size=data_size,
                             lambda_ref=cov_ref,
                             std_ref=std_ref)
  kernel_results = kernel.bootstrap_results(init_state)
  kernel_previous_results = kernel_results._replace(
    preconditioner=std_preconditioner)
  # SAMPLING PHASE
  # now loop over the actual samples from (hopefully) the posterior
  acceptance_list = []
  time_list = []
  proposed_time_list = []
  trace_fn = lambda _, pkr: (pkr.acceptance_ratio, pkr.time, pkr.proposed_time, pkr.velocity)
  # create the trace function
  for loop_iter in range(0, num_loops):
    print('loop iter = {}'.format(loop_iter))
    bps_results = graph_hmc(num_results=num_samples_per_loop,
                            current_state=init_state,
                            kernel=kernel,
                            previous_kernel_results=kernel_previous_results,
                            return_final_kernel_results=True,
                            num_steps_between_results=num_steps_between_results,
                            trace_fn=trace_fn)
    # extract the chain and the final kernel results
    bps_chain = bps_results.all_states
    # add the acceptance ratios to the list
    acceptance_list.append(bps_results.trace[0].numpy())
    # final kernel results used to initialise next call of loop
    kernel_previous_results = bps_results.final_kernel_results
    # time lists
    time_list.append(bps_results.trace[1].numpy())
    proposed_time_list.append(bps_results.trace[2].numpy())
    # save these samples to file
    save_chain(bps_chain, out_dir, loop_iter)
    # get the final state of the chain from the previous loop iter
    init_state = [x[-1] for x in bps_chain]
  print('finished sampling')
  end_sampling_time = time.time()
  print('total sampling time = {}'.format(end_sampling_time - start_time))
  if len(acceptance_list) > 1:
    acceptance_ratio = np.concatenate(acceptance_list)
    result_time = np.concatenate(time_list)
    proposed_time = np.concatenate(proposed_time_list)
  else:
    acceptance_ratio = acceptance_list[0]
    result_time = time_list[0]
    proposed_time = proposed_time_list[0]
  np.save(os.path.join(out_dir, 'acceptance_ratio.npy'), acceptance_ratio)
  print('acceptance_ratio = {}'.format(acceptance_ratio))
  print(type(acceptance_ratio))
  print('num acceptance_ratio > 1 = {}'.format(np.sum(
      (acceptance_ratio > 1.0))))
  print(f'result time = {result_time}')
  print(f'proposed time = {proposed_time}')
  np.save(os.path.join(out_dir, 'proposed_times.npy'), proposed_time)
  np.save(os.path.join(out_dir, 'result_times.npy'), result_time)
  # plot the results if specified to
  if (plot_results or run_eval):
    print('plotting pred posterior and test accuracy')
    # if the output is one dimensional, will be regression task so plot using
    # regression methods
    if (data_dimension_dict['out_dim'] == 1):
      plot_pred_posterior(model, bps_chain, likelihood_str, num_results,
                          training_iter, test_ds, out_dir, 'bps')
    # otherwise will be classification task, so plot using those methods
    else:
      if data_dimension_dict['in_channels'] == 1:
        plt_dims = [
            data_dimension_dict['in_height'], data_dimension_dict['in_width']
        ]
      else:
        plt_dims = [
            data_dimension_dict['in_height'], data_dimension_dict['in_width'],
            data_dimension_dict['in_channels']
        ]
      eval_plot_image_iter_pred_posterior(model, out_dir, training_iter,
                                          test_ds, test_orig_ds,
                                          data_dimension_dict['out_dim'],
                                          data_dimension_dict['out_dim'],
                                          plot_results, plt_dims)


def pbps_iter_main(model,
                   ipp_sampler_str,
                   lambda_ref,
                   std_ref,
                   num_results,
                   num_burnin_steps,
                   out_dir,
                   num_loops,
                   bnn_neg_joint_log_prob,
                   map_initial_state,
                   training_iter,
                   test_ds,
                   test_orig_ds,
                   likelihood_str,
                   batch_size,
                   data_size,
                   data_dimension_dict,
                   plot_results=True,
                   run_eval=True,
                   num_steps_between_results=0):
  """main method for running BPS on model"""
  print('running bps')
  start_time = time.time()
  # finding the number of samples to perform for each iteration
  if ipp_sampler_str == 'adaptive':
    ipp_sampler = AdaptivePSBPSampler
  elif ipp_sampler_str == 'interpolation':
    ipp_sampler = InterpolationPSampler
  else:
    ipp_sampler = PSBPSampler
  print('map_initial_state bps = {}'.format(map_initial_state))
  num_params = np.sum([x.numpy().size for x in map_initial_state])
  print('num_params = {}'.format(num_params))
  # num_samples_per_loop, num_loops = find_num_iters_and_loops(
  # num_params, num_results)
  num_samples_per_loop = np.int32(num_results / num_loops)
  # create the kernel
  kernel = IterPBPSKernel(parent_target_log_prob_fn=bnn_neg_joint_log_prob,
                          store_parameters_in_results=True,
                          ipp_sampler=ipp_sampler,
                          batch_size=batch_size,
                          data_size=data_size,
                          lambda_ref=lambda_ref,
                          std_ref=std_ref)
  # convert the init state into a tensor first
  init_state = [tf.convert_to_tensor(x) for x in map_initial_state]
  # BURNIN PHASE
  # start sampling for burnin, and then discard these samples
  # we may not be able to fit all the burnin samples in a single loop,
  # so we will loop over a few times if we need to
  print('num_samples_per_loop = {}'.format(num_samples_per_loop))
  print('num_loops = {}'.format(num_loops))
  num_burnin_iters = np.ceil(num_burnin_steps / num_samples_per_loop).astype(
      np.int)
  # create the trace function
  trace_fn = lambda _, pkr: pkr.acceptance_ratio
  # run bootstrap here to get the initial state for burnin
  # this allows us to start sampling at each loop exactly where we left off
  kernel_previous_results = kernel.bootstrap_results(init_state)
  # now run the burnin phase
  # for burnin_iter in range(0, num_burnin_iters):
  bps_results = graph_hmc(num_results=1,
                          current_state=init_state,
                          kernel=kernel,
                          previous_kernel_results=kernel_previous_results,
                          return_final_kernel_results=True,
                          num_burnin_steps=num_burnin_steps,
                          trace_fn=trace_fn)
  # extract the chain and the final kernel results
  bps_chain = bps_results.all_states
  # final kernel results used to initialise next call of loop
  kernel_previous_results = bps_results.final_kernel_results
  # get the final state of the chain from the previous burnin iter
  init_state = [x[-1, ...] for x in bps_chain]
  end_warmup_time = time.time()
  print('time warmup = {}'.format(end_warmup_time - start_time))
  # SAMPLING PHASE
  # reset the initial state back to the map state
  init_state = [tf.convert_to_tensor(x) for x in map_initial_state]
  acceptance_list = []
  # now loop over the actual samples from (hopefully) the posterior
  for loop_iter in range(0, num_loops):
    print('loop iter = {}'.format(loop_iter))
    bps_results = graph_hmc(num_results=num_samples_per_loop,
                            current_state=init_state,
                            kernel=kernel,
                            previous_kernel_results=kernel_previous_results,
                            return_final_kernel_results=True,
                            num_steps_between_results=num_steps_between_results,
                            trace_fn=trace_fn)
    # extract the chain and the final kernel results
    bps_chain = bps_results.all_states
    # add the acceptance ratios to the list
    acceptance_list.append(bps_results.trace.numpy())
    # final kernel results used to initialise next call of loop
    kernel_previous_results = bps_results.final_kernel_results
    # save these samples to file
    save_chain(bps_chain, out_dir, loop_iter)
    # get the final state of the chain from the previous loop iter
    init_state = [x[-1] for x in bps_chain]
  print('finished sampling')
  end_sampling_time = time.time()
  print('total sampling time = {}'.format(end_sampling_time - start_time))
  if len(acceptance_list) > 1:
    acceptance_ratio = np.concatenate(acceptance_list)
  else:
    acceptance_ratio = acceptance_list[0]
  print('acceptance_ratio = {}'.format(acceptance_ratio))
  print(type(acceptance_ratio))
  print('num acceptance_ratio > 1 = {}'.format(np.sum(
      (acceptance_ratio > 1.0))))
  # plot the results if specified to
  if (plot_results or run_eval):
    print('plotting pred posterior and test accuracy')
    # if the output is one dimensional, will be regression task so plot using
    # regression methods
    if (data_dimension_dict['out_dim'] == 1):
      plot_pred_posterior(model, bps_chain, likelihood_str, num_results,
                          training_iter, test_ds, out_dir, 'bps')
    # otherwise will be classification task, so plot using those methods
    else:
      if data_dimension_dict['in_channels'] == 1:
        plt_dims = [
            data_dimension_dict['in_height'], data_dimension_dict['in_width']
        ]
      else:
        plt_dims = [
            data_dimension_dict['in_height'], data_dimension_dict['in_width'],
            data_dimension_dict['in_channels']
        ]
      eval_plot_image_iter_pred_posterior(model, out_dir, training_iter,
                                          test_ds, test_orig_ds,
                                          data_dimension_dict['out_dim'],
                                          data_dimension_dict['out_dim'],
                                          plot_results, plt_dims)


def iter_boomerang_neg_joint_log_prob(model, likelihood_fn, dataset_iter,
                                      preconditioner, mean):
  def _fn():
    print('calling iter')
    X, y = dataset_iter.next()
    return boomerang_bnn_neg_joint_log_prob_fn(model, likelihood_fn, X, y,
                                               preconditioner, mean)

  return _fn


@tf.function
def boomerang_warmup_one_step(state, kernel, prev_kernel_results):
  next_state, next_kernel_results = kernel.one_step(state, prev_kernel_results)
  return next_state, next_kernel_results


def boomerang_iter_main(model,
                        ipp_sampler_str,
                        lambda_ref,
                        std_ref,
                        num_results,
                        num_burnin_steps,
                        out_dir,
                        num_loops,
                        hessian_fn,
                        bnn_neg_joint_log_prob,
                        strategy,
                        likelihood_fn,
                        map_initial_state,
                        training_iter,
                        test_ds,
                        test_orig_ds,
                        likelihood_str,
                        batch_size,
                        data_size,
                        data_dimension_dict,
                        plot_results=True,
                        run_eval=True,
                        num_steps_between_results=0):
  """main method for running boomerang on model"""
  print('running boomerang')
  start_time = time.time()
  # finding the number of samples to perform for each iteration
  ipp_sampler = InterpolationBoomSampler
  print('map_initial_state bps = {}'.format(map_initial_state))
  num_params = np.sum([x.numpy().size for x in map_initial_state])
  print('num_params = {}'.format(num_params))
  # num_samples_per_loop, num_loops = find_num_iters_and_loops(
  # num_params, num_results)
  num_samples_per_loop = np.int32(num_results / num_loops)
  # convert the init state into a tensor first
  init_state = [tf.convert_to_tensor(x) for x in map_initial_state]
  hessian_diag = [tf.zeros_like(x) for x in map_initial_state]

  # hessian phase
  # find number of iterations needed to go over entire dataset
  num_hessian_loops = np.ceil(data_dimension_dict['dataset_size'] / batch_size).astype(np.int64)
  trace_fn = lambda _, pkr: (pkr.acceptance_ratio, pkr.time, pkr.proposed_time, pkr.velocity)
  for i in range(0, num_hessian_loops):
    print(f'hessian batch idx = {i}')
    hessian_diag_batch = hessian_fn()
    # add this batch hessian to the sum over all data points next
    hessian_diag = [h + b for h, b in zip(hessian_diag, hessian_diag_batch)]
  var_preconditioner = [std_ref / tf.math.abs(h) for h in hessian_diag]
  # make sure no nans or infs were introduced due to a divide by zero
  # if they were, then replace them with the maximum value seen in that layer
  for i in range(0, len(var_preconditioner)):
    # find the maximum non inf number
    v = var_preconditioner[i]
    finite_idx = tf.math.is_finite(v)
    inf_idx = tf.math.is_inf(v)
    print(type(inf_idx))
    max_non_inf = tf.reduce_max(v[finite_idx])
    print(f'max non inf = {max_non_inf}')
    # print(f'num_infs to replace = {tf.reduce_sum(inf_idx)}')
    print(f'num_infs to replace = {tf.size(inf_idx)}')
    print(f'max value before = {tf.reduce_max(var_preconditioner[i])}')
    print(f'min value before = {tf.reduce_min(var_preconditioner[i])}')
    # now replace any infs with this max
    var_preconditioner[i] = tf.where(
      inf_idx,
      tf.ones_like(var_preconditioner[i]) * max_non_inf,
      var_preconditioner[i])
    print(f'max value after = {tf.reduce_max(var_preconditioner[i])}')
    print(f'min value after = {tf.reduce_min(var_preconditioner[i])}')
  print(var_preconditioner)
  print('var condintioner')
  mean = init_state
  # now want to create a boomerang kernel
  boomerang_target = iter_grad_boomerang_fn(model, likelihood_fn, training_iter,
                                            data_size, batch_size,
                                            var_preconditioner, mean, strategy)
  acceptance_list = []
  time_list = []
  proposed_time_list = []
  boomerang_kernel = BoomerangIterKernel(boomerang_target,
                                         var_preconditioner,
                                         mean,
                                         store_parameters_in_results=True,
                                         ipp_sampler=ipp_sampler,
                                         batch_size=batch_size,
                                         data_size=data_size,
                                         lambda_ref=lambda_ref,
                                         std_ref=std_ref)
  # SAMPLING PHASE
  # now loop over the actual samples from (hopefully) the posterior
  acceptance_list = []
  velocity_list = []
  for loop_iter in range(0, num_loops):
    print('loop iter = {}'.format(loop_iter))
    boomerang_results = graph_hmc(
        num_results=num_samples_per_loop,
        current_state=init_state,
        kernel=boomerang_kernel,
        num_steps_between_results=num_steps_between_results,
        return_final_kernel_results=True,
        trace_fn=trace_fn)
    # extract the chain and the final kernel results
    boomerang_chain = boomerang_results.all_states
    # add the acceptance ratios to the list
    print(boomerang_results.trace[0])
    print(type(boomerang_results.trace[0]))
    print(len(boomerang_results.trace[0]))
    acceptance_list.append(boomerang_results.trace[0].numpy())
    time_list.append(boomerang_results.trace[1].numpy())
    proposed_time_list.append(boomerang_results.trace[2].numpy())

    # final kernel results used to initialise next call of loop
    kernel_previous_results = boomerang_results.final_kernel_results
    # save these samples to file
    save_chain(boomerang_chain, out_dir, loop_iter)
    # get the final state of the chain from the previous loop iter
    init_state = [x[-1] for x in boomerang_chain]
  print('finished sampling')
  end_sampling_time = time.time()
  print('total sampling time = {}'.format(end_sampling_time - start_time))
  if len(acceptance_list) > 1:
    acceptance_ratio = np.concatenate(acceptance_list)
    result_time = np.concatenate(time_list)
    proposed_time = np.concatenate(proposed_time_list)
  else:
    acceptance_ratio = acceptance_list[0]
    result_time = time_list[0]
    proposed_time = proposed_time_list[0]
  print('acceptance_ratio = {}'.format(acceptance_ratio))
  print(type(acceptance_ratio))
  print('num acceptance_ratio > 1 = {}'.format(np.sum(
      (acceptance_ratio > 1.0))))
  print(f'result time = {result_time}')
  print(f'proposed time = {proposed_time}')
  np.save(os.path.join(out_dir, 'acceptance_ratio.npy'), acceptance_ratio)
  np.save(os.path.join(out_dir, 'proposed_times.npy'), proposed_time)
  np.save(os.path.join(out_dir, 'result_times.npy'), result_time)
  if (plot_results or run_eval):
    print('plotting pred posterior and test accuracy')
    # if the output is one dimensional, will be regression task so plot using
    # regression methods
    if (data_dimension_dict['out_dim'] == 1):
      plot_pred_posterior(model, boomerang_chain, likelihood_str, num_results,
                          training_iter, test_ds, out_dir, 'boomerang')
    # otherwise will be classification task, so plot using those methods
    else:
      if data_dimension_dict['in_channels'] == 1:
        plt_dims = [
            data_dimension_dict['in_height'], data_dimension_dict['in_width']
        ]
      else:
        plt_dims = [
            data_dimension_dict['in_height'], data_dimension_dict['in_width'],
            data_dimension_dict['in_channels']
        ]
      eval_plot_image_iter_pred_posterior(model, out_dir, training_iter,
                                          test_ds, test_orig_ds,
                                          data_dimension_dict['out_dim'],
                                          data_dimension_dict['out_dim'],
                                          plot_results, plt_dims)


def save_chain(chain, out_dir, loop_iter=0):
  chain_path = os.path.join(out_dir, 'chain_{}.pkl'.format(loop_iter))
  # check the out directory exists, and if it doesn't than make it
  utils.check_or_mkdir(out_dir)
  # now save it
  with open(chain_path, 'wb') as filehandler:
    pickle.dump(chain, filehandler)


def save_map_weights(map_weights, out_dir):
  map_path = os.path.join(out_dir, 'map_weights.pkl')
  print(map_path)
  # check the out directory exists, and if it doesn't than make it
  utils.check_or_mkdir(out_dir)
  var_values = [x.read_value() for x in map_weights]
  # now save it
  with open(map_path, 'wb') as filehandler:
    #var_names = [x.name for x in map_weights]
    pickle.dump(var_values, filehandler)


# @tf.function
def graph_bps(num_results,
              num_burnin_steps,
              current_state,
              kernel,
              num_steps_between_results=10,
              trace_fn=None):
  # perform burnin phase
  bps_burnin = tfp.mcmc.sample_chain(num_results=1,
                                     current_state=current_state,
                                     kernel=kernel,
                                     num_steps_between_results=num_burnin_steps,
                                     trace_fn=None)
  # get the final state of the chain from the burnin phase
  keras.backend.clear_session()
  init_state = [x[-1] for x in bps_burnin]
  bps_chain = tfp.mcmc.sample_chain(
      num_results=num_results,
      current_state=init_state,
      kernel=kernel,
      num_steps_between_results=num_steps_between_results,
      trace_fn=None)

  return bps_chain


def plot_pred_posterior(model, chain, likelihood_str, num_results,
                        training_iter, test_ds, out_dir, name):
  # get the raw data from the tf.Dataframes
  print(f'dataset type = {type(training_iter)}')
  if isinstance(training_iter, tf.data.Dataset):
    # if is hmc won't be an iterator
    X_train, y_train = next(iter(training_iter))
  else:
    X_train, y_train = next(training_iter)
  X_test, y_test = next(iter(test_ds))
  # make them all numpy arrays
  X_train = X_train.numpy()
  y_train = y_train.numpy()
  X_test = X_test.numpy()
  y_test = y_test.numpy()
  if likelihood_str == 'bernoulli':
    plot_logistic_pred_posterior(model, chain, num_results, X_train, y_train,
                                 X_test, y_test, out_dir, name)
  elif likelihood_str == 'categorical':
    # get the plot dims from the model
    if (X_test.shape[-1] == 1):
      # plot dimensions will exclude the final channel as a supplied dimension
      # as having a dimension of [height, width, 1] doesn't agree with
      # matplotlibs imshow method
      plt_dims = X_test.shape[1:3]
    else:
      # have the dimensions be that of the full image
      # (so include the channel index, but remove the sample index)
      plt_dims = X_test.shape[1:]
    plot_image_pred_posterior(model,
                              chain,
                              num_results,
                              X_train,
                              y_train,
                              X_test,
                              y_test,
                              name,
                              plt_dims=plt_dims)
  elif likelihood_str == 'normal':
    plot_regression_pred_posterior(model, chain, num_results, X_train, y_train,
                                   X_test, y_test, out_dir, name)
  else:
    raise ValueError('Incorrect likelihood specified for plotting output')


def plot_regression_pred_posterior(model, chain, num_results, X_train, y_train,
                                   X_test, y_test, out_dir, name):
  num_returned_samples = len(chain[0])
  print(f'num returned samples {num_returned_samples}')
  # perform prediction for each iteration
  num_samples = np.min([1000, num_returned_samples])
  sample_idx = np.arange(0, num_samples - 1, 10)
  num_plot = sample_idx.size
  pred = np.zeros([num_plot, y_test.size])
  print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
  plt.figure()
  plt.scatter(X_train, y_train, color='b', alpha=0.15)
  pred_idx = 0
  # need to specify pred forward pass method
  # if is hmc, need to use the set_model_params_hmc method
  if name == 'hmc':
    pred_fn = pred_forward_pass_hmc
  else:
    pred_fn = pred_forward_pass
  for mcmc_idx in sample_idx:
    param_list = [x[mcmc_idx, ...] for x in chain]
    pred[pred_idx, :] = pred_fn(model, param_list,
                                X_test.astype(
                                              np.float32)).numpy().ravel().astype(np.float64)
    plt.plot(X_test, pred[pred_idx, :], alpha=0.05, color='k')
    pred_idx += 1
  plt.axis('off')
  print('saving images to {}'.format(out_dir))
  plt.savefig(os.path.join(out_dir, 'pred.png'))
  plt.savefig(os.path.join(out_dir, 'pred.pdf'), bbox_inches='tight')
  pred_mean = np.mean(pred, axis=0)
  # pred_mean = pred_mean / n_samples
  pred_cov = np.zeros([y_test.size, y_test.size])
  for i in range(pred.shape[0]):
    pred_cov += pred[i, :] @ np.transpose(pred[i, :])
  num_samples = pred.shape[0]
  print('num_samples, ', num_samples)
  pred_cov = 0.01**2.0 + (pred_cov / num_samples) - \
    pred_mean @ np.transpose(pred_mean)
  print(pred_cov.shape)
  diag = np.diag(pred_cov)
  std = np.sqrt(diag)
  std = np.std(pred, axis=0)
  # reshaping pred_mean to be a 1d array for when we plot it
  pred_mean = pred_mean.squeeze()
  #as an example, lets plot the predictive posterior distribution contours for
  #some similar classes
  plt.figure()
  #plt.plot(x_test, pred_eval, 'r', label='Sample')
  plt.plot(X_test, y_test, 'b', label='True', alpha=0.2, linewidth=0.5)
  plt.plot(X_test, pred_mean, 'm', label='mean')
  plt.gca().fill_between(np.squeeze(X_test),
                         pred_mean - 2 * std,
                         pred_mean + 2 * std,
                         color="#acc1e3")
  plt.gca().fill_between(np.squeeze(X_test),
                         pred_mean - 1 * std,
                         pred_mean + 1 * std,
                         color="#cbd9f0")
  plt.scatter(X_train, y_train, marker='o', alpha=0.15,
              s=10, c="#7f7f7f", label='Training Samples')
  plt.axis('off')
  plt.savefig(os.path.join(out_dir, 'pred_test.png'))
  plt.savefig(os.path.join(out_dir, 'pred_test.pdf'), bbox_inches='tight')
  plt.close()



def test_plot_regression_pred_posterior(model, chain, velocity, mean, time,
                                        num_results, X_train, y_train, X_test,
                                        y_test, out_dir, name):
  num_returned_samples = len(chain[0])
  # perform prediction for each iteration
  num_samples = np.min([1000, num_returned_samples])
  sample_idx = np.arange(0, num_samples - 1, 10)
  num_plot = sample_idx.size
  pred = np.zeros([num_plot, y_test.size])
  print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
  plt.figure()
  plt.scatter(X_train, y_train, color='b', alpha=0.15)
  pred_idx = 0

  for mcmc_idx in sample_idx:
    # find the mid point
    t = tf.cast(time[mcmc_idx] / 2, tf.float32)
    param_list = [x[mcmc_idx, ...] for x in chain]
    v_list = [x[mcmc_idx, ...] for x in velocity]
    param_dynamics = [
        m + (x - m) * tf.math.cos(t) + v * tf.math.sin(t)
        for m, x, v in zip(mean, param_list, v_list)
    ]
    for i in range(len(param_list)):
      print(
          f"p shape =  {param_list[i].shape}, d =  {param_dynamics[i].shape}, v = {velocity[i].shape}, m = {mean[i].shape}"
      )
    pred[pred_idx, :] = pred_forward_pass(model, param_dynamics,
                                          X_test.astype(
                                              np.float32)).numpy().ravel()
    plt.plot(X_test, pred[pred_idx, :], alpha=0.05, color='k')
    pred_idx += 1
  plt.axis('off')
  print('saving images to {}'.format(out_dir))
  plt.savefig(os.path.join(out_dir, 'pred.png'))
  plt.savefig(os.path.join(out_dir, 'pred.pdf'), bbox_inches='tight')


def plot_logistic_pred_posterior(model, chain, num_results, X_train, y_train,
                                 X_test, y_test, out_dir, name):
  if name == 'hmc':
    pred_fn = pred_forward_pass_hmc
  else:
    pred_fn = pred_forward_pass

  print('plotting logistic')
  idx = 0
  num_plot = np.min([num_results, 10])
  pred_array = np.zeros([num_plot, y_test.size])
  if name == 'hmc':
    pred_fn = pred_forward_pass_hmc

  for mcmc_idx in range(0, num_plot):
    param_list = [x[mcmc_idx, ...] for x in chain]
    pred_array[mcmc_idx, :] = tf.nn.sigmoid(
        pred_fn(model, param_list,
                          X_test.astype(np.float32))).numpy().ravel()
    idx += 1
  pred_mean = np.mean(pred_array, axis=0)
  pred_mean_classification = np.round(pred_mean).astype(np.int64)
  X_test_np = X_test 
  plt.scatter(X_test_np[pred_mean < 0.5, 0],
              X_test_np[pred_mean < 0.5, 1],
              color='b')
  plt.scatter(X_test_np[pred_mean >= 0.5, 0],
              X_test_np[pred_mean >= 0.5, 1],
              color='r')
  # plt.scatter(X_test_np[pred_mean_classification==0, 0],
  #             X_test_np[pred_mean_classification==0, 1], color='b')
  # plt.scatter(X_test_np[pred_mean_classification==1, 0],
  #             X_test_np[pred_mean_classification==1, 1], color='r')
  plt.savefig(os.path.join(out_dir, 'pred_logistic_' + name + '.png'))
  # w = weights_chain[0][:, 0, 0].numpy()
  #az.plot_trace(w.reshape([1, num_results]))
  #plt.savefig(os.path.join(out_dir, 'trace_logistic_' + name + '.png'))
  # create a grid to iterate over
  # using method from Thomas Wiecki's blog
  grid = np.mgrid[-1.5:1.5:100j, -1.5:1.5:100j]
  grid_2d = grid.reshape(2, -1).T
  print('grid_2d shape = {}'.format(grid_2d.shape))
  pred_grid = np.zeros([num_results, grid_2d.shape[0]])
  idx = 0
  for i in range(0, num_results):
    param_list = [x[i, ...] for x in chain]
    pred_grid[idx, :] = tf.keras.activations.sigmoid(
        pred_fn(model, param_list, grid_2d).numpy().ravel())
    idx += 1

  grid_mean = np.mean(pred_grid, axis=0)
  print('grid_mean shape = {}'.format(grid_mean.shape))
  cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
  fig, ax = plt.subplots()
  contour = ax.contourf(grid[0],
                        grid[1],
                        grid_mean.reshape(100, 100),
                        cmap=cmap)
  ax.scatter(X_test_np[pred_mean_classification == 0, 0],
             X_test_np[pred_mean_classification == 0, 1],
             color='b')
  ax.scatter(X_test_np[pred_mean_classification == 1, 0],
             X_test_np[pred_mean_classification == 1, 1],
             color='r')
  cbar = plt.colorbar(contour, ax=ax)
  _ = ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), xlabel='X', ylabel='Y')
  cbar.ax.set_ylabel('Posterior predictive mean probability of class label = 0')
  plt.savefig(os.path.join(out_dir, 'grid_mean_logistic_' + name + '.png'))
  plt.savefig(os.path.join(out_dir, 'grid_mean_logistic_' + name + '.pdf'))
  grid_std = np.std(pred_grid, axis=0)
  cmap = sns.cubehelix_palette(light=1, as_cmap=True)
  fig, ax = plt.subplots()
  contour = ax.contourf(grid[0], grid[1], grid_std.reshape(100, 100), cmap=cmap)
  ax.scatter(X_test_np[pred_mean_classification == 0, 0],
             X_test_np[pred_mean_classification == 0, 1],
             color='b')
  ax.scatter(X_test_np[pred_mean_classification == 1, 0],
             X_test_np[pred_mean_classification == 1, 1],
             color='r')
  cbar = plt.colorbar(contour, ax=ax)
  _ = ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), xlabel='X', ylabel='Y')
  print('saving fig')
  cbar.ax.set_ylabel('Uncertainty (posterior predictive standard deviation)')
  plt.savefig(os.path.join(out_dir, 'grid_var_logistic_' + name + '.png'))
  plt.savefig(os.path.join(out_dir, 'grid_var_logistic_' + name + '.pdf'))
  pred_grid = tf.keras.activations.sigmoid(
      pred_fn(model, param_list, grid_2d).numpy().ravel())
  fig, ax = plt.subplots()
  contour = ax.contourf(grid[0],
                        grid[1],
                        grid_mean.reshape(100, 100),
                        cmap=cmap)
  ax.scatter(X_test_np[pred_mean_classification == 0, 0],
             X_test_np[pred_mean_classification == 0, 1],
             color='b')
  ax.scatter(X_test_np[pred_mean_classification == 1, 0],
             X_test_np[pred_mean_classification == 1, 1],
             color='r')
  cbar = plt.colorbar(contour, ax=ax)
  _ = ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), xlabel='X', ylabel='Y')
  cbar.ax.set_ylabel('MC Approx. p(y=1|x)')
  plt.savefig(os.path.join(out_dir, 'mc_approx_logistic_' + name + '.png'))
  plt.savefig(os.path.join(out_dir, 'mc_approx_logistic_' + name + '.pdf'))

  fig, ax = plt.subplots()
  contour = ax.contour(grid[0], grid[1], grid_mean.reshape(100, 100), levels=6)
  ax.scatter(X_test_np[pred_mean_classification == 0, 0],
             X_test_np[pred_mean_classification == 0, 1],
             color='b')
  ax.scatter(X_test_np[pred_mean_classification == 1, 0],
             X_test_np[pred_mean_classification == 1, 1],
             color='r')
  _ = ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), xlabel='X', ylabel='Y')
  cbar.ax.set_ylabel('MC Approx. p(y=1|x)')
  plt.savefig(
      os.path.join(out_dir, 'mc_approx_contour_logistic_' + name + '.png'))
  plt.savefig(
      os.path.join(out_dir, 'mc_approx_contour_logistic_' + name + '.pdf'))
  print('pred_mean = {}'.format(pred_mean))

def pred_mean(model,
              chain,
              dataset,
              num_samples=100,
              batch_size=100,
              num_classes=10,
              final_activation=tf.keras.activations.softmax):
  """finds the mean in the predictive posterior"""
  pred_eval_array = pred_eval_fn(model, chain, dataset, num_samples, batch_size,
                                 num_classes, final_activation)
  pred_mean = np.mean(pred_eval_array, axis=0)
  classification = np.argmax(pred_mean, axis=1)
  print('classification shape = {}'.format(classification.shape))
  return classification


def pred_eval_fn(model,
                 chain,
                 dataset,
                 num_samples=None,
                 num_classes=10,
                 final_activation=tf.keras.activations.softmax,
                 name='pdmp'):
  """finds the mean in the predictive posterior"""
  if name == 'hmc':
    pred_fn = pred_forward_pass_hmc
  else:
    pred_fn = pred_forward_pass
  # get the batch size
  batch_size = next(iter(dataset))[0].shape[0]
  # if num samples isnt set, make it the number of samples in the chain
  num_samples = chain[0].shape[0] if num_samples is None else num_samples
  pred_eval_array = np.zeros([num_samples, batch_size, num_classes])
  print('pred_eval_array.shape = {}'.format(pred_eval_array.shape))
  print(num_classes)
  # create an iterator for the dataset
  pred_epoch_list = []
  # also creating a list of the labels so I can use them later to compute the
  # accuracy
  label_list = []
  for images, labels in dataset.as_numpy_iterator():
    pred_samples_list = []
    # want to get the predictions of a full epoch for the
    for mcmc_idx in range(0, num_samples):
      param_list = [x[mcmc_idx, ...] for x in chain]
      # now sample over the posterior samples of interest
      # pred_eval_array[mcmc_idx, image_idx, ...] = final_activation(
      # pred_forward_pass(model, param_list, image)).numpy()
      pred = final_activation(pred_fn(model, param_list,
                                      images)).numpy()
      pred_samples_list.append(pred)
    # let's add all the predicts from each sample for this batch of images to a single array now
    # using stack to create a new index for the samples
    # stack the predictions for each batch over each set of weights to combine them
    # into one array
    pred_batch_samples = np.stack(pred_samples_list, axis=0)
    print(f"pred_batch_samples.shape {pred_batch_samples.shape}")
    pred_epoch_list.append(pred_batch_samples)
    # also save the labels
    label_list.append(labels)
  print([x.shape for x in pred_epoch_list])
  print(len(pred_epoch_list))
  # concatenate the pred list over axis 1, which is the image sample index
  pred_posterior = np.concatenate(pred_epoch_list, axis=1)
  print(f"pred_posterior.shape = {pred_posterior.shape}")
  return pred_posterior, np.concatenate(label_list, axis=0)


def pred_eval_map_fn(model,
                     param_list,
                     dataset,
                     batch_size=100,
                     num_classes=10,
                     final_activation=tf.keras.activations.softmax):
  """finds the mean in the predictive posterior"""
  print(model.summary())
  # get a set of the images to perform prediction on
  # setting image index lower value to be zero
  pred_list = []
  # also creating a list of the labels so I can use them later to compute the
  # accuracy
  label_list = []
  for images, labels in dataset.as_numpy_iterator():
    pred = final_activation(pred_forward_pass(model, param_list,
                                              images)).numpy()
    pred_list.append(pred)
    label_list.append(labels)
  pred_final = np.concatenate(pred_list, axis=0)
  return pred_final, np.concatenate(label_list, axis=0)


def plot_image_pred_posterior(model,
                              chain,
                              num_results,
                              X_train,
                              y_train,
                              X_test,
                              y_test,
                              X_test_orig,
                              save_dir,
                              plt_dims=[28, 28]):
  """Plot misclassified with credible intervals"""
  classification = pred_mean(model,
                             chain,
                             X_test,
                             y_test,
                             final_activation=tf.keras.activations.softmax)
  #display._display_accuracy(model, X_test, y_test, 'Testing Data')
  num_classes = 10
  # create a figure
  plt.figure()
  fig, axs = plt.subplots(2, 1)
  # iterate over all classes
  correct_preds = np.argmax(y_test, axis=1)
  for label_i in range(0, num_classes):
    # check to see if a directory exists. If it doesn't, create it.
    utils.check_or_mkdir(os.path.join(save_dir, str(label_i)))
    locs = np.where(
        np.logical_and(classification != correct_preds, y_test[:,
                                                               label_i] == 1))
    pred_eval = np.zeros([num_results, locs[0].size, num_classes])
    images = X_test[locs[0], ...]
    plot_images = X_test_orig[locs[0], ...]
    weights_chain = chain[::2]
    biases_chain = chain[1::2]
    idx = 0
    for i in range(num_results - pred_eval.shape[0], num_results):
      weights_list = [x[i, ...] for x in weights_chain]
      biases_list = [x[i, ...] for x in biases_chain]
      pred_eval[idx, ...] = tf.keras.activations.softmax(
          pred_forward_pass(model, weights_list, biases_list, images))
      idx += 1
    # now get the mean and credible intervals for these images and plot them
    # creating a counter variable for each individual misclassified image
    count = 0
    x_tick = np.arange(0, 10)
    for im_idx in range(0, pred_eval.shape[1]):
      # approximate the mean and credible intervals
      cred_ints = display.mc_credible_interval(
          pred_eval[:, im_idx, :].reshape([-1, num_classes]),
          np.array([0.025, 0.975]))
      pred_mean_array = np.mean(pred_eval[:, im_idx, :], axis=0)
      # PLOTTING
      # formatting the credible intervals into what is needed to be plotted
      # with pyplot.errorbar()
      cred_plot = np.array([
          pred_mean_array - cred_ints[0, :], cred_ints[1, :] - pred_mean_array
      ])
      # reshape it to correct dims
      cred_plot = cred_plot.reshape(2, num_classes)
      #now lets plot it and save it
      cmap = 'gray' if len(plt_dims) == 2 else None

      plt.subplot(2, 1, 1)
      print(plt_dims)
      plt.imshow(plot_images[im_idx].reshape(plt_dims), cmap=cmap)
      plt.axis('off')
      plt.subplot(2, 1, 2)
      plt.errorbar(np.linspace(0, pred_mean_array.size - 1,
                               pred_mean_array.size),
                   pred_mean_array.ravel(),
                   yerr=cred_plot,
                   fmt='o')
      plt.xlim(-1, num_classes)
      plt.ylim(-0.1, 1.1)
      plt.xticks(range(num_classes), x_tick, size='large', rotation='vertical')
      plt.xlabel("class", fontsize=40)
      plt.ylabel("Predicted Probability\nwith 95% CI", size=40)
      plt.savefig(os.path.join(save_dir, str(label_i),
                              "{}_{}.png".format(label_i, count)))
      plt.savefig(os.path.join(save_dir, str(label_i),
                               "{}_{}.eps".format(label_i, count)),
                  format='eps',
                  bbox_inches="tight")
      plt.clf()
      #increment counter
      count += 1


def eval_iter_chain(model: keras.Model,
                    save_dir: str,
                    dataset: tf.data.Dataset,
                    num_classes: int = 10,
                    data_desc_str: str = 'test',
                    final_activation: callable = tf.keras.activations.softmax,
                    name='pdmp'):
  """performs eval on data supplied.

  Is designed so that can call a single eval function and just
  change the data that is being evaluated

  Parameters
  ----------
  model : tf.keras.model
      model to be used
  save_dir : str
      where to save the output files
  dataset : tf.data.Dataset
      dataset we are evaluating over
  final_acttivation : tf.keras.activation
      activation to be used on final layer. Depends on the likelihood fn.
      ie. categorical likelihood = softmax, Gaussian = Linear, Bernoulli = sigmoid or probit

  Returns
  -------
    pred_posterior : array(tf.Tensor)
       samples from pred posterior for dataset
    pred_mean : array(tf.Tensor)
        mean of the predictive posterior
    classification : array(tf.Tensor)
        argmax of pred_mean
  """
  chain_files = glob(os.path.join(save_dir, 'chain_*.pkl'))
  # create a list to store all samples from predictive posterior
  pred_list = []
  idx = 0
  for chain_file in chain_files:
    print('idx = {}'.format(idx))
    idx += 1
    with open(chain_file, 'rb') as f:
      chain = pickle.load(f)
    preds, labels = pred_eval_fn(model,
                                 chain,
                                 dataset,
                                 num_classes=num_classes,
                                 final_activation=final_activation,
                                 name=name)
    pred_list.append(preds)
  print("hereee", [x.shape for x in pred_list])
  pred_posterior = np.concatenate(pred_list, axis=0)
  # now delete the pred list to save memory
  del (pred_list)
  del (preds)
  del (chain)
  print("pred_posterior.shape", pred_posterior.shape)
  # now perform classification based on the mean
  pred_mean_array = np.mean(pred_posterior, axis=0)
  # lets try the posterior median as well
  pred_median_array = np.median(pred_posterior, axis=0)
  # now get the classification performance
  print('pred_mean shape = {}, y argmax shape = {}'.format(
      pred_mean_array.shape,
      np.argmax(labels, axis=1).shape))
  classification = np.argmax(pred_mean_array, axis=1)
  classification_median = np.argmax(pred_median_array, axis=1)
  accuracy = accuracy_score(np.argmax(labels, axis=1),
                            classification,
                            normalize=True)
  print('{} Accuracy from quadratic loss = {}'.format(data_desc_str, accuracy),
        flush=True)
  accuracy_median = accuracy_score(np.argmax(labels, axis=1),
                                   classification_median,
                                   normalize=True)
  print('{} Accuracy from L1 loss = {}'.format(data_desc_str, accuracy_median),
        flush=True)
  return pred_posterior, pred_mean_array, classification, labels


def eval_plot_image_iter_pred_posterior(
    model,
    save_dir,
    training_iter,
    test_ds,
    test_orig_ds,
    num_classes,
    plot_results,
    plt_dims=[28, 28],
    final_activation=tf.keras.activations.softmax,
    name='pdmp'):
  """performs eval, and will plot pred posterior if needed"""
  print('numm classes = {}'.format(num_classes))
  (test_pred_posterior, test_pred_mean_array, test_classification,
   label_array) = eval_iter_chain(model,
                                  save_dir,
                                  test_ds,
                                  num_classes=num_classes,
                                  data_desc_str='test',
                                  name=name)
  # lets save the predictive posterior to file and also the mean classification as well
  np.save(os.path.join(save_dir, 'test_pred_posterior.npy'),
          test_pred_posterior)
  np.save(os.path.join(save_dir, 'test_pred_posterior_mean.npy'),
          test_pred_mean_array)
  # now will plot if we asked nicely
  test_classification = np.argmax(test_pred_mean_array, axis=1)
  if plot_results:
    label_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
    plot_image_iter_pred_posterior(save_dir, plt_dims, label_dict, test_orig_ds, label_array,
                                   test_classification, test_pred_posterior,
                                   test_pred_mean_array)


def plot_image_iter_pred_posterior(save_dir: str, plt_dims: list[int],
                                   class_names: dict, orig_ds: tf.data.Dataset,
                                   label_array: np.array,
                                   classification: np.array,
                                   pred_posterior: np.array,
                                   pred_mean_array: np.array):
  """Plot image with pred. posterior credible intervals.

  Handles models that were run in an 'iter' state, as
  all the samples would not fit in memory so need to
  iterate over them in the save directory to compute expectations.

  Parameters
  ----------
  save_dir : str
      where to save images
  plt_dims : list[int]
      dimensions for plotting images.
  class_names : dict
      dict with class names
  label_array : np.array
      array with the true class labels
  orig_ds : tf.data.Dataset
      original data set with no preprocessing or augmentation
  classification : np.array
      predictions made, (argmax(pred_mean))
  pred_posterior : np.array
      samples from the predictive posterior
  pred_mean_array : np.array
      predictive mean (mean(pred_posterior))

  Returns
  -------
  NA
  """
  # let's get the data from the dataset
  X_test = []
  y_test = []
  for images, labels in orig_ds.as_numpy_iterator():
    X_test.append(images)
    y_test.append(labels)

  # get the plotting dims
  plt_dims = X_test[0].shape[1::]
  # drop the 1 channel dim if grayscale
  if plt_dims[-1] == 1:
    plt_dims = plt_dims[0:-1]
  print('plot_dims', plt_dims)
  # now concatenate them
  X_test = np.concatenate(X_test, axis=0)
  y_test = np.concatenate(y_test, axis=0)
  print(X_test.shape)
  print(y_test.shape)
  print("plotting misclassified samples")
  num_classes = 10
  # create a figure
  plt.figure()
  # iterate over all classes
  if len(y_test.shape) == 1:
    correct_labels = y_test
    y_test = tf.keras.utils.to_categorical(y_test)
  else:
    correct_labels = np.argmax(y_test, axis=1)
  for label_i in range(0, num_classes):
    # check to see if a directory exists. If it doesn't, create it.
    utils.check_or_mkdir(os.path.join(save_dir, str(label_i)))
    locs = np.where(
        np.logical_and(classification != correct_labels, y_test[:,
                                                                label_i] == 1))
    pred_posterior_misclassified = pred_posterior[:, locs[0], ...]
    pred_mean_misclassified = pred_mean_array[locs[0], ...]
    plt_images = X_test[locs[0], ...]
    # now get the mean and credible intervals for these images and plot them
    # creating a counter variable for each individual misclassified image
    count = 0
    x_tick = np.arange(0, 10)
    for im_idx in range(0, plt_images.shape[0]):
      # approximate the mean and credible intervals
      cred_ints = display.mc_credible_interval(
          pred_posterior_misclassified[:, im_idx, :].reshape([-1, num_classes]),
          np.array([0.025, 0.975]))
      pred_mean_im = pred_mean_misclassified[im_idx, :]
      # PLOTTING
      # formatting the credible intervals into what is needed to be plotted
      # with pyplot.errorbar()
      # ensure there are no negative values
      # can happen for super tiny preds with noise due to interpolation
      # used in mc_credible_interval
      cred_plot = np.array(
          [np.maximum(0, pred_mean_im - cred_ints[0, :]),
           np.maximum(0, cred_ints[1, :] - pred_mean_im)])
      # reshape it to correct dims
      cred_plot = cred_plot.reshape(2, num_classes)
      cmap = 'gray' if len(plt_dims) == 2 else None
      #now lets plot it and save it
      plt_image = plt_images[im_idx, ...]
      print('image range = {}, {}, plt_dims = {}'.format(
          np.min(plt_image), np.max(plt_image), plt_dims))
      # if the image data is in integer range, divide by 255 to normalise
      # valid float range of [0.0, 1.0]
      if (np.max(plt_image) > 1.0):
        plt_image = plt_image / 255.0
      plt.subplot(2, 1, 1)
      plt.imshow(plt_image.reshape(plt_dims), cmap=cmap)
      plt.axis('off')
      plt.subplot(2, 1, 2)
      plt.errorbar(np.linspace(0, pred_mean_im.size - 1, pred_mean_im.size),
                   pred_mean_im.ravel(),
                   yerr=cred_plot,
                   fmt='o')
      plt.xlim(-1, num_classes)
      plt.ylim(-0.1, 1.1)
      plt.xticks(range(num_classes), x_tick, size='large', rotation='vertical')
      plt.xlabel("class", fontsize=16)
      plt.ylabel("Predicted Probability\nwith 95% CI", fontsize=16)
      #plt.savefig(os.path.join(save_dir, str(label_i),
      #                         "{}_{}.png".format(label_i, count)))
      plt.savefig(os.path.join(save_dir, str(label_i),
                               "{}_{}.eps".format(label_i, count)),
                  format='eps',
                  bbox_inches="tight")
      plt.savefig(os.path.join(save_dir, str(label_i),
                               "{}_{}.png".format(label_i, count)),
                  format='png',
                  bbox_inches="tight")
      plt.clf()
      # now make a separate plot of just the image and the
      # outputs separately
      plt.imshow(plt_image.reshape(plt_dims), cmap=cmap)
      plt.axis('off')
      plt.savefig(os.path.join(save_dir, str(label_i),
                               "{}_{}_image.png".format(label_i, count)),
                  format='png',
                  bbox_inches="tight")
      plt.clf()
      plt.figure(figsize=(8, 4))
      plt.errorbar(np.linspace(0, pred_mean_im.size - 1, pred_mean_im.size),
                   pred_mean_im.ravel(),
                   yerr=cred_plot,
                   fmt='o')
      plt.xlim(-1, num_classes)
      plt.ylim(-0.1, 1.1)
      plt.xticks(range(num_classes), x_tick, size='large', rotation='vertical')
      plt.xlabel("class", fontsize=16)
      # plt.ylabel("Predicted Probability\nwith 95% CI")
      plt.savefig(os.path.join(save_dir, str(label_i),
                               "{}_{}_pred.pdf".format(label_i, count)),
                  format='pdf',
      bbox_inches="tight")

      # increment counter
      count += 1


def create_entropy_hist(model,
                        save_dir,
                        test_ds,
                        data_name,
                        num_classes,
                        create_reliability=True,
                        final_activation=tf.keras.activations.softmax):
  """Plot entropy for the predicted output

  Handles models that were run in an 'iter' state, as
  all the samples would not fit in memory so need to
  iterate over them in the save directory to compute expectations.
  """
  # get a list of all the chain files
  (test_pred_posterior, test_pred_mean_array, test_classification,
   label_array) = eval_iter_chain(model,
                                  save_dir,
                                  test_ds,
                                  num_classes=num_classes,
                                  data_desc_str='test')
  np.save(os.path.join(save_dir, f'{data_name}_test_pred_posterior.npy'),
          test_pred_posterior)
  np.save(os.path.join(save_dir, f'{data_name}_test_pred_posterior_mean.npy'),
          test_pred_mean_array)
  # now find the entropy value here
  print(np.sum(np.isnan(test_pred_mean_array)))
  print(np.min(test_pred_mean_array))
  entropy = -np.sum(test_pred_mean_array * np.log2(test_pred_mean_array + 1e-7),
                    axis=1)
  print(entropy)
  print(np.max(entropy))
  print(np.min(entropy))
  print(np.sum(np.isnan(entropy)))
  print(np.sum(np.isinf(entropy)))
  np.save(os.path.join(save_dir, '{}_entropy.npy'.format(data_name)), entropy)
  plt.figure()
  plt.hist(entropy, 100, density=True)
  plt.savefig(os.path.join(save_dir, '{}_entropy.pdf'.format(data_name)))
  plt.savefig(os.path.join(save_dir, '{}_entropy.png'.format(data_name)))
  print(np.argmax(label_array, axis=1).shape)
  print(test_classification.shape)
  print(np.argmax(label_array, axis=1).shape)
  print(test_pred_mean_array[:, test_classification].reshape(-1).shape)
  print(test_pred_mean_array.shape)
  print(test_pred_mean_array[0, :])
  if create_reliability:
    # also get nll
    likelihood = tf.keras.metrics.CategoricalCrossentropy()
    likelihood.update_state(label_array, test_pred_mean_array)
    acc = accuracy_score(np.argmax(label_array, axis=1),
                         np.argmax(test_pred_mean_array, axis=1))
    print(f'neg ll = {likelihood.result().numpy()}, accuracy = {acc}')
    # now the reliability figure
    ece, fig = reliability_diagrams.reliability_diagram(np.argmax(label_array,
                                                             axis=1),
                                                   test_classification,
                                                   np.max(test_pred_mean_array,
                                                          axis=1),
                                                   num_bins=10,
                                                   draw_ece=True,
                                                   draw_bin_importance="alpha",
                                                   draw_averages=True,
                                                   title=None,
                                                   figsize=(6, 6),
                                                   dpi=100,
                                                   return_fig=True)
    fig.savefig(os.path.join(save_dir, '{}_calibration.pdf'.format(data_name)))
    fig.savefig(os.path.join(save_dir, '{}_calibration.png'.format(data_name)))
    return acc, likelihood.result().numpy(), ece

def create_entropy_hist_map(model,
                            save_dir,
                            test_ds,
                            data_name,
                            num_classes,
                            create_reliability=True,
                            final_activation=tf.keras.activations.softmax):
  """Plot entropy for the predicted output for MAP
  """
  # get a list of all the chain files
  map_path = os.path.join(save_dir, 'map_weights.pkl')
  # create a list to store all samples from predictive posterior
  pred_list = []
  with open(map_path, 'rb') as f:
    chain = pickle.load(f)
  pred_map, label_array = pred_eval_map_fn(
      model, chain, test_ds, final_activation=tf.keras.activations.softmax)
  #pred_forward_pass(model, weights_list, biases_list, elem)).numpy()
  # now find the entropy value here
  entropy = -np.sum(pred_map * np.log2(pred_map + 1e-7), axis=1)
  np.save(os.path.join(save_dir, '{}_entropy_map.npy'.format(data_name)),
          entropy)
  # now find the entropy value here
  plt.figure()
  plt.hist(entropy, 100, density=True)
  plt.savefig(os.path.join(save_dir, '{}_entropy_map.pdf'.format(data_name)))
  plt.savefig(os.path.join(save_dir, '{}_entropy_map.png'.format(data_name)))
  if create_reliability:
    # also get nll
    likelihood = tf.keras.metrics.CategoricalCrossentropy()
    likelihood.update_state(label_array, pred_map)
    acc = accuracy_score(np.argmax(label_array, axis=1),
                         np.argmax(pred_map, axis=1))
    print(f'neg ll = {likelihood.result().numpy()}, accuracy = {acc}')
    fig = reliability_diagrams.reliability_diagram(np.argmax(label_array,
                                                             axis=1),
                                                   np.argmax(pred_map, axis=1),
                                                   np.max(pred_map, axis=1),
                                                   num_bins=10,
                                                   draw_ece=True,
                                                   draw_bin_importance="alpha",
                                                   draw_averages=True,
                                                   title=None,
                                                   figsize=(6, 6),
                                                   dpi=100,
                                                   return_fig=True)
    fig.savefig(
        os.path.join(save_dir, '{}_calibration_map.pdf'.format(data_name)))
    fig.savefig(
        os.path.join(save_dir, '{}_calibration_map.png'.format(data_name)))


def build_network(json_path, x_train, data_dimension_dict):
  with open(json_path) as file_p:
    data = json.load(file_p)
  if (data['posterior'] == 'dense mcmc'):
    model = MCMCMLP("point",
                    "point",
                    json_path,
                    dimension_dict=data_dimension_dict)
  elif (data['posterior'] == 'conv mcmc'):
    model = MCMCConv("point",
                     "point",
                     json_path,
                     dimension_dict=data_dimension_dict)
  else:
    raise ValueError('Unsuitable model type supplied')
  # running one batch of training just to initialise the model
  _ = model(x_train)
  return model



def plot_regression_pred_posterior_neurips(model, chain, num_results, X_train,
                                           y_train, X_test, y_test, out_dir,
                                           name):
  num_returned_samples = len(chain[0])
  # perform prediction for each iteration
  num_samples = 1000  #np.min([1000, num_returned_samples])
  sample_idx = np.linspace(0, num_returned_samples - 2,
                           num_samples).astype(np.int64)
  num_plot = sample_idx.size
  pred_mean_train = np.zeros([num_plot, y_train.size])
  pred_mean_test = np.zeros([num_plot, y_test.size])
  pred_std_train = np.zeros([num_plot, y_train.size])
  pred_std_test = np.zeros([num_plot, y_test.size])
  print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
  pred_idx = 0
  for mcmc_idx in sample_idx:
    param_list_a = [x[mcmc_idx, ...] for x in chain]
    param_list_b = [x[mcmc_idx + 1, ...] for x in chain]
    param_list = [(a + b) / 2.0 for a, b in zip(param_list_a, param_list_b)]
    outputs_train = pred_forward_pass(model, param_list,
                                      X_train.astype(np.float32)).numpy()
    out_mean_train = outputs_train[:, 0]
    out_inv_softplus_std_train = outputs_train[:, 1]
    outputs_test = pred_forward_pass(model, param_list,
                                     X_test.astype(np.float32)).numpy()
    out_mean_test = outputs_test[:, 0]
    out_inv_softplus_std_test = outputs_test[:, 1]
    pred_mean_train[pred_idx, :] = out_mean_train
    pred_std_train[pred_idx, :] = tf.math.softplus(out_inv_softplus_std_train)
    pred_mean_test[pred_idx, :] = out_mean_test
    pred_std_test[pred_idx, :] = tf.math.softplus(out_inv_softplus_std_test)
    # plt.plot(X_test, pred_mean_train[pred_idx, :], alpha=0.05, color='k')
    pred_idx += 1
  # now lets get the mean of the mean, and the mean of the std and see what they are
  final_pred_mean_train = np.mean(pred_mean_train, axis=0)
  final_pred_std_train = np.mean(pred_std_train, axis=0)
  plt.figure(figsize=(10, 5))
  idx = np.argsort(y_train)
  plt.plot(final_pred_mean_train[idx], label="Predictive mean")
  plt.fill_between(np.arange(len(final_pred_mean_train)),
                   final_pred_mean_train[idx] + 2 * final_pred_std_train[idx],
                   final_pred_mean_train[idx] - 2 * final_pred_std_train[idx],
                   alpha=0.5,
                   label="2-Sigma region")
  plt.plot(y_train[idx], "-r", lw=3, label="Target Values")
  plt.legend(fontsize=14)
  plt.savefig(os.path.join(out_dir, 'pred.png'))
  plt.savefig(os.path.join(out_dir, 'pred.pdf'), bbox_inches='tight')
  # new lets save the data for submission
  # will need to transpose it though as asks for (num_data, samples)
  pred_test = pred_mean_test + np.random.randn(
      *pred_std_test.shape) * pred_std_test
  print('pred test shape = {}'.format(pred_test.shape))
  print('pred mean shape = {}'.format(pred_mean_test.shape))
  np.savetxt(os.path.join(out_dir, 'uci_test.csv'), np.transpose(pred_test))
