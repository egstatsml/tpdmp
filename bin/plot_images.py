"""plot images with predictions on file"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow_probability as tfp
import tensorflow as tf

from tbnn.pdmp.networks import get_network
from tbnn.pdmp import networks

from tbnn.pdmp.model import plot_image_iter_pred_posterior
# import some helper functions
#
from tbnn.pdmp.nfnets import ScaledStandardizedConv2D
from tbnn.utils import utils
from tbnn.pdmp.utils import compute_dot_prod


import argparse
import os
import sys
import time
import pickle

tfd = tfp.distributions

def get_data(data, batch_size, strategy):
  dataset = utils.load_dataset(data, batch_size)
  # create distributed version of it as required by the
  # distribution strategy
  training_data = strategy.experimental_distribute_dataset(dataset.train)
  # now create an iter object of it
  training_iter = iter(training_data)
  # training_iter = training_data
  return (training_iter, dataset.test, dataset.test_orig,
          dataset.dimension_dict, dataset.label_dict)



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
          if isinstance(_layer, (tf.keras.layers.Dense, ScaledStandardizedConv2D)):
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
  else:
    with open(args.map_path, 'rb') as f:
      map_initial_state = pickle.load(f)
  return map_initial_state



def main(args):
  """main function to fit the models"""
  # get the distribution strategy.

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

  strategy = get_distribution_strategy(args.gpus)
  # getting the data
  (training_iter, test_ds, orig_test_ds, data_dimension_dict,
   label_dict) = get_data(args.data, args.batch_size, strategy)
  # if is regression model
  if args.data in ['toy_a', 'toy_b', 'toy_c']:
    input_dims = [data_dimension_dict['in_dim']]
  else:
    input_dims = [
        data_dimension_dict['in_height'], data_dimension_dict['in_width'],
        data_dimension_dict['in_channels']
    ]
  # get single sample to build the model
  print(f"memory usage {tf.config.experimental.get_memory_info('GPU:0')['current'] / 10 ** 9} GB")
  model = get_network(args.network,
                      strategy,
                      input_dims,
                      data_dimension_dict['out_dim'],
                      prior=1.0,
                      map_training=True)#args.map_path == None)
  print(model)
  print(model.summary())
  print('model type = {}'.format(model))

  # load in the prediction data, specifically the pred posterior
  pred_posterior = np.load(
    os.path.join(args.out_dir, f'{args.data}_test_pred_posterior.npy'))
  # check to see if softmax needs to be applied
  if np.max(pred_posterior) > 1.0 or np.min(pred_posterior) < 0.0:
    pred_posterior = tf.nn.softmax(pred_posterior).numpy()

  pred_mean = np.mean(pred_posterior, axis=0)
  classification = np.argmax(pred_mean, axis=1)
  plot_image_iter_pred_posterior(args.out_dir,
                                 [32, 32, 3],
                                 label_dict,
                                 orig_test_ds,
                                 [],
                                 classification,
                                 pred_posterior,
                                 pred_mean)


def get_distribution_strategy(gpus):
  if len(gpus) == 1:
    print(gpus)
    return tf.distribute.OneDeviceStrategy(gpus[0])
  else:
    return tf.distribute.MirroredStrategy(gpus)



def check_cmd_args(args):
  """check all the commandline arguments are valid"""
  # check network argument
  networks.check_network_arg(args.network)
  utils.check_or_mkdir(args.out_dir)
  # let's also make sure that the gpu specification is valid
  args.gpus = check_format_gpus(args.gpus)
  return args


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
  parser.add_argument('data',
                      type=str,
                      help='data set to use')
  parser.add_argument('--out_dir',
                      type=str,
                      default='./out',
                      help='out directory where data is saved')
  parser.add_argument('--gpus',
                      type=str,
                      default='all',
                      help='gpus to use (default to all)')
  parser.add_argument('--batch_size',
                      type=int,
                      default=100,
                      help='Number of samples per batch')

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
  main(args)
