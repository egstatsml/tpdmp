import cv2
import gzip
import os
import pickle
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
import tensorflow_datasets as tfds
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow import keras
from matplotlib import pyplot as plt
import time

from keras_cv_attention_models.imagenet.data import (RandomProcessDatapoint,
                                                     apply_mixup_cutmix,
                                                     init_mean_std_by_rescale_mode,
                                                     evaluation_process_crop_resize)


class Data(object):
  """Will hold data set and some properties"""

  def __init__(self, train, test, test_orig, label_dict, dimension_dict):
    self.train = train
    self.test = test
    self.test_orig = test_orig
    self.label_dict = label_dict
    self.dimension_dict = dimension_dict


def load_dataset(data_class, batch_size):
  """Helper function to load in some standard data sets"""
  print(data_class.lower())
  if data_class.lower() == 'mnist':
    data_class = load_mnist(batch_size)
  elif data_class.lower() == 'mnist_im':
    data_class = load_mnist_im(batch_size)
  elif data_class.lower() == 'mnist_pad':
    data_class = load_mnist_pad(batch_size)
  elif data_class.lower() == 'fashion_mnist':
    data_class = load_fashion_mnist(batch_size)
  elif data_class.lower() == 'fashion_mnist_im':
    data_class = load_fashion_mnist_im(batch_size)
  elif data_class.lower() == 'cifar_10':
    data_class = load_cifar_10(batch_size)
  elif data_class.lower() == 'cifar_100':
    data_class = load_cifar_100(batch_size)
  elif data_class.lower() == 'svhn':
    data_class = load_svhn(batch_size)
  elif data_class.lower() == 'imagenet':
    data_class = load_imagenet(batch_size)
  elif (data_class.lower() == 'toy_a'):
    data_class = load_toy_a()
  elif (data_class.lower() == 'toy_b'):
    data_class = load_toy_b()
  elif (data_class.lower() == 'toy_c'):
    data_class = load_toy_c()
  elif (data_class.lower() == 'test_a'):
    data_class = load_test_a()
  elif (data_class.lower() == 'test_b'):
    data_class = load_test_b()
  elif (data_class.lower() == 'linear'):
    data_class = load_linear()
  elif (data_class.lower() == 'linear_norm'):
    data_class = load_linear_norm()
  elif data_class.lower() == 'moons':
    data_class = load_moons()
  elif data_class.lower() == 'clusters':
    data_class = load_clusters()
  else:
    raise ValueError("Invalid data set supplied")
  return data_class


def load_toy_a():
  """Load toy data from Bayes by Backprop. paper

  Original function from
  "Weight Uncertainty in Deep Learning" by Blundell et al.
  https://arxiv.org/abs/1505.05424
  """
  num_train = 500
  num_test = 1000
  epsilon = np.random.randn(num_train, 1) * 0.015
  x_train = np.sort(np.random.uniform(0, 0.5,
                                      num_train)).reshape(-1, 1)
  x_test = np.linspace(-0.2, 0.8, num_test).reshape(-1, 1)
  y_train = x_train + 0.3 * np.sin(
      2 * np.pi * (x_train + epsilon)) + (
          0.3 * np.sin(4 * np.pi * (x_train + epsilon)) + epsilon)
  y_test = x_test + 0.3 * np.sin(
      2 * np.pi * x_test) + (0.3 *
                             np.sin(4 * np.pi * x_test))
  x_train = x_train - 0.25
  x_test = x_test - 0.25
  # now create the dictionary to hold the dimensions of our data
  dimension_dict = {
      'in_dim': 1,
      'out_dim': 1,
      'in_width': 1,
      'in_height': 1,
      'in_channels': 1,
      'dataset_size': 500
  }
  print(x_train.shape)
  print(y_train.shape)
  # now make some tf.Datasets with them
  train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(500).batch(50)
  train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(num_train)
  test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(num_test)
  # make the training dataset an iterable and repeat it
  train = train.repeat()
  data_class = Data(train=train,
                    test=test,
                    test_orig=None,
                    label_dict=None,
                    dimension_dict=dimension_dict)
  return data_class


def load_toy_b():
  """Generate data with holes in it

  Original function taken from
  "Deep Exploration via Bootstrapped DQN" by Osband et al.
  https://arxiv.org/pdf/1602.04621.pdf
  In Appendix A
  """
  num_train = 500
  num_test = 500
  epsilon = np.random.randn(num_train, 1) * 0.015
  x_train = np.concatenate(
      (np.random.uniform(0.0, 0.6, np.int(num_train / 2)),
       np.random.uniform(0.8, 1.0, np.int(num_train / 2))))
  # reshape to ensure is correct dims
  x_train = np.sort(x_train).reshape(-1, 1)
  x_test = np.linspace(-0.5, 1.5, num_test).reshape(-1, 1)
  y_train = (x_train +
                        np.sin(4.0 * (x_train + epsilon)) +
                        np.sin(13.0 * (x_train + epsilon)) + epsilon)
  y_test = (x_test + np.sin(4.0 * x_test) +
                       np.sin(13.0 * x_test))
  x_val = x_test
  y_val = y_test
  x_train = x_train - 0.5
  x_test = x_test - 0.5
  x_val = x_val - 0.5
  # now create the dictionary to hold the dimensions of our data
  dimension_dict = {
      'in_dim': 1,
      'out_dim': 1,
      'in_width': 1,
      'in_height': 1,
      'in_channels': 1,
      'dataset_size': 500
  }
  # now make some tf.Datasets with them
  train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(num_train)
  test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(num_test)
  # make the training dataset an iterable and repeat it
  train = train.repeat()
  data_class = Data(train=train,
                    test=test,
                    test_orig=None,
                    label_dict=None,
                    dimension_dict=dimension_dict)


  #validation data is the same as the test data
  return data_class


def load_toy_c():
  """non-linear function centered at x = 0

  Function from Yarin Gal's Blog
  http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html
  """
  num_train = 500
  num_test = 500
  epsilon = np.random.randn(num_train, 1) * 0.015
  x_train = np.linspace(-0.45, 0.45, num_train).reshape(-1, 1)
  x_test = np.linspace(-0.6, 0.6, num_test).reshape(-1, 1)
  y_train = (
      x_train * np.sin(4.0 * np.pi * x_train) + epsilon)
  y_test = x_test * np.sin(
      4.0 * np.pi * x_test)
  # now create the dictionary to hold the dimensions of our data
  dimension_dict = {
      'in_dim': 1,
      'out_dim': 1,
      'in_width': 1,
      'in_height': 1,
      'in_channels': 1,
      'dataset_size': 500
  }
  #
  # now make some tf.Datasets with them
  train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(num_train)
  test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(num_test)
  # make the training dataset an iterable and repeat it
  train = train.repeat()
  data_class = Data(train=train,
                    test=test,
                    test_orig=None,
                    label_dict=None,
                    dimension_dict=dimension_dict)
  return data_class


def load_clusters(num_data=200, num_test=100):
  """creates clusters sampled from two Gaussians
  """
  data_class = Data()
  x_1 = np.random.multivariate_normal([0.0, 1.0],
                                      np.array([[1.0, 0.8], [0.8, 1.0]]),
                                      size=num_data // 2)
  x_2 = np.random.multivariate_normal([2.0, 0.0],
                                      np.array([[1.0, -0.4], [-0.4, 1.0]]),
                                      size=num_data // 2)
  X = np.vstack([x_1, x_2])
  y = np.ones(num_data).reshape(-1, 1)
  y[:np.int(num_data / 2)] = 0
  print('X shape = {}'.format(X.shape))
  print('y shape = {}'.format(y.shape))
  scaler = MinMaxScaler((-1.0, 1.0)).fit(X)
  X = scaler.transform(X)
  # create a random split now
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=num_test)
  data_class.x_train = x_train
  data_class.x_test = x_test
  data_class.y_train = y_train
  data_class.y_test = y_test
  # now create the dictionary to hold the dimensions of our data
  data_class.x_val = data_class.x_test
  data_class.y_val = data_class.y_test
  data_class.dimension_dict = {
      'in_dim': 2,
      'out_dim': 1,
      'in_width': 1,
      'in_height': 1,
      'in_channels': 1,
      'dataset_size': 200
  }
  #validation data is the same as the test data
  return data_class


def load_moons(num_data=5000, num_test=100):
  """The make moons data_class from sklearn

  Function from Yarin Gal's Blog
  http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_2248.html
  """
  num_train = 150
  num_test = 500
  X, Y = make_moons(noise=0.1, random_state=0, n_samples=num_data)
  Y = Y.reshape([-1, 1])
  scaler = MinMaxScaler((-1.0, 1.0)).fit(X)
  X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float32),
                                                      Y.astype(np.float32))
  x_train = scaler.transform(X_train)
  x_test = scaler.transform(X_test)
  y_train = y_train
  y_test = y_test
  # now create the dictionary to hold the dimensions of our data
  x_val = scaler.transform(X_test)
  y_val = y_test
  dimension_dict = {
      'in_dim': 2,
      'out_dim': 1,
      'in_width': 1,
      'in_height': 1,
      'in_channels': 1,
      'dataset_size': 150
  }
  # now make some tf.Datasets with them
  train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(num_train)
  test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(num_test)
  # make the training dataset an iterable and repeat it
  train = train.repeat()
  data_class = Data(train=train,
                    test=test,
                    test_orig=None,
                    label_dict=None,
                    dimension_dict=dimension_dict)

  #validation data is the same as the test data
  return data_class


def load_mnist_old(batch_size):
  """loads in mnist data in flattened array format"""
  file_p = gzip.open(os.path.join(os.environ['DATA_PATH'], 'mnist.pkl.gz'),
                     'rb')
  data_set_list = pickle.load(file_p, encoding='latin1')
  file_p.close()
  data_class = Data()
  #concatenate the data from the predefined train/val/test splits
  combined_x = np.concatenate(
      (data_set_list[0][0], data_set_list[1][0], data_set_list[2][0]), axis=0)
  combined_y = np.concatenate(
      (data_set_list[0][1], data_set_list[1][1], data_set_list[2][1]), axis=0)
  #centre data around zero in range of [-1, 1]
  combined_x = data_centre_zero(combined_x)
  #split the data sets how I want
  train_split, val_split, test_split = convert_split(split, combined_y.size)
  print('train_split = {}'.format(train_split))
  #seperate the data now
  print('combined type = {}'.format(type(combined_x)))
  print('combined = {}'.format(combined_x))

  data_class.x_train = combined_x[0:train_split, :]
  data_class.y_train = one_hot(combined_y[0:train_split], 10)
  data_class.x_val = combined_x[train_split:train_split + val_split, :]
  data_class.y_val = one_hot(combined_y[train_split:train_split + val_split],
                             10)
  test_start = train_split + val_split
  data_class.x_test = combined_x[test_start:test_start + test_split, :]
  data_class.y_test = one_hot(combined_y[test_start:test_start + test_split],
                              10)
  data_class.label_dict = {
      0: 0,
      1: 1,
      2: 2,
      3: 3,
      4: 4,
      5: 5,
      6: 6,
      7: 7,
      8: 8,
      9: 9
  }
  # now create the dictionary to hold the dimensions of our data
  data_class.dimension_dict = {
      'in_dim': 784,
      'out_dim': 10,
      'in_width': 1,
      'in_height': 1,
      'in_channels': 1
  }
  print(np.max(data_class.x_train), np.min(data_class.x_train))
  return data_class


def load_mnist(batch_size):
  """loads in mnist data in flattened array format"""
  data_class = Data()
  (x_train, y_train), (x_test, y_test) = keras.data_classs.mnist.load_data()
  print(x_train.shape)
  print(type(x_train))
  print(type(y_train))
  data_class.x_train = data_centre_zero(
      np.expand_dims(x_train, axis=-1).astype(np.float32))
  data_class.x_test = data_centre_zero(
      np.expand_dims(x_test, axis=-1).astype(np.float32))
  data_class.y_train = keras.utils.to_categorical(y_train, 10)
  data_class.y_test = keras.utils.to_categorical(np.array(y_test), 10)
  data_class.x_val = data_class.x_test
  data_class.y_val = data_class.y_test
  data_class.dimension_dict = {
      'in_dim': 784,
      'out_dim': 10,
      'in_height': 28,
      'in_width': 28,
      'in_channels': 1,
      'dataset_size': 50000
  }
  print(type(data_class.x_train))
  return data_class


def load_mnist_im(batch_size):
  """load in mnist data in image format"""
  ds_train = tfds.load('mnist',
                       split='train',
                       shuffle_files=True,
                       download=True,
                       as_supervised=True)
  ds_test = tfds.load('mnist',
                      split='test',
                      shuffle_files=False,
                      download=True,
                      as_supervised=True)

  image, label = next(iter(ds_train.batch(50000)))
  print(np.max(image))
  print(image.shape)
  # image = tf.cast(image, dtype=tf.float32) / 255.0
  # print(np.mean(image))
  # print(np.std(image))
  # time.sleep(10)




  # need to preprocess, will center around zero
  def mnist_pre(images, label):
    images = tf.cast(images, tf.float32)
    # divide by 255 and then subtract mean and std.
    images = images /  255.0#data_centre_zero(images)
    images = (images - 0.13069087) / 0.30814707
    return images, tf.one_hot(label, 10)

  # for the original testing data, I just want to apply one hot encoding to
  # the labels
  def one_hot_orig_test(image, label):
    return image, tf.one_hot(label, 10)

  # batch the entire training set and perform the data preprocessing
  # ds_train_preprocessed = ds_train.batch(50000).map(
      # mnist_pre, num_parallel_calls=tf.data.AUTOTUNE).cache()
  # ds_test_preprocessed = ds_test.batch(10000).map(
      # mnist_pre, num_parallel_calls=tf.data.AUTOTUNE).cache()
  # ds_test_orig = ds_test.batch(10000).map(
      # one_hot_orig_test, num_parallel_calls=tf.data.AUTOTUNE).cache()
  ds_train_preprocessed = ds_train.map(
      mnist_pre, num_parallel_calls=tf.data.AUTOTUNE).cache()
  ds_test_preprocessed = ds_test.map(
      mnist_pre, num_parallel_calls=tf.data.AUTOTUNE).cache()

  # now unbatch
  # ds_train_preprocessed = ds_train_preprocessed.unbatch()
  # ds_test_preprocessed = ds_test_preprocessed.unbatch()
  # now to shuffle
  ds_train_preprocessed = ds_train_preprocessed.shuffle(50000)
  # now batch the datasets
  ds_train_preprocessed = ds_train_preprocessed.batch(batch_size)
  ds_test_preprocessed = ds_test_preprocessed.batch(1000)
  # now to prefetch
  ds_train_preprocessed = ds_train_preprocessed.prefetch(tf.data.AUTOTUNE)
  ds_test_preprocessed = ds_test_preprocessed.prefetch(tf.data.AUTOTUNE)
  ds_test_orig = ds_test.map(one_hot_orig_test).batch(1000)
  # repeat the training data
  ds_train_preprocessed = ds_train_preprocessed.repeat()
  # now create the dictionary to hold the dimensions of our data
  dimension_dict = {
      'in_dim': 784,
      'out_dim': 10,
      'in_height': 28,
      'in_width': 28,
      'in_channels': 1,
      'dataset_size': 60000
  }
  label_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
  # now putting it in a class for all the data
  data_class = Data(train=ds_train_preprocessed,
                    test=ds_test_preprocessed,
                    test_orig=ds_test_orig,
                    label_dict=label_dict,
                    dimension_dict=dimension_dict)
  return data_class


def load_mnist_pad(batch_size):
  """load in mnist data and pad with zeros to be 32x32

  This is what is done in the original LeNet paper
  """
  data_class = load_mnist_im(batch_size)
  print('mnist before shape = {}'.format(data_class.x_train.shape))
  # now reshape all the X variables to 3-d arrays
  pad_sequence = ((0, 0), (2, 2), (2, 2), (0, 0))
  data_class.x_train = np.pad(data_class.x_train, pad_sequence, 'constant')
  data_class.x_val = np.pad(data_class.x_val, pad_sequence, 'constant')
  data_class.x_test = np.pad(data_class.x_test, pad_sequence, 'constant')
  data_class.x_test_orig = np.copy(data_class.x_test)
  print('mnist padded shape = {}'.format(data_class.x_train.shape))
  # now need to update the dictionary, as the dimensions are now different
  data_class.dimension_dict = {
      'in_dim': 1024,
      'out_dim': 10,
      'in_height': 32,
      'in_width': 32,
      'in_channels': 1
  }
  return data_class


def load_fashion_mnist(batch_size):
  """loads in mnist data in flattened array format"""
  print('WARNING!, have not implemented validation yet')
  print('Currently just using validation set as test set')
  data_class = Data()
  data_dir = os.path.join(os.environ['DATA_PATH'], 'fashion_mnist')
  # train_x_file_p = gzip.open(os.path.join(
  #   data_dir, 'train-images-idx3-ubyte.gz'), 'rb')
  # train_y_file_p = gzip.open(os.path.join(
  #   data_dir, 'train-labels-idx1-ubyte.gz'), 'rb')
  with gzip.open(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'),
                 'rb') as f:
    y_train = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
  with gzip.open(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'),
                 'rb') as f:
    x_train = np.frombuffer(f.read(), dtype=np.uint8,
                            offset=16).reshape(len(y_train), 784)
  with gzip.open(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'),
                 'rb') as f:
    y_test = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
  with gzip.open(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'),
                 'rb') as f:
    x_test = np.frombuffer(f.read(), dtype=np.uint8,
                           offset=16).reshape(len(y_test), 784)
  #centre data around zero in range of [-1, 1]
  data_class.x_train = data_centre_zero(x_train.astype(np.float32))
  data_class.y_train = one_hot(y_train, 10)
  data_class.x_test = data_centre_zero(x_test.astype(np.float32))
  data_class.y_test = one_hot(y_test, 10)
  # validation set not implemented for this data_class, so just
  # setting to same as testing data.
  # printed a big ole fat warning at the statr to remind me each time
  data_class.x_val = data_class.x_test
  data_class.y_val = data_class.y_test
  data_class.label_dict = {
      0: 't-shirt',
      1: 'trouser',
      2: 'pullover',
      3: 'dress',
      4: 'coat',
      5: 'sandal',
      6: 'shirt',
      7: 'sneaker',
      8: 'bag',
      9: 'ankle boot'
  }
  # now create the dictionary to hold the dimensions of our data
  data_class.dimension_dict = {
      'in_dim': 784,
      'out_dim': 10,
      'in_width': 1,
      'in_height': 1,
      'in_channels': 1,
      'dataset_size': 50000
  }
  return data_class



def load_fashion_mnist_im(batch_size):
  """load in mnist data in image format"""
  ds_train = tfds.load('fashion_mnist',
                       split='train',
                       shuffle_files=True,
                       download=True,
                       as_supervised=True)
  ds_test = tfds.load('fashion_mnist',
                      split='test',
                      shuffle_files=False,
                      download=True,
                      as_supervised=True)

  # need to preprocess, will center around zero
  def fashion_mnist_pre(images, label):
    images = tf.cast(images, tf.float32)
    # divide by 255 and then subtract mean and std.
    images = images / 255.0
    images = (images - 0.2860361) / 0.3531331
    return images, tf.one_hot(label, 10)

  # for the original testing data, I just want to apply one hot encoding to
  # the labels
  def one_hot_orig_test(image, label):
    return image, tf.one_hot(label, 10)

  ds_train_preprocessed = ds_train.map(
      fashion_mnist_pre, num_parallel_calls=tf.data.AUTOTUNE).cache()
  ds_test_preprocessed = ds_test.map(
      fashion_mnist_pre, num_parallel_calls=tf.data.AUTOTUNE).cache()
  # now unbatch
  # now to shuffle
  ds_train_preprocessed = ds_train_preprocessed.shuffle(60000)
  # now batch the datasets
  ds_train_preprocessed = ds_train_preprocessed.batch(batch_size)
  ds_test_preprocessed = ds_test_preprocessed.batch(1000)
  # now to prefetch
  ds_train_preprocessed = ds_train_preprocessed.prefetch(tf.data.AUTOTUNE)
  ds_test_preprocessed = ds_test_preprocessed.prefetch(tf.data.AUTOTUNE)
  ds_test_orig = ds_test.map(one_hot_orig_test).batch(1000)
  # repeat the training data
  ds_train_preprocessed = ds_train_preprocessed.repeat()
  # now create the dictionary to hold the dimensions of our data
  dimension_dict = {
      'in_dim': 784,
      'out_dim': 10,
      'in_height': 28,
      'in_width': 28,
      'in_channels': 1,
      'dataset_size': 60000
  }
  label_dict = {
      0: 't-shirt',
      1: 'trouser',
      2: 'pullover',
      3: 'dress',
      4: 'coat',
      5: 'sandal',
      6: 'shirt',
      7: 'sneaker',
      8: 'bag',
      9: 'ankle boot'
  }

  # now putting it in a class for all the data
  data_class = Data(train=ds_train_preprocessed,
                    test=ds_test_preprocessed,
                    test_orig=ds_test_orig,
                    label_dict=label_dict,
                    dimension_dict=dimension_dict)
  return data_class


def load_cifar_10(batch_size):
  num_classes = 10
  label_dict = {
      0: 'airplane',
      1: 'automobile',
      2: 'bird',
      3: 'cat',
      4: 'deer',
      5: 'dog',
      6: 'frog',
      7: 'horse',
      8: 'ship',
      9: 'truck'
    }
  return _load_cifar('cifar10', num_classes, label_dict, batch_size, augment=False)

def load_cifar_100(batch_size):
  num_classes = 100
  label_dict = {
      0: 'airplane',
      1: 'automobile',
      2: 'bird',
      3: 'cat',
      4: 'deer',
      5: 'dog',
      6: 'frog',
      7: 'horse',
      8: 'ship',
      9: 'truck'
    }
  return _load_cifar('cifar100', num_classes, label_dict, batch_size, augment=False)



def _load_cifar(cifar_type, num_classes, label_dict, batch_size, augment=False):
  """Load and preprocess cifar data_class

  Can handle both cifar10 and cifar100, will do the same preprocessing to both.
  This method is designed to be called by the load_cifar_10 and load_cifar_100
  functions.

  Will normalise the data so that is centered around zero with unit standard
  deviation with respect to the channels.

  Parameters
  ----------
  batch_size : int
      batch size for training

  """
  ds_train, _ = tfds.load(cifar_type,
                          split='train',
                          as_supervised=True,
                          with_info=True)
  ds_test, _ = tfds.load(cifar_type,
                         split='test',
                         as_supervised=True,
                         with_info=True)

  def img_to_float32(image, label):
    return tf.image.convert_image_dtype(image, tf.float32), label

  # ensuring is 32 bit float, and doing it over the entire data_class before
  # batching as can fit it all in memory.
  # am also cacheing the operation to save repetitive calculations.
  ds_train = ds_train.map(img_to_float32).cache()
  ds_test = ds_test.map(img_to_float32).cache()

  # preprocessing needed
  def img_normalize(image, label):
    """Normalize the image to zero mean and unit variance."""
    # mean = (0.49, 0.48, 0.44)
    # std = (0.2, 0.2, 0.2)
    # image -= tf.constant(mean, shape=[1, 1, 3], dtype=image.dtype)
    # image /= tf.constant(std, shape=[1, 1, 3], dtype=image.dtype)
    return image, tf.one_hot(label, num_classes)

  def apply_augmentation(image, label):
    """apply augmentation used for cifar10 in resnet paper"""
    # pad image 4 pixels each side
    image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40)
    # now take a random crop
    image = tf.image.random_crop(image, [32, 32, 3])
    # now do a random fliplr
    image = tf.image.random_flip_left_right(image)
    return image, label

  # now get the train and test data
  # can also cache this operation
  ds_train_preprocessed = ds_train.map(img_normalize).cache()
  ds_test_preprocessed = ds_test.map(img_normalize).cache()
  # if we are doing some augmentation, let's do it now
  if augment:
   ds_train_preprocessed = ds_train_preprocessed.map(
      apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
  # shuffle the training data
  # data set is small enough for us to shuffle everything
  ds_train_preprocessed = ds_train_preprocessed.shuffle(
      ds_train_preprocessed.cardinality())
  # now will batch the data_class
  ds_train_preprocessed = ds_train_preprocessed.batch(batch_size)
  ds_test_preprocessed = ds_test_preprocessed.batch(batch_size)
  ds_test = ds_test.batch(batch_size)
  # now repeat the training data and then turn into an iter
  ds_train_preprocessed = ds_train_preprocessed.repeat()
  # create a data_class class to save it all
  dimension_dict = {
      'in_dim': 3072,
      'out_dim': num_classes,
      'in_height': 32,
      'in_width': 32,
      'in_channels': 3,
      'dataset_size': 50000
  }

  # now putting it in a class for all the data
  data_class = Data(train=ds_train_preprocessed,
                    test=ds_test_preprocessed,
                    test_orig=ds_test,
                    label_dict=label_dict,
                    dimension_dict=dimension_dict)
  return data_class

def load_imagenet(batch_size, augment=False):
  manual_dir = os.path.join(os.environ['DATA_PATH'], 'imagenet_tf', 'tar')
  extract_dir = os.path.join(os.environ['DATA_PATH'], 'imagenet_tf', 'tf',
                             'extracted')
  download_dir = os.path.join(os.environ['DATA_PATH'], 'imagenet_tf', 'tf',
                              'download')
  data_dir = os.path.join(os.environ['DATA_PATH'], 'imagenet_tf', 'tf', 'data')
  # Construct a tf.data.Data_Class
  download_config = tfds.download.DownloadConfig(extract_dir=extract_dir,
                                                 manual_dir=manual_dir)
  download_and_prepare_kwargs = {
      'download_dir': download_dir,
      'download_config': download_config,
  }
  # ds_train = tfds.load('imagenet2012',
  #                      data_dir=data_dir,
  #                      split='train',
  #                      shuffle_files=True,
  #                      download=True,
  #                      as_supervised=True,
  #                      download_and_prepare_kwargs=download_and_prepare_kwargs)
  # ds_val = tfds.load('imagenet2012',
  #                    data_dir=data_dir,
  #                    split='validation',
  #                    shuffle_files=False,
  #                    download=True,
  #                    as_supervised=True,
  #                    download_and_prepare_kwargs=download_and_prepare_kwargs)

  # # need some additional functions to preprocess the data nicely
  # def resnet_preprocess(image, label):
  #   image = tf.keras.applications.imagenet_utils.preprocess_input(image,
  #                                                                 mode='torch')
  #   return image, tf.one_hot(label, 1000)

  # def resize_with_crop(image, label):
  #   image = image
  #   image = tf.cast(image, tf.float32)
  #   image = tf.image.resize_with_crop_or_pad(image, 180, 180)
  #   return (image, label)

  # # need to prefetch the data
  # # ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
  # # ds_val = ds_val.prefetch(tf.data.AUTOTUNE)
  # # want to parallelize reading of the data
  # # ds_train = ds_train.interleave(tf.data.AUTOTUNE)
  # # ds_val = ds_val.interleave(tf.data.AUTOTUNE)
  # # want to shuffle the training data_class
  # # ds_train = ds_train.shuffle(10000)
  # # Preprocess the images
  # # am keeping a version of the original data_class without this preprocessing, as I
  # # want to use it a bit later on when plotting the actual output at the end.
  # ds_train = ds_train.map(resize_with_crop, num_parallel_calls=tf.data.AUTOTUNE)
  # ds_val_orig = ds_val.map(resize_with_crop,
  #                          num_parallel_calls=tf.data.AUTOTUNE)
  # # now batch, as we want to vectorize application of certain mappings
  # ds_train = ds_train.batch(batch_size)
  # ds_val = ds_val_orig.batch(200)
  # ds_train_preprocessed = ds_train.map(resnet_preprocess,
  #                                      num_parallel_calls=tf.data.AUTOTUNE)
  # ds_val_preprocessed = ds_val.map(resnet_preprocess,
  #                                  num_parallel_calls=tf.data.AUTOTUNE)
  # ds_tmp = ds_train_preprocessed.take(1)
  # # now repeat the training data and then turn into an iter
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

  (ds_train_preprocessed,
   ds_val_preprocessed,
   ds_val, total_images, num_classes, steps_per_epoch) = init_imagenet_dataset(
     'imagenet2012', augment, data_dir, download_and_prepare_kwargs, input_shape=(224, 224), batch_size=batch_size)

  # ds_train_preprocessed = ds_train_preprocessed.with_options(options)
  # ds_val_preprocessed = ds_val_preprocessed.with_options(options)
  ds_train_preprocessed = ds_train_preprocessed.prefetch(tf.data.AUTOTUNE)
  ds_val_preprocessed = ds_val_preprocessed.prefetch(tf.data.AUTOTUNE)
  ds_train_preprocessed = ds_train_preprocessed.repeat()


  dimension_dict = {
      'in_dim': 224 * 224 * 3,
      'out_dim': 1000,
      'in_height': 224,
      'in_width': 224,
      'in_channels': 3,
      'dataset_size': 1280000
  }
  label_dict = {
      0: 'airplane',
      1: 'automobile',
      2: 'bird',
      3: 'cat',
      4: 'deer',
      5: 'dog',
      6: 'frog',
      7: 'horse',
      8: 'ship',
      9: 'truck'
  }
  # now putting it in a class for all the data
  data_class = Data(train=ds_train_preprocessed,
                    test=ds_val_preprocessed,
                    test_orig=ds_val,
                    label_dict=label_dict,
                    dimension_dict=dimension_dict)

  return data_class


def init_imagenet_dataset(
    data_name, # imagenet or subset
    augment,
    data_dir,
    download_and_prepare_kwargs,
    input_shape=(224, 224),
    batch_size=64,
    buffer_size=500,
    info_only=False,
    mixup_alpha=0.2,  # mixup / cutmix params
    cutmix_alpha=0.2,
    rescale_mode="tf",  # rescale mode, ["tf", "torch"], or specific `(mean, std)` like `(128.0, 128.0)`
    eval_central_crop=1.0,  # augment params
    random_crop_min=1.0,
    resize_method="bilinear",  # ["bilinear", "bicubic"]
    resize_antialias=False,
    random_erasing_prob=0.0,
    magnitude=0,
    num_layers=2,
    use_positional_related_ops=True,
    use_shuffle=True,
    seed=None,
    token_label_file=None,
    token_label_target_patches=-1,
    teacher_model=None,
    teacher_model_input_shape=-1,  # -1 means same with input_shape
    **augment_kwargs,  # Too many...
):
  """Is taken from keras-cv-attention.

  I have adjusted it to work with the datasets I need.
  """
  is_tpu = True if len(tf.config.list_logical_devices(
      "TPU")) > 0 else False  # Set True for try_gcs and drop_remainder
  use_token_label = False if token_label_file is None else True
  use_distill = False if teacher_model is None else True
  teacher_model_input_shape = input_shape if teacher_model_input_shape == -1 else teacher_model_input_shape

  if data_name.endswith(".json"):
    dataset, total_images, num_classes, num_channels = recognition_dataset_from_custom_json(
        data_name, with_info=True)
  else:
    dataset, info = tfds.load(data_name,
                              with_info=True,
                              data_dir=data_dir,
                              download_and_prepare_kwargs=download_and_prepare_kwargs,
                              try_gcs=is_tpu)
    num_classes = info.features["label"].num_classes
    num_channels = info.features["image"].shape[-1]
    total_images = info.splits["train"].num_examples
  steps_per_epoch = int(tf.math.ceil(total_images / float(batch_size)))

  if info_only:
    return total_images, num_classes, steps_per_epoch, num_channels  # return num_channels in case it's not 3
  """ Train dataset """
  train_dataset = dataset["train"]
  if use_token_label:
    train_dataset = build_token_label_dataset(train_dataset, token_label_file)
  # creating functions for preprocessing/augmentation.
  # if not augmenting, will use test preprocessing
  AUTOTUNE = tf.data.AUTOTUNE
  test_pre_batch = lambda xx: evaluation_process_crop_resize(
    xx, input_shape[:2], eval_central_crop, resize_method, resize_antialias)
  if augment:
    train_pre_batch = RandomProcessDatapoint(
        target_shape=teacher_model_input_shape if use_distill else input_shape,
        central_crop=
        -1,  # Resize directly w/o crop, if random_crop_min not in (0, 1)
        random_crop_min=random_crop_min,
        resize_method=resize_method,
        resize_antialias=resize_antialias,
        random_erasing_prob=random_erasing_prob,
        magnitude=magnitude,
        num_layers=num_layers,
        use_positional_related_ops=use_positional_related_ops,
        use_token_label=use_token_label,
        token_label_target_patches=token_label_target_patches,
        num_classes=num_classes,
        **augment_kwargs,
      )
  else:
    # use the same preprocessing as the testing
    train_pre_batch = test_pre_batch
  if use_shuffle:
    train_dataset = train_dataset.shuffle(buffer_size, seed=seed)
  train_dataset = train_dataset.map(train_pre_batch,
                                    num_parallel_calls=AUTOTUNE).batch(
                                        batch_size, drop_remainder=is_tpu)

  mean, std = init_mean_std_by_rescale_mode(rescale_mode)
  if use_token_label:
    train_post_batch = lambda xx, yy, token_label: (
        (xx - mean) / std, tf.one_hot(yy, num_classes), token_label)
  else:
    train_post_batch = lambda xx, yy: (
        (xx - mean) / std, tf.one_hot(yy, num_classes))
  train_dataset = train_dataset.map(train_post_batch,
                                    num_parallel_calls=AUTOTUNE)
  if augment:
    train_dataset = apply_mixup_cutmix(train_dataset,
                                       mixup_alpha,
                                       cutmix_alpha,
                                       switch_prob=0.5)

  if use_token_label:
    train_dataset = train_dataset.map(lambda xx, yy, token_label:
                                      (xx, (yy, token_label)))
  elif use_distill:
    print(">>>> KLDivergence teacher model provided.")
    train_dataset = build_distillation_dataset(train_dataset, teacher_model,
                                               input_shape)

  train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
  # return train_dataset
  """ Test dataset """
  test_dataset = dataset['validation']#.get("validation", dataset.get("test", None))
  if test_dataset is not None:
    # test_pre_batch = lambda xx: evaluation_process_resize_crop(xx, input_shape[:2], eval_central_crop, resize_method, resize_antialias)  # timm
    test_dataset = test_dataset.map(test_pre_batch, num_parallel_calls=AUTOTUNE)
    # Have to drop_remainder also for test set...
    test_dataset = test_dataset.batch(batch_size, drop_remainder=is_tpu)
    if use_token_label:
      test_post_batch = lambda xx, yy: (
          (xx - mean) / std, (tf.one_hot(yy, num_classes), None
                             ))  # just give None on token_label data position
    elif use_distill:
      test_post_batch = lambda xx, yy: ((xx - mean) / std,
                                        (tf.one_hot(yy, num_classes), None))
    else:
      test_post_batch = lambda xx, yy: (
          (xx - mean) / std, tf.one_hot(yy, num_classes))
    test_dataset_preprocessed = test_dataset.map(test_post_batch)
  return train_dataset, test_dataset_preprocessed, test_dataset, total_images, num_classes, steps_per_epoch


# def load_cifar_10(batch_size):
#   """load in cifar-10 data"""
#   print('WARNING!, have not implemented validation yet')
#   print('Currently just using validation set as test set')
#   data_class = Data()
#   cifar_dir = os.path.join(os.environ['DATA_PATH'], 'cifar-10-batches-py/')
#   data_files = [f for f in os.listdir(cifar_dir) if ('data_batch' in f)]
#   train_data = {}
#   for data_file in data_files:
#     file_path = os.path.join(cifar_dir, data_file)
#     with open(file_path, 'rb') as file_p:
#       batch_data = pickle.load(file_p, encoding='latin1')
#     # if this is the first batch we are loading in
#     if('data' not in train_data):
#       train_data = batch_data
#       # otherwise append the new data to the other pre-loaded batches
#       train_data['data'] = np.concatenate((train_data['data'],
#                                            batch_data['data']), axis=0)
#       train_data['labels'].extend(batch_data['labels'])

#  # now load in the testing data
#   print('cifar_dir = {}'.format(cifar_dir))
#   test_path = os.path.join(cifar_dir, 'test_batch')
#   with open(test_path, 'rb') as file_p:
#     test_data = pickle.load(file_p, encoding='latin1')
#   data_class.x_train, data_class.y_train, image_dg, min_, max_ = unpack_normalise_cifar(train_data)
#   data_class.x_test, data_class.y_test, _, _, _ = unpack_normalise_cifar(test_data,
#                                                                    image_dg,
#                                                                    min_, max_)
#   data_class.x_val, data_class.y_val, _, _,  _ = unpack_normalise_cifar(test_data,
#                                                                   image_dg,
#                                                                   min_, max_)
#   data_class.x_test_orig = test_data['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
#   data_class.data_dict = {0:'airplane', 1:'automobile', 2:'bird',
#                3:'cat', 4:'deer', 5:'dog',
#                6:'frog', 7:'horse', 8:'ship',
#                9:'truck'}
#   # now create the dictionary to hold the dimensions of our data
#   data_class.dimension_dict = {'in_dim' : 3072, 'out_dim' : 10, 'in_height': 32,
#                     'in_width': 32, 'in_channels': 3}
#   return data_class


def unpack_normalise_cifar(data,
                           image_dg=None,
                           min_=None,
                           max_=None,
                           mean_=None):
  # # """unpack data from cifar format and normalise it"""
  images = data['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
  labels = data['labels']
  images = images.astype(np.float32)
  pre_images = images / 255.0
  mean = np.array((0.49, 0.48, 0.44)).reshape([1, 1, 1, 3])
  stds = np.array((0.2, 0.2, 0.2)).reshape([1, 1, 1, 3])
  pre_images = (pre_images - mean) / stds
  return pre_images, one_hot(np.array(labels), 10), 0, 0, 0
  # # make sure the data is a float now
  # images = images.astype(np.float32)
  # print('image max  = {}, image min = {}, image mean = {}'.format(
  #   np.max(images), np.min(images), np.mean(images)))
  # pre_images = images / 255.0
  # # now performing ZCA whitening
  # # am using the ImageDataGenerator from Keras to do this, as it is built into
  # # there. Should only be fit on the training data, and use the same
  # # transformations on the test/val data.
  # if image_dg == None:
  #   image_dg = keras.preprocessing.image.ImageDataGenerator(
  #     zca_whitening=True)
  #     #featurewise_center=True)
  #   #featurewise_std_normalization=False)
  #   image_dg.fit(images)
  # # # now flow from there
  # image_iter = image_dg.flow(images, batch_size=images.shape[0])
  # pre_images = image_iter.next()
  # # now apply global contrast normalization
  # pre_images = global_contrast_normalization(images)
  # # now subtract mean and divide by std.
  # if (min_ == None) or (max_ == None):
  #   min_ = np.min(pre_images)
  #   max_ = np.max(pre_images)
  # pre_images = data_centre_zero(pre_images, min_, max_)
  # # pre_images, min_, max_ = data_normalise(pre_images, min_, max_)
  # print('Stats after preprocessing')
  # print('max = {}, min = {}, mean = {}, std = {}'.format(np.max(pre_images),
  #                                                        np.min(pre_images),
  #                                                        np.mean(pre_images),
  #                                                        np.std(pre_images)))
  # return pre_images, one_hot(np.array(labels), 10), image_dg, min_, max_


def load_svhn(batch_size):
  """load in preprocessed SVHN data_class

  refer to `load_svhn_and_preprocess` function for details on
  preprocessing
  """
  num_classes = 10
  svhn_dir = os.path.join(os.environ['DATA_PATH'], 'svhn_preprocessed')
  x_train = np.load(os.path.join(svhn_dir, 'x_train.npy'))
  x_test = np.load(os.path.join(svhn_dir, 'x_test.npy'))
  y_train = np.load(os.path.join(svhn_dir, 'y_train.npy'))
  y_test = np.load(os.path.join(svhn_dir, 'y_test.npy'))
  # load in the original data
  orig_svhn_dir = os.path.join(os.environ['DATA_PATH'], 'svhn')
  x_test_orig = np.moveaxis(
      loadmat(os.path.join(orig_svhn_dir, 'test_32x32.mat'))['X'], 3, 0)
  plt.figure()
  plt.imshow(x_test_orig[0, ...].reshape(32, 32, 3))
  plt.savefig('test.png', cmap=None)

  ds_train_preprocessed = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  ds_test_preprocessed = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  # load in the original data
  ds_test = tf.data.Dataset.from_tensor_slices((x_test_orig, y_test))
  # shuffle the training data
  # data set is small enough for us to shuffle everything
  ds_train_preprocessed = ds_train_preprocessed.shuffle(
      ds_train_preprocessed.cardinality())
  # now will batch the data_class
  ds_train_preprocessed = ds_train_preprocessed.batch(batch_size)
  ds_test_preprocessed = ds_test_preprocessed.batch(batch_size)
  ds_test = ds_test.batch(batch_size)
  # now repeat the training data and then turn into an iter
  ds_train_preprocessed = ds_train_preprocessed.repeat()
  # create a data_class class to save it all
  dimension_dict = {
      'in_dim': 3072,
      'out_dim': num_classes,
      'in_height': 32,
      'in_width': 32,
      'in_channels': 3
  }
  # creating the label_dict
  label_dict = {
      0: 0,
      1: 1,
      2: 2,
      3: 3,
      4: 4,
      5: 5,
      6: 6,
      7: 7,
      8: 8,
      9: 9
  }
  # now create the dictionary to hold the dimensions of our data
  dimension_dict = {
      'in_dim': 3072,
      'out_dim': 10,
      'in_height': 32,
      'in_width': 32,
      'in_channels': 3,
      'dataset_size': 73257
  }
  # now putting it in a class for all the data
  data_class = Data(train=ds_train_preprocessed,
                    test=ds_test_preprocessed,
                    test_orig=ds_test,
                    label_dict=label_dict,
                    dimension_dict=dimension_dict)
  return data_class



def load_svhn_and_preprocess(batch_size):
  """load in svhn data and performs preprocessing

  The SVHN data_class is in a Matlab format.
  Need to load it in using scipy.io.loadmat
  When loading it in, will be a dict with items 'X' for the input
  images and 'y' for the labels.
  """
  print('WARNING!, have not implemented validation yet')
  print('Currently just using validation set as test set')
  data_class = Data()
  svhn_dir = os.path.join(os.environ['DATA_PATH'], 'svhn')
  # load in the SVHN data into dict objects
  svhn_train = loadmat(os.path.join(svhn_dir, 'train_32x32.mat'))
  svhn_test = loadmat(os.path.join(svhn_dir, 'test_32x32.mat'))
  # extract the input data and preprocess
  data_class.x_train = svhn_preprocess(svhn_train['X'])
  data_class.x_test = svhn_preprocess(svhn_test['X'])
  # extract the label data and one-hot encode it
  # included minus one as svhn data_class is labelled from [1, 10]
  # want to convert to [0, 9]
  # currently zero is stored as class 10, so will just replace
  # all the entries with a 10 as a zero
  y_train = svhn_train['y']
  y_test = svhn_test['y']
  y_train[y_train == 10] = 0
  y_test[y_test == 10] = 0
  data_class.y_train = one_hot(y_train, 10)
  data_class.y_test = one_hot(y_test, 10)
  # setting validation data to just be the same as the test data
  data_class.x_val = data_class.x_test
  data_class.y_val = data_class.y_test
  # creating the label_dict
  data_class.label_dict = {
      0: 0,
      1: 1,
      2: 2,
      3: 3,
      4: 4,
      5: 5,
      6: 6,
      7: 7,
      8: 8,
      9: 9
  }
  # now create the dictionary to hold the dimensions of our data
  data_class.dimension_dict = {
      'in_dim': 3072,
      'out_dim': 10,
      'in_height': 32,
      'in_width': 32,
      'in_channels': 3,
      'dataset_size': 73257
  }
  np.save(
      os.path.join(os.environ['DATA_PATH'], 'svhn_preprocessed', 'x_train.npy'),
      data_class.x_train)
  np.save(
      os.path.join(os.environ['DATA_PATH'], 'svhn_preprocessed', 'y_train.npy'),
      data_class.y_train)
  np.save(
      os.path.join(os.environ['DATA_PATH'], 'svhn_preprocessed', 'x_test.npy'),
      data_class.x_test)
  np.save(
      os.path.join(os.environ['DATA_PATH'], 'svhn_preprocessed', 'y_test.npy'),
      data_class.y_test)
  np.save(
      os.path.join(os.environ['DATA_PATH'], 'svhn_preprocessed', 'x_val.npy'),
      data_class.x_val)
  np.save(
      os.path.join(os.environ['DATA_PATH'], 'svhn_preprocessed', 'y_val.npy'),
      data_class.y_val)

  return data_class


def svhn_preprocess(svhn):
  """preprocessing of the street view house numbers data_class

  First reshape to correct dimensions and normalise.
  Then, further preprocessing is done using local contrast normalization
  in accordance with [1, 2].

  Args:
    svhn (np.array):
      street view house numbers data_class
  Returns:
    preprocessed svhn images

  References:
    [1] Sermanet, P., Chintala, S. and LeCun, Y., Convolutional Neural Networks
    Applied toHouse Numbers Digit Classification.
    [2] Zeiler, M. and Fergus, R., Stochastic Pooling for Regularization of Deep
    Convolutional Neural Networks
  """
  # swap the end axis to make the image index be the first dimension.
  # ie. go from [H, W, C, idx] to [idx, H, W, C]
  svhn = np.moveaxis(svhn, 3, 0)
  # now perform local contrast normalization
  svhn_normalized = local_contrast_normalization(svhn.astype(np.float32), 7,
                                                 True)
  # apply global contrast normalization
  svhn_normalized = global_contrast_normalization(svhn_normalized)
  # now need to scale back to range [0, 1] so can convert back
  svhn_normalized = ((svhn_normalized - np.min(svhn_normalized)) /
                     (np.max(svhn_normalized) - np.min(svhn_normalized)))
  # just saving a test figure
  svhn = svhn
  plt.figure()
  plt.imshow(255 * svhn[0, ...].reshape([32, 32, 3]).astype(np.uint8))
  plt.savefig('./test.png')
  print('saved the figure')
  return svhn


def convert_color_space(data, conversion_code):
  """convert batch array images to new color space

  Args:
    data (np.array):
      4D array of images, with first dimension being batch
    conversion_code (opencv color code):
      Code to tell us how to perform the conversion (ie. cv2.COLOR_RGB2YUV)

  Returns:
    4D array with all images converted
  """
  # iterate over each image and convert to new color space
  print(data.shape)
  for i in range(0, data.shape[0]):
    data[i, ...] = cv2.cvtColor(data[i, ...].reshape(data.shape[1:]),
                                conversion_code)
  return data


def global_contrast_normalization(data, s=1, lmda=10, epsilon=1e-8):
  """ from
  https://datascience.stackexchange.com/questions/15110/how-to-implement-global-contrast-normalization-in-python"""
  for img_idx in range(0, data.shape[0]):
    for channel_idx in range(0, data.shape[3]):
      X_average = np.mean(data[img_idx, :, :, channel_idx])
      X = data[img_idx, :, :, channel_idx] - X_average
      # `su` is here the mean, instead of the sum
      contrast = np.sqrt(lmda + np.mean(X**2))
      data[img_idx, :, :, channel_idx] = s * X / np.maximum(contrast, epsilon)
  return data


def local_contrast_normalization(data, radius, use_divisor, threshold=1e-6):
  """ Applies LeCun local contrast normalization [1]

  Original code taken from [2] and modified for tensorflow usage

  Args:
    data (np.array):
      4D image data in RGB format, and first dim being batch
    radius (int)
      determines size of Gaussian filter patch
    use_divisor (bool):
      whether or not to apply divisive normalization
    threshold (float):
      the threshold will be used to avoid division by zeros

  Returns:
    4D array with local contrast normalization applied to each image

  References:
    [1] Jarret, K. etal, What is the Best Multi-Stage Architecture for
        Object Recognition?
    [2] https://github.com/jostosh/theano_utils/blob/master/lcn.py
  """
  # Get Gaussian filter
  filters = gaussian_filter(radius, 3)
  # Compute the Guassian weighted average by means of convolution
  convout = tf.nn.conv2d(data, filters, 1, "SAME")
  # subract from the original data
  centered = data - convout
  # Boolean marks whether or not to perform divisive step
  if use_divisor:
    # Note that the local variances can be computed by using the centered_X
    # tensor. If we convolve this with the mean filter, that should give us
    # the variance at each point. We simply take the square root to get our
    # denominator
    # Compute variances
    sum_sqr = tf.nn.conv2d(tf.math.square(centered), filters, 1, "SAME")
    # Take square root to get local standard deviation
    denom = tf.math.sqrt(sum_sqr)
    # find the mean of each image for each channel
    per_img_mean = np.mean(denom, axis=(1, 2))
    # reshape the mean to go across all pixels so can compare
    per_img_mean = np.repeat(per_img_mean[..., np.newaxis], 32, axis=2)
    per_img_mean = np.repeat(per_img_mean[..., np.newaxis], 32, axis=3)
    # now swap the axis so the channel axis is on the end again
    per_img_mean = np.moveaxis(per_img_mean, 1, 3)
    divisor = tf.math.maximum(
        per_img_mean, tf.reshape(denom, [-1, data.shape[1], data.shape[2], 3]))
    # Divisise step
    new_X = centered / tf.math.maximum(divisor, threshold)
  else:
    new_X = centered
  return np.array(new_X)


def gaussian_filter(radius, channels):
  """create a Gaussian filter to be applied to a batch of images

  Args:
    radius (int):
      square size of rhe kernel
    channels (int):
      the number of channels of Gaussian kernels

  Returns:
    Gaussian kernel of size [1, radius, radius, channels]
  """
  # creating a 2d Gaussian
  x, y = np.meshgrid(np.linspace(-1, 1, radius), np.linspace(-1, 1, radius))
  d = np.sqrt(x * x + y * y)
  sigma, mu = 1.0, 0.0
  gaussian_2d = np.exp(-((d - mu)**2 / (2.0 * sigma**2)))
  print(gaussian_2d.shape)
  # repeat for all the channels
  gaussian_2d = gaussian_2d.reshape([radius, radius, 1])
  gaussian_3d = np.repeat(gaussian_2d[..., np.newaxis], channels, axis=3)
  print('here')
  print(gaussian_3d.shape)
  # then reshape to add a batch dimension of one, then return
  return gaussian_3d  #reshape([1, *gaussian_3d.shape])


def load_test_a():
  """Simple function used during testing
    will just be a parabola for small number of points
       f(x) = x^2   for x in [-2, 2)
  """
  n = 10
  n_s = 10
  data_class = Data()
  data_class.x_train = np.linspace(-2, 2, n).reshape(-1, 1)
  data_class.x_test = np.copy(data_class.x_train)
  data_class.y_train = data_class.x_train**2.0
  data_class.y_test = np.copy(data_class.y_train)
  data_class.x_val = data_class.x_train**2.0
  data_class.y_val = np.copy(data_class.y_train)
  data_class.dimension_dict = {
      'in_dim': 1,
      'out_dim': 1,
      'in_width': 1,
      'in_height': 1,
      'in_channels': 1
  }
  return data_class


def load_test_b():
  """Simple function used during testing
  Mostly for testing event rate calculations, can compare
  directly with analytic solution for simple gaussian model
  """
  n = 1
  n_s = 1
  data_class = Data()
  data_class.x_train = np.array(1.0).reshape(1, 1)
  data_class.x_test = data_class.x_train
  data_class.y_train = np.array(1.5).reshape(1, 1)
  data_class.y_test = data_class.y_train
  data_class.x_val = data_class.x_train
  data_class.y_val = data_class.y_train
  data_class.dimension_dict = {
      'in_dim': 1,
      'out_dim': 1,
      'in_width': 1,
      'in_height': 1,
      'in_channels': 1
  }
  return data_class


def load_linear():
  """Simple function used during testing
    will just be a parabola for small number of points
       f(x) = x^2   for x in [-2, 2)
  """
  n = 20
  n_s = 20
  data_class = Data()
  data_class.x_train = np.linspace(0, 5, n).reshape(-1, 1)
  data_class.x_test = np.copy(data_class.x_train)
  data_class.y_train = 2.0 * data_class.x_train + 1.0 + np.random.randn(
      *data_class.x_train.shape)
  data_class.y_test = 2.0 * data_class.x_train + 1.0 + np.random.randn(
      *data_class.x_train.shape)
  data_class.x_val = np.copy(data_class.x_train)
  data_class.y_val = 2.0 * data_class.x_train + 1.0 + np.random.randn(
      *data_class.x_train.shape)
  data_class.dimension_dict = {
      'in_dim': 1,
      'out_dim': 1,
      'in_width': 1,
      'in_height': 1,
      'in_channels': 1
  }
  return data_class


def load_linear_norm():
  """Simple function used during testing
    will just be a parabola for small number of points
       f(x) = x^2   for x in [-2, 2)
  """
  n = 20
  n_s = 20
  data_class = load_linear()
  # will now scale it down

  return data_class


def batch_split(input_, output_, num_batches):
  """Split data into batches"""
  x_train = np.array_split(input_, num_batches)
  y_train = np.array_split(output_, num_batches)
  batch_size = y_train[0].size
  return x_train, y_train, batch_size


def save_model(sess, saver, epoch, model_name='model', model_dir='model'):
  """saves weights at current epoch"""
  # check the directory for where we are going to save exists
  # and if not, lets bloody make one
  check_or_mkdir(model_dir)
  saver.save(sess, os.path.join(model_dir, model_name), global_step=epoch)


def check_or_mkdir(arg):
  """Will make a directory if doesn't exist"""
  if not os.path.isdir(arg):
    os.mkdir(arg)


def var_dict_to_list(var_dict_list):
  """
  var_dict_to_list()

  Description:
  Will conver a list of dicts in the gradient


  var_list: list[dict{Tensors}] ::default=None
    list of dicts of trainable tensors in the graph
    One dict element in list for each layer
    eg. [{'w_sigma':w0_s, 'w_mu':w0_m},
    {'w_sigma':w1_s, 'w_mu':w1_m}, ... ]

  Returns:
    list[iterables]:

  """
  raise (NotImplementedError())


def convert_split(tvn_split, num_samples):
  """
  convert_split()

  Description:
  Will get the percentage values for split as specified
  by the input arg

  Args:
  tvn_split: (str)
    train_val_test_split
    command line arg that specifies the format for how we
    want to split the data (in percentages)
    ie. split = 80-10-10 for 80% test, 10% val and test
  num_samples: (int)
    the number of samples in this data set
  Returns:
    int(train_percent), int(val_percent), int(test_percent)
  """

  #seperate the input string by the slash locations
  str_split = tvn_split.split('_')
  #now check that 3 args were supplied
  if (len(str_split) != 3):
    raise (ValueError(
        'Incorrect data split arg supplied: {}'.format(tvn_split)))
  #convert the split vals to strings
  train_split = int(str_split[0])
  val_split = int(str_split[1])
  test_split = int(str_split[2])
  #now check that they all sum up to 100
  #if not all of the data is used, print a warning
  if ((train_split + val_split + test_split) < 100):
    print('WARNING, not all data used for experiment')
    print('train = {}%, val = {}%, test = {}%'.format(train_split, val_split,
                                                      test_split))
  #if more than 100% data supplied, raise an exception
  elif ((train_split + val_split + test_split) > 100):
    raise (ValueError(
        ('Invalid split provided for data '
         'does not sum to 100% \n'
         'train = {}%, val = {}%, test = {}%'.format(train_split, val_split,
                                                     test_split))))
  #now change the percentage values to number of samples
  #from the data set
  train_split = np.int(num_samples * train_split / 100)
  val_split = np.int(num_samples * val_split / 100)
  test_split = np.int(num_samples * test_split / 100)
  return train_split, val_split, test_split


def data_centre_zero(data, min_=None, max_=None):
  """centre data around zero"""
  if (min_ == None) or (max_ == None):
    min_ = tf.math.reduce_min(data)
    max_ = tf.math.reduce_max(data)
  # put in range of [0, 1]
  data = (data + tf.abs(min_)) / (max_ + tf.abs(min_))
  # now scale to range [-1, 1]
  return (data * 2.0) - 1.0  #, min_, max_


def data_normalise(data, mean_=None, std_=None):
  """centre data around zero"""
  if (mean_ is None) or (std_ is None):
    mean_ = np.mean(data, axis=0)
    std_ = np.max(data, axis=0)
  # put in range of [0, 1]
  data = (data - mean_) / std_
  # now scale to range [-1, 1]
  return data, mean_, std_
