"""
Module to display summary info during training
and show testing results.
"""
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, mean_squared_error

from tbnn.utils import utils
import tensorflow_datasets as tfds

class PlotCallback(keras.callbacks.Callback):

  def __init__(self, model, out_dir, posterior_type, dataset,
               display_rate, num_samples):
    self.out_dir = out_dir
    self.model = model
    self.posterior_type = posterior_type
    self.num_samples = num_samples
    self.display_rate = display_rate
    self.dataset = dataset
    super(PlotCallback, self).__init__()

  def on_epoch_end(self, epoch, logs={}):
    if(epoch % self.display_rate == 0):
      plot_predictive_posterior(self.model,
                                self.dataset.x_train,
                                self.dataset.y_train,
                                self.dataset.x_test,
                                self.dataset.y_test,
                                self.posterior_type,
                                self.out_dir,
                                self.num_samples,
                                name='pred_{}'.format(epoch))



def display_iter(epoch, cost, outputs, labels, predict_type):
  """display the current cost and output of the prev example

  Args:
  epoch (int):
    current epoch
  cost (tensor):
    Cost operation for network
  outputs (tensor):
    output of the network
  labels (np.array):
    Label vector for a single sample from the previous batch
    (If a complete batch is passed, only a single variable
     will be evaluated and printed for the example)
  predict_type (str):
    type of prediction being done
    either classify or regression
  """
  print("Epoch:", '%04d' % (epoch + 1),
        "cost=", "{:.9f}".format(cost))
  if(predict_type == 'classify'):
    # now just get the last input to show as an example
    test = outputs[0:1, :]
    label = labels[0:1, :]
    # compute softmax of test output
    test_soft = np.exp(test - np.max(test)) / (
      np.sum(np.exp(test - np.max(test))))
    np.set_printoptions(precision=2)
    print_input_compare(label, test_soft)



def print_input_compare(label, pred):
  """ Will just fancy print evaluated example"""
  print('{:*^80}'.format('  Example From Previous Batch  '))
  class_str = '{:<10}'.format('Class')
  label_str = '{:<10}'.format('Labels')
  pred_str = '{:<10}'.format('Preds.')
  for ii in range(0, label.shape[1]):    # pylint: disable=invalid-name
    class_str += '{:^7}'.format(ii)
    label_str += '{:<7.3f}'.format(np.float(label[0, ii]))
    pred_str += '{:<7.3f}'.format(pred[0, ii])
  print(class_str)
  print(label_str)
  print(pred_str)
  print('\n')



def display_all_performance(model, dataset):
  """display the current performance on for all parts of dataset
  Args:
    model (keras.model):
      neural network model to evaluate
    dataset (Dataset):
      Dataset object containing data to work with

  Returns:
    NA
  """
  # performance for training set
  display_train_accuracy(model, dataset)
  # performance for validation set
  display_validation_accuracy(model, dataset)
  # performance for testing set
  display_test_accuracy(model, dataset)

  

def display_train_accuracy(model, dataset):
  """display the current performance on the training set
  Args:
    model (keras.model):
      neural network model to evaluate
    dataset (Dataset):
      Dataset object containing data to work with

  Returns:
    NA
  """
  _display_accuracy(model, dataset.x_train, dataset.y_train, "Train")

  

def display_validation_accuracy(model, dataset):
  """display the current performance on the validation set
  Args:
    model (keras.model):
      neural network model to evaluate
    dataset (Dataset):
      Dataset object containing data to work with

  Returns:
    NA
  """
  _display_accuracy(model, dataset.x_val, dataset.y_val, "Validation")

  

def display_test_accuracy(model, dataset):
  """display the current performance on the test set
  Args:
    model (keras.model):
      neural network model to evaluate
    dataset (Dataset):
      Dataset object containing data to work with

  Returns:
    NA
  """
  _display_accuracy(model, dataset.x_test, dataset.y_test, "Test")

    

def _display_accuracy(model, input_, labels, data_category):
  """ Display prediction accuracy for the data supplied

  Is designed to work for both regression and classification. For
  regression will compute the mean squared error, and classification is
  just the accuracy.

  Is a private function that is intended to be called by the
  public display_train/validation/test functions within this module.

  Args:
    model (keras.Model):
      neural network model to evaluate
   input_ (np.array):
      input data to be passed through the model
    labels (np.array):
      true label data corresponding to the supplied input_
    data_category (str):
      tells us which component of the total data set we are working
      with. Valid values are ("test", "validation", "train")
  Returns:
    NA
  """
  if(model.predict_type == 'regression'):
    accuracy = regression_accuracy(model, input_, labels)
    print('{} MSE = {}'.format(data_category, accuracy))
  # otherwise we are performing classification
  else:
    accuracy = classification_accuracy(model, input_, labels)
    print('{} classification accuracy = {}'.format(data_category, accuracy))

    

def classification_accuracy(model, input_, labels):
  """get classification accuracy

  Args:
    model (keras.Model):
      neural network model to evaluate
   input_ (np.array):
      input data to be passed through the model
    labels (np.array):
      true label data corresponding to the supplied input_

  Returns:
    classification accuracy
  """
  classification_array, correct_labels = classify(model, input_, labels)
  # now find the accuracy
  accuracy = accuracy_score(correct_labels,
                            classification_array,
                            normalize=True)
  return accuracy



def regression_accuracy(model, input_, labels):
  """get regression accuracy using mean squared error

  Args:
    model (keras.Model):
      neural network model to evaluate
   input_ (np.array):
      input data to be passed through the model
    labels (np.array):
      true label data corresponding to the supplied input_

  Returns:
    Mean squared error
  """
  # reshape the input data to have a batch dimension
  input_batch = input_.reshape([1, *input_.shape])
  # get the predictive output
  output = model.call(input_batch)
  # get the MSE
  mse = mean_squared_error(labels, output)
  return mse



def classify(model, input_, labels):
  """Helper func. classify all the examples

  Args:
  input_ (np.array):
    input data to evaluate
  labels (np.array):
    true labels corresponding to the inputs

  Returns:
  classification_array (np.array):
    list of what the classifier actually predicted
  correct_prediction_eval (np.array):
    list of what the correct prediction labels should be
  """
  data = tf.data.Dataset.from_tensor_slices(
    (input_, labels))
  # batch testing data based on prediction type and the no. test samples
  # send the 2 times the batch size across
  # if the GPU can handle the original batch size for training,
  # then shouldn't have an issue with double for testing
  data = data.batch(np.int(2 * model.batch_size))
  data = tfds.as_numpy(data)
  # forming lists to save the output
  classification_list = []
  label_list = []
  # classify each batch and store the results
  num_samples = 100
  for input_batch, labels_batch in data:
    batch_classification = np.zeros([num_samples, *labels_batch.shape])
    # sample 100 times and then find the mean to get classification
    for i in range(0, num_samples):
      output_ = model(input_batch)
      batch_classification[i, ...] = output_
    mean_output = np.mean(batch_classification, axis=0)
    #print('output_ = {}'.format(output_))
    classification_list.extend(np.argmax(mean_output, axis=1))
    label_list.extend(np.argmax(labels_batch, axis=1))
  classification_array = np.array(classification_list)
  correct_prediction = np.array(label_list)
  #Wprint('classification_array = {}'.format(classification_array))
  #print('correct_prediction   = {}'.format(correct_prediction))
  return classification_array, correct_prediction



def plot_output(model, dataset, save_dir, plot_name):
  """ Plot the output of our networks

  if we are doing classification, plot some misclassified examples
  if we are doing regression, plot the distribution on the output
  Will call the specific function here to deal with the regression and
  classification case

  Args:
  sess: tf.Session
  Tensorflow session passed from main script
  output_: tf.Operation
    Tensorflow operation for the output of the network
  num_classes: int
    Number of classes in this problem
  dims: list(ints)
    true dimensions of input data
  input_p: tf.placeholder
    Placeholder used for feeding input data in graph
  label_p: tf.placeholder
    Placeholder used for feeding label data in graph
  x_train: np.array
    Input array
  y_train: np.array
    Label vector
  x_test: np.array
    test data that contains the ground truth
  y_test: np.array
    test output that contains the ground truth
  label_dict: dict
    dictionary that maps from numeric label values to string names
  posterior_type: string
    Type of posterior approx used
  save_dir: string
    path to where root directory for images will be
  Returns: N/A
  """

  # if we are performing regression, 1-d output
  if(len(dataset.label_dict) <= 1):
    plot_regression(model, dataset.x_train, dataset.y_train,
                    dataset.x_test, dataset.y_test,
                    save_dir,
                    name=plot_name)
  # otherwise we are doing classification
  else:
    plot_misclassified(model, dataset, save_dir,
                       name=plot_name)

    

def plot_misclassified(model, dataset, save_dir,
                       n_samples=100, name='pred.eps'):
  """Will plot misclassified samples

  Will work for both probabilistic and point estimate networks.
  If using point estimate, will set n_samples to 1 and won't try to
  find credible intervals. If we are using BNN, will approximate
  credible intervals and will plot them.

  Plots saved in tree structure of:

  save_dir
  |-- one
  |__ |-- 1_1.png
  |__ |-- 1_2.png
  |-- two
      |-- 2_1.png
  etc.

  Args:
  sess: tf.Session
    Tensorflow session passed from main script
  output_: tf.Operation
    Tensorflow operation for the output of the network
  num_classes: int
    Number of classes in this problem
  input_p: tf.placeholder
    Placeholder used for feeding input data in graph
  label_p: tf.placeholder
    Placeholder used for feeding label data in graph
  label_dict: dict{}
    dictionary linking class names to numeric value
  input_: np.array
    Input array
  label: np.array
    Label vector
  save_dir: string
    path to where root directory for images will be

  Returns: N/A
  """
  # if the number of channels is equal to one, will drop it from the dimensions,
  # as we don't need it and it will corrupt the grayscale plotting
  if(dataset.dimension_dict['in_channels'] == 1):
    plt_dims = [dataset.dimension_dict['in_height'],
                dataset.dimension_dict['in_width']]
  else:
    plt_dims = [dataset.dimension_dict['in_height'],
                dataset.dimension_dict['in_width'],
                dataset.dimension_dict['in_channels']]
  print("plotting misclassified samples")
  # check to see if save directory exists
  # if it doesn't, this function will make it
  utils.check_or_mkdir(save_dir)
  # find the misclassified images
  preds, correct_preds = classify(model, dataset.x_test, dataset.y_test)
  if(model.posterior_type != "point"):
    plot_misclassified_posterior(model, preds, correct_preds, dataset,
                                 save_dir, plt_dims,
                                 n_samples=100, name='pred.eps')
    plot_conv_layer_uncertainty(model, preds, correct_preds, dataset,
                                save_dir, plt_dims)
  else:
    plot_misclassified_point(sess, preds, correct_preds, outputs, dims, num_classes,
                             label_dict, epoch, x_test, y_test,
                             posterior_type, save_dir)

    

def plot_misclassified_posterior(model, preds, correct_preds, dataset,
                                 save_dir, plt_dims,
                                 n_samples=100, name='pred.eps'):
  """Plot misclassified with credible intervals"""
  num_classes = len(dataset.label_dict)
  # create a figure
  plt.figure()
  # iterate over all classes
  for label_i in range(0, num_classes):
    # check to see if a directory exists. If it doesn't, create it.
    utils.check_or_mkdir(os.path.join(save_dir, str(label_i)))
    # go through all the examples and see if we classified
    # it correctly or not
    locs = np.where(np.logical_and(preds != correct_preds,
                                   dataset.y_test[:, label_i] == 1))
    # perform Monte Carlo Integration to approximate predictive posterior
    pred_eval = np.zeros([n_samples, locs[0].size, num_classes])
    images = dataset.x_test[locs[0], ...]
    for ii in range(n_samples):   # pylint: disable=invalid-name
      pred_eval[ii, ...] = tf.nn.softmax(model.call(images))
    # now get the mean and credible intervals for these images and plot them
    # creating a counter variable for each individual misclassified image
    count = 0
    for im_idx in range(0, pred_eval.shape[1]):
      # approximate the mean and credible intervals
      cred_ints = mc_credible_interval(
        pred_eval[:, im_idx, :].reshape([-1, num_classes]),
        np.array([0.025, 0.975]))
      pred_mean = np.mean(pred_eval[:, im_idx, :], axis=0)
      # PLOTTING
      # formatting the credible intervals into what is needed to be plotted
      # with pyplot.errorbar()
      cred_plot = np.array([pred_mean - cred_ints[0, :],
                            cred_ints[1, :] - pred_mean])
      # reshape it to correct dims
      cred_plot = cred_plot.reshape(2, num_classes)
      #now lets plot it and save it
      plt.subplot(2, 1, 1)
      plt.imshow(images[im_idx].reshape(plt_dims), cmap='gray')
      plt.axis('off')
      plt.subplot(2, 1, 2)
      plt.errorbar(np.linspace(0, pred_mean.size - 1, pred_mean.size),
                   pred_mean.ravel(), yerr=cred_plot, fmt='o')
      plt.xlim(-1, num_classes)
      plt.ylim(-0.1, 1.1)
      plt.xticks(range(num_classes),
                 dataset.label_dict.values(),
                 size='small',
                 rotation='vertical')
      plt.xlabel("class")
      plt.ylabel("Predicted Probability\nwith 95% CI")
      #plt.savefig(os.path.join(save_dir, str(label_i),
      #                         "{}_{}.png".format(label_i, count)))
      plt.savefig(os.path.join(save_dir, str(label_i),
                               "{}_{}.eps".format(label_i, count)),
                  format='eps', bbox_inches="tight")
      plt.clf()
      #increment counter
      count += 1

      


def plot_misclassified_mcmc_posterior(model, chain, X_test, y_test,
                                      save_dir, plt_dims,
                                      n_samples=100, name='pred.eps'):
  """Plot misclassified with credible intervals"""
  num_classes = 10
  # create a figure
  plt.figure()
  # iterate over all classes
  for label_i in range(0, num_classes):
    # check to see if a directory exists. If it doesn't, create it.
    utils.check_or_mkdir(os.path.join(save_dir, str(label_i)))
    pred_eval = np.zeros([n_samples, y_test.shape[0], num_classes])
    images = dataset.x_test[locs[0], ...]
    weights_chain = chain[::2]
    biases_chain = chain[1::2]
    print(weights_chain[0].shape)
    for i in range(num_results - pred_array.shape[0], num_results):
      weights_list = [x[i, ...] for x in weights_chain]
      biases_list = [x[i, ...] for x in biases_chain]
      pred_eval[idx, ...] = pred_forward_pass(model, weights_list,
                                              biases_list, X_test)
      idx +=1
    # now get the mean and credible intervals for these images and plot them
    # creating a counter variable for each individual misclassified image
    count = 0
    for im_idx in range(0, pred_eval.shape[1]):
      # approximate the mean and credible intervals
      cred_ints = mc_credible_interval(
        pred_eval[:, im_idx, :].reshape([-1, num_classes]),
        np.array([0.025, 0.975]))
      pred_mean = np.mean(pred_eval[:, im_idx, :], axis=0)
      # PLOTTING
      # formatting the credible intervals into what is needed to be plotted
      # with pyplot.errorbar()
      cred_plot = np.array([pred_mean - cred_ints[0, :],
                            cred_ints[1, :] - pred_mean])
      # reshape it to correct dims
      cred_plot = cred_plot.reshape(2, num_classes)
      #now lets plot it and save it
      plt.subplot(2, 1, 1)
      plt.imshow(images[im_idx].reshape(plt_dims), cmap='gray')
      plt.axis('off')
      plt.subplot(2, 1, 2)
      plt.errorbar(np.linspace(0, pred_mean.size - 1, pred_mean.size),
                   pred_mean.ravel(), yerr=cred_plot, fmt='o')
      plt.xlim(-1, num_classes)
      plt.ylim(-0.1, 1.1)
      plt.xticks(range(num_classes),
                 dataset.label_dict.values(),
                 size='small',
                 rotation='vertical')
      plt.xlabel("class")
      plt.ylabel("Predicted Probability\nwith 95% CI")
      #plt.savefig(os.path.join(save_dir, str(label_i),
      #                         "{}_{}.png".format(label_i, count)))
      plt.savefig(os.path.join(save_dir, str(label_i),
                               "{}_{}.eps".format(label_i, count)),
                  format='eps', bbox_inches="tight")
      plt.clf()
      #increment counter
      count += 1

      

def plot_conv_layer_uncertainty(model, preds, correct_preds, dataset,
                                save_dir, plt_dims,
                                n_samples=100, name='pred.eps'):
  num_classes = len(dataset.label_dict)
  # create a figure
  plt.figure()
  # iterate over all classes
  for label_i in range(0, num_classes):
    # check to see if a directory exists. If it doesn't, create it.
    utils.check_or_mkdir(os.path.join(save_dir, str(label_i)))
    # find which samples were predicted correctly
    locs = np.where(np.logical_and(preds != correct_preds,
                                   dataset.y_test[:, label_i] == 1))
    # iterate over all the layers
    for layer_idx in range(0, len(model.layer)):
      # if this layer is not a convolutional layer, than skip forward to
      # the next layer
      if 'conv' not in model.layer[layer_idx].name:
        continue
      # perform Monte Carlo Integration to approximate predictive posterior
      pred_eval = []
      images = dataset.x_test[locs[0], ...]
      for _ in range(n_samples):   # pylint: disable=invalid-name
        pred_eval.append(model.layer_call(images, layer_idx))
      # now find the emperical variance for this image
      # stack to create an array of samples, where the last dimension is the
      # dimension for each sample
      # stacked array should be of dimensions
      # [n_samples, batch_size, height, width, channels]
      pred = np.stack(pred_eval)
      # find the variance over all the samples
      # dimensions of var will be [batch_size, height, width, channels]
      var = np.var(pred, axis=0)
      # now save each output as a numpy array so we can plot them later
      for im_idx in range(var.shape[0]):
        # now plot the output
        utils.check_or_mkdir(os.path.join(save_dir, str(label_i), str(im_idx)))
        array_name = "{}.eps".format(model.layer[layer_idx].name)
        fig_path = os.path.join(save_dir, str(label_i), str(im_idx), array_name)
        np.save(fig_path, var[im_idx, ...])

        

def plot_misclassified_point(sess, preds, correct_preds, output_, dims,
                             plt_dims, num_classes, input_p, label_p,
                             label_dict, input_, label, save_dir):
  """Plot misclassified point estimates"""
  plt.figure()
  pred = tf.nn.softmax(outputs, axis=1)
  for label_i in range(0, num_classes):
    # check to see if a directory exists
    utils.check_or_mkdir(os.path.join(save_dir, str(label_i)))
    # go through all the examples and see if we classified
    # it correctly or not
    locs = np.where(
      np.logical_and(preds != correct_preds, y_test[:, label_i] == 1))
    # now lets plot and save all of these images
    count = 0
    for loc in locs[0]:
      # reshape the unpacked image to correct dims
      # numpy reshape will do it correctly for us
      image = x_test[loc, :].reshape([1, dims[0], dims[1], dims[2]])
      # create a Dataset with this image
      ds = tf.data.Dataset.from_tensor_slices(
        (image, y_test[loc, :].reshape(1, -1))).batch(1)
      # now initialise the iterator with this sample
      # initialise the dataset with the single misclassified sample
      sess.run(iterator.make_initializer(ds))
      pred_eval = sess.run(pred)
      #now lets plot it and save it
      plt.subplot(2, 1, 1)
      plt.imshow(image.reshape(plt_dims), cmap='gray')
      plt.axis('off')
      plt.subplot(2, 1, 2)
      plt.scatter(np.linspace(0, pred_eval.size - 1, pred_eval.size),
                  pred_eval.ravel(), marker='o')
      plt.xlim(-1, num_classes)
      plt.ylim(-0.1, 1.1)
      plt.xticks(range(num_classes), label_dict.items(), size='small')
      plt.xlabel("class")
      plt.ylabel("Predicted Point Estimate")
      plt.savefig(os.path.join(save_dir, str(label_i),
                               "{}_{}.png".format(label_i, count)),
                  bbox_inches="tight")
      plt.clf()
      #increment counter
      count += 1

      

def plot_regression(model, x_train, y_train, x_test,
                    y_test, save_dir,
                    n_samples=100, name='pred.eps'):
  """plot regression output and will compare with the true output

  Depending on the type of model being fit, the predictive posterior
  or the predictive point estimates will be plot with this method.

  Args:
  model (nn):
    neural network model
  x_train (np.array):
    array for training data input
  y_train (np.array):
    array for training data labels
  x_test (np.array):
    array for testing data input
  y_test (np.array):
    array for testing data output
  save_dir (str):
    path to output directory to save the figures
  n_samples (int):
    number of samples to generate for predictive posterior
    Only used for probabilistic model
  name (str):
    name to save the plot as

  Returns:
    N/A
  """

  #check to see if save directory exists
  #if it doesn't, this function will make it
  utils.check_or_mkdir(save_dir)
  print(model.posterior_type)
  if((model.posterior_type == 'factorised gaussian') |
     (model.posterior_type == 'factorised bernoulli') |
     (model.posterior_type == 'factorised neuron gaussian')):
    plot_predictive_posterior(model, x_train, y_train,
                              x_test, y_test,
                              save_dir, n_samples, name)
  else:
    raise NotImplementedError('Havent ported to TF2 yet')
    


def plot_predictive_posterior(model, x_train, y_train, x_test,
                              y_test, save_dir,
                              n_samples=100, name='pred.eps'):
  """Will approximate the predictive posterior and plot it

  Args:
    Look at plot_regression() above
  """
  #raise(NotImplementedError())
  #perform Monte Carlo Integration to approximate predictive posterior
  pred_eval = tf.zeros(y_test.shape)
  pred_mean = tf.zeros(y_test.shape)
  pred_cov = tf.zeros([y_test.size, y_test.size])
  print(model)
  print(type(model.predict))
  print(type(x_test))
  for _ in range(n_samples):
    pred_eval = model.call(x_test)
    pred_mean += pred_eval
    pred_cov += tf.matmul(pred_eval, tf.transpose(pred_eval))
  #take average now
  pred_mean = pred_mean / n_samples
  pred_cov = 0.03**2.0 + np.divide(pred_cov, n_samples) - \
             tf.matmul(pred_mean, tf.transpose(pred_mean))
  diag = tf.linalg.diag_part(pred_cov)
  std = tf.math.sqrt(diag)
  # reshaping pred_mean to be a 1d array for when we plot it
  pred_mean = tf.reshape(pred_mean, [-1])
  #as an example, lets plot the predictive posterior distribution contours for
  #some similar classes
  plt.figure()
  #plt.plot(x_test, pred_eval, 'r', label='Sample')
  plt.plot(x_test, y_test, 'b', label='True', alpha=0.2, linewidth=0.5)
  plt.plot(x_test, pred_mean, 'm', label='mean')
  plt.gca().fill_between(x_test.flat,
                         pred_mean - 2 * std,
                         pred_mean + 2 * std,
                         color="#acc1e3")
  plt.gca().fill_between(x_test.flat,
                         pred_mean - 1 * std,
                         pred_mean + 1 * std,
                         color="#cbd9f0")
  plt.scatter(x_train, y_train, marker='o', alpha=0.15,
              s=10, c="#7f7f7f", label='Training Samples')
  plt.xlim([np.min(x_test), np.max(x_test)])
  plt.ylim([np.min(y_test) * 1.25, np.max(y_test) * 1.35])
  # plotting with no axis
  plt.axis('off')
  #plt.axis([np.min(x_test), np.max(x_test),
  #          np.min(y_test)  - 0.3 * np.abs(np.max(y_test)),
  #          np.max(y_test)  + 0.3 * np.abs(np.max(y_test))])
  #plt.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
  plt.savefig(os.path.join(save_dir, name), format='eps', bbox_inches="tight")
  plt.close()

  

def plot_point_predict(input_p, label_p, output_, x_train, y_train,
                       x_test, y_test, save_dir):
  """plots point estimate of output
  Args:
    Look at plot_regression() above
  """
  sess = tf.get_default_session()
  # create a Dataset with the test data
  ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(y_test.size)
  sess.run(iterator.make_initializer(ds))
  for _ in range(n_samples):
    pred_eval = sess.run(outputs)

  plt.figure()
  plt.plot(x_test, pred_eval, 'r', label='Sample')
  plt.plot(x_test, y_test, 'b', label='True', alpha=0.2, linewidth=0.5)
  plt.scatter(x_train, y_train, marker='o', alpha=0.15,
              s=10, c="#7f7f7f", label='Training Samples')
  plt.xlim([np.min(x_test), np.max(x_test)])
  plt.ylim([np.min(y_test) * 1.25, np.max(y_test) * 1.35])
  plt.axis('off')
  plt.savefig(os.path.join(save_dir, 'pred_point.eps'), format='eps',
              bbox_inches="tight")
  plt.close()

  

def display_training_time_info(training_time, epoch_time_list):
  """displays training time info
  mean epoch time and total training time
  """
  #find the mean time and round using ceil operator
  mean_epoch_time = np.ceil(np.mean(epoch_time_list))
  #convert the mean epoch time and training time to a nice time
  #format and print it
  print('Total Training Time: {}'.format(
    datetime.timedelta(seconds=np.ceil(training_time))))
  print('Mean Epoch Time: {}\n\n'.format(
    datetime.timedelta(seconds=mean_epoch_time)))



def display_training_params(args, bnn):
  """Prints some training info to stdout"""
  print("\n\n")
  print("{:*^80}".format(" Training info "))
  print("{:30}: {}".format("Number of Epochs", bnn.num_epochs))
  print("{:30}: {}".format("Number of Batches", bnn.num_batches))
  print("{:30}: {}".format("Batch", bnn.batch_size))
  print("{:30}: {}".format("Learning Rate", bnn.learning_rate))
  print("{:30}: {}".format("Likelihood", bnn.likelihood))
  print("{:30}: {}".format("Loss function", bnn.loss_func))
  print("{:30}: {}".format("Optimiser", bnn.optimiser))
  print("{:30}: {}".format("Data set", args.data))
  print("{:30}: {}".format("Config file", args.config))
  print("{:30}: {}".format("Data split (TR%/V%/TE%)", args.split))
  print("{:30}: {}".format("Tboard Summaries Dir:", args.summary))
  print("{:30}: {}".format("Log dir", args.log))
  print("\n\n")



def mc_credible_interval(smp, quantiles):
  """Approximates confidience intervals based on samples

  Based on methodology discussed in Murphey's book, sec 5.2.2
  and in the PMTK toolbox.
  will order the samples, and use them as empirical approximations
  to quantiles in the predictive posterior.

  Args:
    smp (2D np.array):
      samples from predictive dist. Rows are individual samples
      columns correspond to each class
    quantiles (np.array([lower, upper])):
      1D array with the lower and upper values for the quantiles.
      Ie. if looking for 95% CI, quantiles = np.array([0.025, 0.975])

  Return:
    2D array with lower and upper CI
  """
  # check the quantiles are in the correct format
  # if given in percent and not decimal
  if(quantiles.any() > 1):
    quantiles = quantiles / 100.0

  # sort each of the columns
  cred_ints = np.zeros([2, smp.shape[1]])
  q = np.linspace(0, 1, smp.shape[0]) # pylint: disable=invalid-name
  for ii in range(0, smp.shape[1]):   # pylint: disable=invalid-name
    smp_sorted = np.sort(smp[:, ii])
    # now approximate credible intervals
    prob_func = interp1d(q, smp_sorted)
    cred_ints[:, ii] = prob_func(quantiles)
  return(cred_ints)
