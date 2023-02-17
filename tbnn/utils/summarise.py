"""
Functions that help me add stuff to tensorboard
"""
import tensorflow as tf

  

def summarise_cost_gradients(cost, bnn):
  """Add cost and gradients to tensorboard"""
  #add cost to the tensorboard
  summarise_cost(cost)
  #summarise the gradients
  summarise_gradients(cost, bnn)



def summarise_variables(bnn, writer, step):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  params = [x for x in bnn.trainable_weights]
  with writer.as_default():
    # setting scope for the gradients
    with tf.name_scope('variables'):
      for param in params:
        tf.summary.scalar('{}_mean'.format(param.name),
                          tf.reduce_mean(param), step)
        tf.summary.scalar('{}_min'.format(param.name),
                          tf.reduce_min(param), step)
        tf.summary.scalar('{}_max'.format(param.name),
                          tf.reduce_max(param), step)
        #   tf.summary.scalar('min', tf.reduce_min(var))
        #   tf.summary.scalar('max', tf.reduce_max(var))


  # with tf.name_scope('summaries'):
  #   mean = tf.reduce_mean(var)
  #   tf.summary.scalar('mean', mean)
  #   tf.summary.scalar('min', tf.reduce_min(var))
  #   tf.summary.scalar('max', tf.reduce_max(var))
    #summarising the stddev takes wayyy too long
  #with tf.name_scope('stddev'):
  #  stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
  #  tf.summary.scalar('stddev', stddev)
  #  tf.summary.histogram('histogram', var)


def summarise_cost(cost, writer, step):
  """Add cost variable to tensorboard so we can view it

  Args:
  cost (tf.Variable):
    cost for current batch
  writer (tf.FileWriter):
    object to handle logging for Tensorboard
  step (tf.int64):
    iteration in training
  """
  with writer.as_default():
    with tf.name_scope('Cost'):
      tf.summary.scalar('cost', cost, step=step)

  

def summarise_gradients(bnn, gradients, writer, step):
  """Adds gradient info for trainable variables to tensorboard

  Args:
  bnn (keras.Model):
    network object
  gradients (list(tf.Variable)):
    list of gradient ops
  writer (tf.FileWriter):
    object to handle logging for Tensorboard
  step (tf.int64):
    iteration in training
  """
  param_names = [w.name for w in bnn.trainable_weights]
  with writer.as_default():
    # setting scope for the gradients
    with tf.name_scope('gradients'):
      for i in range(0, len(gradients)):
        mean_grad = tf.reduce_mean(gradients[i])
        tf.summary.scalar('{}_grad'.format(param_names[i]),
                          mean_grad, step=step)
