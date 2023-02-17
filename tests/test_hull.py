import numpy as np
import tensorflow as tf
from tbnn.pdmp.hull_tf import compute_hulls, compute_domain_range, envelope, sample_poisson_thinning, integrated_envelope

from tbnn.pdmp import arspy_hull_raw
import matplotlib.pyplot as plt

if __name__ == '__main__':
  # create some dummy data
  def gaussian(x, mu=3.0, sigma=1.0):
    Z = 1 / np.sqrt(2 * np.pi * sigma**2.0)
    exponent = - (x - mu)**2.0 / (2 * sigma**2.0)
    return Z * np.exp(exponent)

  # t = np.linspace(0, 4 * np.pi, 20).astype(np.float32)
  # y = np.sin(t * np.pi / 16.0 )
  t = np.linspace(0.1, 6, 15).astype(np.float32)
  y = np.log(gaussian(t))
  t = tf.convert_to_tensor(t)
  y = tf.convert_to_tensor(y)
  m_array, b_array, left_array, right_array = compute_hulls(
    t, y, domain=[0.0, 10.0])
  print('m array = {}'.format(m_array.stack()))

  (integrated_constant_array,
   integrated_range_lower,
   integrated_range_upper,
   inverse_integrated_domain_lower,
   inverse_integrated_domain_upper,
   m_tensor, b_tensor,
   left_tensor, right_tensor) = compute_domain_range(m_array,
                                                     b_array,
                                                     left_array,
                                                     right_array)

  print(integrated_constant_array)
  print(tf.experimental.numpy.diff(integrated_constant_array))


  #print('left_tensor = {}'.format(left_tensor))
  def print_fn(arg):
    print(f'{arg:.20f}')
  print('right_tensor')
  for i in range(0, right_tensor.shape[0]):
    print_fn(right_tensor[i, 0])
  print('difference between right_tensor and left_tensor')
  for i in range(0, right_tensor.shape[0]):
    print_fn(right_tensor[i, 0] - left_tensor[i, 0])
  print('m_tensor')
  for i in range(0, m_tensor.shape[0]):
    print_fn(m_tensor[i, 0])

  #print('right_array - left_array = {}'.format(right_array - left_array))


  lower_hull, upper_hull = arspy_hull_raw.compute_hulls(t.numpy(), y.numpy(), domain=[0, 10])
  #print(t.numpy())
  #print(upper_hull)
  right_ars = []
  left_ars = []
  print('ARS right values')
  for x in upper_hull:
    print('left = {}'.format(x.left))
    print('m = {}'.format(x.m))
    print('b = {}'.format(x.b))
    print('right = {}\n'.format(x.right))
    left_ars.append(x.left)
    right_ars.append(x.right)


  # print('comparing values')
  # for x in upper_hull:
  #   print('left ars = {}, left tf = {}'.format(x.left, left_array.numpy()[i]))
  #   print('m = {}'.format(x.m))
  #   print('b = {}'.format(x.b))
  #   print('right = {}\n'.format(x.right))
  #   left_ars.append(x.left)
  #   right_ars.append(x.right)

  # print('comparing my tf method with original')
  print('LOWER LIMIT FOR HULL (LEFT)')
  print(left_tensor.numpy().shape)
  print(np.array(left_ars).shape)
  print(left_tensor.numpy().reshape(-1) - np.array(left_ars).reshape(-1))


  print('comparing values')
  i = 0
  for x in upper_hull:
    print('left ars = {}, left tf = {}'.format(x.left, left_tensor[i]))
    print('right ars = {}, right tf = {}'.format(x.right, right_tensor[i]))
    print('m ars = {}, m tf = {}'.format(x.m, m_tensor[i]))
    print('b ars = {}, b tf = {}\n'.format(x.b, b_tensor[i]))
    i += 1

  # print('UPPER LIMIT FOR HULL (RIGHT)')
  # print(right_array.numpy().shape)
  # print(np.array(right_ars).shape)
  # print(right_array.numpy().reshape(-1) - np.array(right_ars).reshape(-1))

  t_full = np.linspace(t[0], t[-1], 100)
  y_full = np.log(gaussian(t_full))
  env = envelope(left_tensor.numpy().reshape(-1), right_tensor.numpy().reshape(-1),
                 m_tensor.numpy().reshape(-1), b_tensor.numpy().reshape(-1),
                 t_full)
  plt.figure()
  plt.plot(t_full, y_full)
  plt.plot(t_full, env, 'r')
  plt.scatter(t, y)
  plt.savefig('test.png')


  int_env = integrated_envelope(left_tensor.numpy().reshape(-1),
                                right_tensor.numpy().reshape(-1),
                                m_tensor.numpy().reshape(-1),
                                b_tensor.numpy().reshape(-1),
                                integrated_constant_array.numpy().reshape(-1),
                                t_full)
  print(integrated_constant_array)
  plt.figure()
  plt.plot(t_full, int_env)
  plt.plot(t_full, np.cumsum(y_full) * (t_full[1] - t_full[0]), 'r--')
  plt.savefig('test_int.png')
  print(left_tensor.numpy().reshape(-1))
  print(right_tensor.numpy().reshape(-1))
  print(m_tensor.numpy().reshape(-1))
  print(b_tensor.numpy().reshape(-1))
  print(integrated_constant_array.numpy().reshape(-1))


  m_ars = np.array([x.m for x in upper_hull]).reshape(-1)
  b_ars = np.array([x.b for x in upper_hull]).reshape(-1)
  left_ars = np.array([x.left for x in upper_hull]).reshape(-1)
  right_ars = np.array([x.right for x in upper_hull]).reshape(-1)

  env_ars = envelope(left_ars, right_ars, m_ars, b_ars, t_full)
  plt.figure()
  plt.plot(t_full, y_full)
  plt.plot(t_full, env_ars, 'r')
  plt.scatter(t, y)
  plt.savefig('test_ars.png')


  print(m_tensor.numpy().reshape(-1))
  print(left_tensor.numpy().reshape(-1))

  # now test it with a gaussian dist
  @tf.function
  def tf_wrapper(hull_samples, time):
    return sample_poisson_thinning(hull_samples, time)

  print('testing in eager mode')
  sample = sample_poisson_thinning(gaussian(t), t)
  #print('testing in graph mode')
  # sample = tf_wrapper(tf.convert_to_tensor(gaussian(t)),
  #                     tf.convert_to_tensor(t))
  print('sample = {}'.format(sample))


  # lets look at some inverse functions
  # F1 in range [0, 1)
  # m1 = 0.5                       #
  # b1 = 0.5
  # m2 =
  # b2 = 1.0
  # F1
