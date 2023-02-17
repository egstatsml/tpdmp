import numpy as np
import tensorflow as tf
from tbnn.pdmp.hull_tf import compute_hulls, compute_domain_range, envelope, sample_poisson_thinning, integrated_envelope, inverse_integrated_envelope
from tbnn.pdmp.poisson_process import SBPSampler
import matplotlib.pyplot as plt
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

S_hi = np.load(
    '/home/XXXXXX/exp_data/XXXXXX_data/old_pdmp/linear_sbps/time_array_0.npy')
fS_hi = np.load(
    '/home/XXXXXX/exp_data/XXXXXX_data/old_pdmp/linear_sbps/test_array_0.npy')
#S_hi = np.load('/home/XXXXXX/exp_data/XXXXXX_data/old_pdmp/toy_b_small_adaptive/time_array_1.npy')
#fS_hi = np.load('/home/XXXXXX/exp_data/XXXXXX_data/old_pdmp/toy_b_small_adaptive/test_array_1.npy')
fS_hi[fS_hi < 0] = 0.01
sample = np.sort(random.sample(list(np.arange(0, S_hi.size)), 10))
sample = [0, 30, 80, 110, 155, 371, 441, 465, 629, 660, 875]
#sample = np.arange(0, np.round(np.size(fS_hi)),
#                        np.round(np.size(fS_hi) * 0.3)).astype(np.int64)
print(sample)
#sample = np.hstack([[0], sample])
print(sample)
t = tf.convert_to_tensor(S_hi[sample])
y = tf.convert_to_tensor(fS_hi[sample])
m_array, b_array, left_array, right_array = compute_hulls(t,
                                                          y,
                                                          domain=[0.0, 10.0])
print('m array = {}'.format(m_array.stack()))

(integrated_constant_array, integrated_range_lower, integrated_range_upper,
 inverse_integrated_domain_lower, inverse_integrated_domain_upper, m_tensor,
 b_tensor, left_tensor,
 right_tensor) = compute_domain_range(m_array, b_array, left_array, right_array)
env = envelope(left_tensor.numpy().reshape(-1),
               right_tensor.numpy().reshape(-1),
               m_tensor.numpy().reshape(-1),
               b_tensor.numpy().reshape(-1), S_hi)

int_env = integrated_envelope(left_tensor.numpy().reshape(-1),
                              right_tensor.numpy().reshape(-1),
                              m_tensor.numpy().reshape(-1),
                              b_tensor.numpy().reshape(-1),
                              integrated_constant_array.numpy().reshape(-1),
                              S_hi)

inv_S_hi = np.linspace(inverse_integrated_domain_lower[0],
                       inverse_integrated_domain_upper[-1], np.size(S_hi))
inv_int_env = inverse_integrated_envelope(
    inverse_integrated_domain_lower.numpy().reshape(-1),
    inverse_integrated_domain_upper.numpy().reshape(-1),
    m_tensor.numpy().reshape(-1),
    b_tensor.numpy().reshape(-1),
    integrated_constant_array.numpy().reshape(-1), int_env)

plt.figure()
plt.plot(S_hi, fS_hi)
plt.plot(S_hi, env, 'r')
plt.scatter(t, y)
plt.savefig('test_hull_tf.png')

#plt.figure()
#plt.plot(S_hi, int_env)
#plt.plot(inv_S_hi, inv_int_env, 'r')
#plt.savefig('test_hull_int_tf.png')

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(S_hi, int_env)
ax2.plot(int_env, inv_int_env, 'r')
fig.savefig('test_hull_int_tf.png')

print(env)

print(t)

print(integrated_constant_array.shape)
print(left_tensor.shape)
print(inverse_integrated_domain_lower)

print(
    inverse_integrated_envelope(
        inverse_integrated_domain_lower.numpy().reshape(-1),
        inverse_integrated_domain_upper.numpy().reshape(-1),
        m_tensor.numpy().reshape(-1),
        b_tensor.numpy().reshape(-1),
        integrated_constant_array.numpy().reshape(-1), int_env))

print(S_hi)

# now want to get the eSBPS envelop as well
esbps = SBPSampler()
# need to put the envelope and time samples in a TF array

G = tf.TensorArray(tf.float32,
                   size=0,
                   dynamic_size=True,
                   clear_after_read=False)
X = tf.TensorArray(tf.float32,
                   size=0,
                   dynamic_size=True,
                   clear_after_read=False)

for i in range(0, np.size(t)):
  G.write(i, tf.cast(y[i], tf.float32))
  a = tf.cast(tf.reshape([t[i], 1.0], [1, 2]), tf.float32)
  X.write(i, a)
  # X.write(i, np.array([t[i], 1]).astype(np.float32))

beta_mean, beta_cov = esbps.sbps_beta_posterior(G, X, 0)
beta_one = beta_mean[0, 0]
beta_zero = beta_mean[1, 0]
beta_zero = tf.cast(beta_zero, tf.float64)
beta_one = tf.cast(beta_one, tf.float64)
beta_cov = tf.cast(beta_cov, tf.float64)
x_time = tf.reshape(X.stack(), [-1, 2])
x = tf.reshape(x_time[-1, :], (1, 2))
x = tf.cast(x, dtype=tf.float64)
print('x = {}'.format(x))
likelihood_var = tf.cast(esbps.likelihood_var, tf.float64)
print(esbps.k)
print(x @ beta_cov)
additional_term = esbps.k * tf.reshape(
    x @ beta_cov @ tf.transpose(x) + likelihood_var, beta_zero.shape)

esbps_hull = beta_one * t + beta_zero + additional_term
plt.figure()
plt.plot(S_hi[0:sample[-1]],
         fS_hi[0:sample[-1]],
         linewidth=7,
         alpha=0.6,
         label='target')
plt.plot(t[0:sample[-1]],
         esbps_hull[0:sample[-1]],
         '--',
         linewidth=3.5,
         alpha=0.6,
         label='linear')
plt.plot(S_hi[0:sample[-1]],
         env[0:sample[-1]],
         'r--',
         linewidth=3.5,
         alpha=1.0,
         label='adaptive')
plt.scatter(t[0:sample[-1]], y[0:sample[-1]], s=60, label='samples')
plt.legend()
plt.savefig('test_hull_both.png')
plt.savefig('test_hull_both.pdf')
print(esbps_hull)
print(x_time)

plt.figure()
plt.plot(S_hi[0:sample[-1]], int_env[0:sample[-1]])
plt.title('Integrated adaptive rate')
plt.savefig('test_hull_integrated.png')
plt.savefig('test_hull_integrated.pdf')

plt.figure()
plt.plot(int_env[0:sample[-1]], inv_int_env[0:sample[-1]])
plt.title('Inverse integrated rate')
plt.savefig('test_hull_inv_integrated.png')
plt.savefig('test_hull_inv_integrated.pdf')

plt.figure()
plt.plot(S_hi[0:sample[-1]],
         fS_hi[0:sample[-1]],
         linewidth=5,
         alpha=0.6,
         label='target')
plt.plot(S_hi[0:sample[-1]],
         env[0:sample[-1]],
         'r--',
         linewidth=2,
         alpha=1.0,
         label='atSBPS')
plt.scatter(t[0:sample[-1]], y[0:sample[-1]], s=60, label='samples')
plt.title('Adaptive approx. upper bound from atSBPS')
plt.legend(loc='lower right')
plt.savefig('test_hull__.png')
plt.savefig('test_hull__.pdf')
