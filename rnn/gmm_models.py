import theano
import theano.tensor as T
import numpy as np


### theano implementation ###
def _sum_log_exp(X, mus, dcovs, weights):

    dim = T.cast(mus.shape[1], theano.config.floatX)
    _lnD = T.sum(T.log(dcovs), axis=1)


    diff = X.dimshuffle('x', 0, 1) - mus.dimshuffle(0, 'x', 1) # c x n x d
    diff_times_inv_cov = diff * 1. / dcovs.dimshuffle(0, 'x', 1) # c x n x d
    sum_sq_dist_times_inv_cov = T.sum(diff_times_inv_cov * diff, axis=2)  # c x n 
    ln2piD = T.log(2 * np.pi) * dim
    #log_coefficients = tf.expand_dims(ln2piD + tf.log(self._D), 1) # c x 1
    log_coefficients = (ln2piD + _lnD).dimshuffle(0, 'x') # c x 1
    log_components = -0.5 * (log_coefficients + sum_sq_dist_times_inv_cov)  # c x n
    log_weighted = log_components + T.log(weights).dimshuffle(0, 'x')  # c x n + c x 1
    log_shift = T.max(log_weighted, 0, keepdims=True)

    return log_weighted, log_shift


def _log_gradient(X, mus, dcovs, weights):  

    # X: n_samples x d; mu: c x d; cov: c x d x d
    x_shape = X.shape
    assert X.ndim == 2, 'illegal inputs'

    def posterior(X):
        log_weighted, log_shift = _sum_log_exp(X, mus, dcovs, weights)
        prob = T.exp(log_weighted - log_shift) # c x n
        prob = prob / T.sum(prob, axis=0, keepdims=True)
        return prob

    diff = X.dimshuffle('x', 0, 1) - mus.dimshuffle(0, 'x', 1) # c x n x d
    diff_times_inv_cov = -diff * 1. / dcovs.dimshuffle(0, 'x', 1)  # c x n x d

    P = posterior(X)  # c x n
    score = T.batched_dot(
        P.dimshuffle(1, 'x', 0), 
        diff_times_inv_cov.dimshuffle(1, 0, 2)
    ) 
    return T.squeeze(score) # n x d


def mixture_weights_and_grads(X, mus=None, dcovs=None, weights=None):  
    # X: n_samples x d; 
    x_shape = X.shape
    assert X.ndim == 2, 'illegal inputs'
    
    if mus is None:
        mus = theano.gradient.disconnected_grad(X)
    if dcovs is None:
        dcovs = T.ones_like(mus).astype(theano.config.floatX)
    # uniform weights, only care about ratio
    if weights is None: 
        weights = T.ones([mus.shape[0],], theano.config.floatX)

    log_weighted, log_shift = _sum_log_exp(X, mus, dcovs, weights)
    exp_log_shifted = T.exp(log_weighted - log_shift) # c x n
    exp_log_shifted_sum = T.sum(exp_log_shifted, axis=0, keepdims=True) # 1 x n
    p = exp_log_shifted / exp_log_shifted_sum

    # weights
    mix = p.dimshuffle(1, 0)  # n * c
    d_log_gmm = _log_gradient(X, mus, dcovs, weights) # n * d

    d_log_gau = -(X.dimshuffle(0, 'x', 1) - mus.dimshuffle('x', 0, 1)) / dcovs.dimshuffle('x', 0, 1) # n x c x d
    mix_grad = d_log_gau - d_log_gmm.dimshuffle(0, 'x', 1)

    # c * n, c * n * d
    return mix.dimshuffle(1, 0), mix_grad.dimshuffle(1, 0, 2)



#from models import GaussianMixture
#
#def _simulate_mixture_target(n_components=10, dim = 1, val=5., seed=123):
#
#    with tf.variable_scope('p_target') as scope:
#        np.random.seed(seed)
#        mu0 = tf.get_variable('mu', initializer=np.random.uniform(-val, val, size=(n_components, dim)).astype('float32'), dtype=tf.float32,  trainable=False)
#
#        log_sigma0 = tf.zeros((n_components, dim))
#        weights0 = tf.ones(n_components) / n_components
#        p_target = GaussianMixture(n_components, mu0, log_sigma0, weights0)
#
#        return p_target
#
#
#
#if __name__ == '__main__':
#
#    #X = T.matrix()
#    x0 = np.random.normal(size=(2, 3)).astype('float32')
#    X = theano.shared(x0)
#    mix, mix_grads, g01 = mixture_weights_and_grads(X)
#
#    f_update = theano.function([], [mix, mix_grads, g01])
#
#    m0, mg0, g01 = f_update()
#    print(g01)
#    print(mg0[1, 0])
#    
#    #dxk1, dxk2 = sess.run([dk1, dk2])
#    #print (dxk1)
#    #print (np.sum(dxk2, 0))
#    #k1, dk1 = rbf_kernel(x_train)
#    
