from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from math import pi

import sys
from ops import svgd_gradient, rbf_kernel
from gmm_models import mixture_weights_and_grads

class Layer():

    def __init__(self, n_p, n_in, n_out, activation_fn=tf.nn.relu, name='l1'):
        # n_p: number of particles
        # n_in: input dimension
        # n_out: output dimension

        #self.__dict__.update(locals())
        self.n_p, self.n_in, self.n_out = n_p, n_in, n_out
        self.activation_fn = activation_fn
        with tf.variable_scope(name) as scope:
            w0 = (1.0 / np.sqrt(self.n_in + 1) * np.random.randn(self.n_p, self.n_in, self.n_out)).astype('float32')
            self.w = tf.get_variable('w',
                    shape=(self.n_p, self.n_in, self.n_out), dtype=tf.float32,
                    initializer = tf.glorot_uniform_initializer() )
                    #initializer=w0, dtype=tf.float32)

            self.params = [self.w]


    def forward(self, inputs):
        assert tf.keras.backend.ndim(inputs) == 3
        # inputs: n_p x B x n_in
        # w: n_p x n_in x n_out
        a = tf.matmul(inputs, self.w)
        if self.activation_fn is not None:
            h = self.activation_fn(a)
        else:
            h = a
        return a, h
       

class SVGD():

    def __init__(self, config):
        self.config = config

        # create placeholders for the input
        self.X = tf.placeholder(
            name='X', dtype=tf.float32,
            shape=[None, self.config.dim],
        )

        self.y = tf.placeholder(
            name='y', dtype=tf.float32,
            shape=[None],
        )

        #self.log_v_noise = tf.get_variable('log_v_noise', 
        #        initializer=tf.constant(np.log(1.0,).astype('float32')),
        #        dtype=tf.float32)

        #self.v_noise_vars = [self.log_v_noise]

        self.step = tf.placeholder_with_default(1., shape=(), name='step')
        #self.neg_log_var = tf.placeholder_with_default(0., shape=(), name='neg_log_var')
        self.n_neurons = [self.config.dim, self.config.n_hidden, self.config.n_hidden, 1]
        #self.n_neurons = [self.config.dim, self.config.n_hidden, 1]

        # build network
        self.nnet = []
        for i in range(len(self.n_neurons) - 1):
            activation_fn = tf.nn.relu
            if i == len(self.n_neurons) - 2:
                activation_fn = None
            self.nnet.append( Layer(self.config.n_particles, self.n_neurons[i], self.n_neurons[i+1], activation_fn=activation_fn, name='l_%d' % i) )

        # forward, A, H
        n_layers = len(self.nnet)
        cache = []
        self.train_vars = []
        h = tf.tile(tf.expand_dims(self.X, 0), (self.config.n_particles, 1, 1))
        cache.append(h)
        for i in range(n_layers):
            a, h = self.nnet[i].forward(h)
            cache.append(a)
            if i != n_layers-1: cache.append(h) # last layer
            self.train_vars += self.nnet[i].params
        self.y_pred = tf.squeeze(h) # n_p x B

        #self.log_prob = tf.reduce_sum(self.get_log_liklihood(self.y, self.y_pred))
        self.log_lik, self.log_prior = self.get_log_liklihood(self.y, self.y_pred)
        self.log_prob =  self.log_lik + self.log_prior  # []
        self.net_grads = tf.gradients(self.log_prob, self.train_vars)

        #############################################################

        # kfac parameters
        self.A = [tf.Variable(tf.zeros((self.config.n_particles, p.get_shape()[1], p.get_shape()[1])), trainable=False, dtype=tf.float32) for p in self.train_vars]
        self.G = [tf.Variable(tf.zeros((self.config.n_particles, p.get_shape()[2], p.get_shape()[2])), trainable=False, dtype=tf.float32) for p in self.train_vars]

        # KFAC
        self.G_inv = [tf.Variable(tf.zeros(p.get_shape()), trainable=False, dtype=tf.float32) for p in self.G]
        self.A_inv = [tf.Variable(tf.zeros(p.get_shape()), trainable=False, dtype=tf.float32) for p in self.A]

        #batch_size = tf.cast(tf.shape(self.X)[0], tf.float32)
        batch_size = tf.cast(tf.shape(self.X)[0], tf.float32) 
        self.A_, self.G_ = [], []
        for p in cache[::2]:
            self.A_.append( tf.matmul(p, p, transpose_a=True) / batch_size )
        #h_grads = tf.gradients(self.log_prob, cache[1::2])
        h_grads = tf.gradients(self.log_lik / self.config.n_train, cache[1::2])
        print(cache[1::2])
        for g in h_grads:
            self.G_.append( tf.matmul(g, g, transpose_a=True) / batch_size )
        assert len(self.A_) == len(self.G_) 
        #############################################################


        #############################################################
        # vanilla svgd 
        self.svgd_grads = []
        for p, g in zip(self.train_vars, self.net_grads):
            svgd_grad = svgd_gradient(p, g, kernel=self.config.kernel)
            self.svgd_grads.append( -svgd_grad ) # maximize

        #############################################################
        # SVGD KFAC 
        self.inc_ops = self.inc_add_step()
        self.inv_ops = self.mat_inv_step(self.config.eps)
        #self.scaling_ops = self.ra_scaling_step(self.net_grads) # compute scalings
        #self.eig_ops = self.eig_bas_step() # eigen decomposition
        if self.config.method == 'svgd_kfac':
            self.svgd_kfac_grads = self.kfac_gradients(self.svgd_grads)

        if self.config.method == 'map_kfac':
            self.map_kfac_grads = self.kfac_gradients(self.net_grads)

        if self.config.method == 'mixture_kfac':
            ## mixture KFAC
            self.mixture_kfac_grads = self.mixture_kfac_gradient(self.net_grads)


        #############################################################
        if self.config.method in ['SGLD', 'pSGLD']:
            # for stability
            self.acc = [tf.Variable(tf.ones(p.get_shape()), trainable=False, dtype=tf.float32) for p in self.train_vars]

            mean_log_lik = self.log_lik / self.config.n_train
            mean_log_lik_grads = tf.gradients(mean_log_lik, self.train_vars)
            self.moment_op = self.moment_inc_step(mean_log_lik_grads)

            M = [tf.sqrt(acc_new + 1e-6) for acc_new in self.acc]
            
            assert len(self.train_vars) == len(M)
            self.psgld_grads = [ -self.config.learning_rate * (g - p/self.config.n_train) / m + \
                        tf.sqrt(self.config.learning_rate / m) * 2. /self.config.n_train * tf.random_normal(p.get_shape())
                         for p, g, m in zip(self.train_vars, mean_log_lik_grads, M)]

        tf.summary.scalar("log_prob", tf.reduce_sum(self.log_prob))


    def mixture_kfac_gradient(self, in_grads):

        # for the \ell-th cluster
        def _weighted_svgd(x, d_log_pw, w):
            kxy, dxkxy = rbf_kernel(x, to3d=True)
            velocity = tf.reduce_sum(tf.expand_dims(tf.expand_dims(w, 0), 2) * tf.expand_dims(kxy, 2) * tf.expand_dims(d_log_pw, 0), axis=1) + \
                        tf.reduce_sum(tf.expand_dims(tf.expand_dims(w, 1), 2) * dxkxy, axis=0)
            # n * d , d x d
            return velocity


        def _mixture_svgd_grads(x, x_shape, d_log_p, mix, mix_grads, g_inv, a_inv):
            velocity = 0
            for i in range(self.config.n_particles):
                w_i_svgd = _weighted_svgd(x, d_log_p + mix_grads[i], mix[i])
                w_i_svgd = tf.reshape(w_i_svgd, x_shape)

                # H_\ell
                g = tf.tile(tf.expand_dims(g_inv[i], 0), [self.config.n_particles, 1, 1])
                a = tf.tile(tf.expand_dims(a_inv[i], 0), [self.config.n_particles, 1, 1])
                delta = tf.matmul(tf.matmul(g, w_i_svgd, transpose_b=True), a)
                velocity += tf.expand_dims(tf.expand_dims(mix[i], 1), 2) * delta

            return tf.transpose(velocity, [0,2,1])


        out_grads = []
        for k in range(len(self.train_vars)):
            x_k = self.train_vars[k]
            x_k_shape = tf.shape(x_k)
            x_k_flat = tf.reshape(x_k, [self.config.n_particles, -1])
            x_k_grad = tf.reshape(in_grads[k], [self.config.n_particles, -1])
            mix_k, mix_grads_k = mixture_weights_and_grads(x_k_flat)  # c * n, c * n * d

            g_inv_k = self.G_inv[k]
            a_inv_k = self.A_inv[k]
            vel_k = _mixture_svgd_grads(x_k_flat, x_k_shape, x_k_grad, mix_k, mix_grads_k, g_inv_k, a_inv_k)
            out_grads.append( -vel_k )

        return out_grads



    def moment_inc_step(self, in_grads, rho=0.95):
        assert len(self.acc) == len(in_grads)
        assign_ops = []
        for k in range(len(self.acc)):
            assign_ops.append(tf.assign(self.acc[k], rho*self.acc[k]+ (1.-rho)*in_grads[k]**2))
        return tf.group(*assign_ops)


    def inc_add_step(self,):
        rho = tf.minimum(1. - 1./self.step, 0.95)
        assign_ops = []
        for k in range(len(self.A)):
            assign_ops.append(tf.assign(self.A[k], rho * self.A[k] + (1.-rho) * self.A_[k]))
            assign_ops.append(tf.assign(self.G[k], rho * self.G[k] + (1.-rho) * self.G_[k]))
        return tf.group(*assign_ops)


    def mat_inv_step(self, eps=5e-3):
        assign_ops = []
        for k in range(len(self.A_inv)):
            #print(self.A_inv[k].get_shape(), self.G_inv[k].get_shape())
            assign_ops.append( tf.assign(self.A_inv[k], tf.linalg.inv(self.A[k] + eps * tf.expand_dims(tf.eye(tf.shape(self.A[k])[1]), 0))) )
            assign_ops.append( tf.assign(self.G_inv[k], tf.linalg.inv(self.G[k] + eps * tf.expand_dims(tf.eye(tf.shape(self.G[k])[1]), 0))) )
        return tf.group(*assign_ops)


    def kfac_gradients(self, in_grads):
        out_grads = []
        for k in range(len(self.train_vars)):
            if self.config.method == 'svgd_kfac':
                g = tf.tile(tf.reduce_mean(self.G_inv[k], 0, keep_dims=True), [self.config.n_particles, 1, 1])
                a = tf.tile(tf.reduce_mean(self.A_inv[k], 0, keep_dims=True), [self.config.n_particles, 1, 1])
                delta = tf.matmul(tf.matmul(g, in_grads[k], transpose_b=True), a)
            elif self.config.method == 'map_kfac':
                delta = tf.matmul(tf.matmul(self.G_inv[k], -in_grads[k], transpose_b=True), self.A_inv[k])
            else:
                raise NotImplementedError
            #print(g.get_shape(), a.get_shape())
            #delta = tf.matmul(tf.matmul(self.G_inv[k], in_grads[k], transpose_b=True), self.A_inv[k])
            out_grads.append(tf.transpose(delta, [0,2,1]))
        return out_grads 

    #def eig_bas_step(self):
    #    assign_ops = []
    #    for k in range(len(self.U_A)):
    #        _, U_Ak = tf.linalg.eigh(self.U_A[k])
    #        _, U_Gk = tf.linalg.eigh(self.U_G[k])
    #        assign_ops.append(tf.assign(self.U_A[k], U_Ak))
    #        assign_ops.append(tf.assign(self.U_G[k], U_Gk))
    #    return assign_ops

    #def ra_scaling_step(self, net_grads):
    #    rho = tf.minimum(1. - 1./self.step, 0.95)
    #    assign_ops = []
    #    for k in range(len(self.RA_S)):
    #        s = tf.square( tf.matmul(tf.matmul(self.U_G[k], net_grads[k], transpose_b=True), self.U_A[k], transpose_b=True) )
    #        assign_ops.append( tf.assign(self.RA_S[k], rho*self.RA_S[k] + (1.-rho) * s ) )
    #    return assign_ops

    #def eakfac_gradients(self, net_grads, eps=1e-2):
    #    train_grads = []
    #    for k in range(len(self.net_grads)):
    #        g = tf.tile(tf.reduce_mean(self.U_G[k], 0, keep_dims=True), [self.config.n_particles, 1, 1])
    #        a = tf.tile(tf.reduce_mean(self.U_A[k], 0, keep_dims=True), [self.config.n_particles, 1, 1])
    #        delta = tf.matmul(tf.matmul(g, net_grads[k], transpose_b=True), a, transpose_b=True)
    #        delta /= (self.RA_S[k] + eps)
    #        # project back to the original basis
    #        delta = tf.matmul(tf.matmul(g, delta, transpose_a=True), a)
    #        train_grads.append(tf.transpose(delta, [0,2,1]))
    #    return train_grads


    def get_log_liklihood(self, y_true, y_pred):
        #v_noise = tf.exp(self.log_v_noise)
        log_v_noise, v_noise = np.log(0.5), 0.5
        # location = 0, scale = 1
        log_lik_data = -self.config.n_train * 0.5 * tf.log(2.*np.pi) * log_v_noise \
                       -self.config.n_train * 0.5 * tf.reduce_mean((y_pred - tf.expand_dims(y_true, 0))**2 / v_noise, axis=1)

        log_prior_w = 0
        for p in self.train_vars: 
            log_prior_w += ( -0.5*tf.reduce_sum(tf.reshape(p**2, (self.config.n_particles, -1)), axis=1) )

        # sub-sampling mini-batches of data, where (X, y) is the batch data, and N is the number of whole observations
        #log_posterior = log_lik_data + log_prior_w
        #return log_posterior
        return log_lik_data, log_prior_w


#    def get_error_and_ll(self, y_pred, y_true, neg_log_var):
#        y_pred = y_pred * self.scale + self.location
#        prob = tf.sqrt(tf.exp(neg_log_var) / (2*np.pi)) * tf.exp( -0.5*(y_pred - tf.expand_dims(y_true, 0))**2 * tf.exp(neg_log_var) )
#
#        rmse = tf.sqrt(tf.reduce_mean((y_true - tf.reduce_mean(y_pred, 0))**2))
#        ll = tf.reduce_mean( tf.log(tf.mean(prob, axis=0)) )
#        return rmse, ll


    def get_feed_dict(self, batch_chunk, step=None):
        fd = {
            self.X: batch_chunk['X'],  
            self.y: batch_chunk['y'],  
        }
        if step is not None:
            fd[self.step] = step
        return fd


    #def mixture_kfac_gradient(self,):

    #    out_grads = []
    #    for x, grad, minIdx, G_inv, A_inv in zip(self.train_vars, self.net_grads, self.minIdx, self.G_inv, self.A_inv):
    #        partitioned_particles = tf.dynamic_partition(data=x, partitions=minIdx, num_partitions=self.config.n_clusters)
    #        partitioned_gradients = tf.dynamic_partition(data=grad, partitions=minIdx, num_partitions=self.config.n_clusters)

    #        partitioned_G_inv = tf.dynamic_partition(data=G_inv, partitions=minIdx, num_partitions=self.config.n_clusters)
    #        partitioned_A_inv = tf.dynamic_partition(data=A_inv, partitions=minIdx, num_partitions=self.config.n_clusters)

    #        partitioned_indices = tf.dynamic_partition(
    #                  data=tf.cast(tf.range(tf.shape(x)[0]), tf.int32),
    #                  partitions=minIdx,
    #                  num_partitions=self.config.n_clusters)

    #        mixture_grads = [] #[None for _ in range(self.config.n_clusters)]
    #        for sp, sg, pg, pa in zip(partitioned_particles, partitioned_gradients, partitioned_G_inv, partitioned_A_inv):
    #            g = tf.tile(tf.reduce_mean(pg, 0, keep_dims=True), [tf.shape(sp)[0], 1, 1])
    #            a = tf.tile(tf.reduce_mean(pa, 0, keep_dims=True), [tf.shape(sp)[0], 1, 1])
    #            ss = tf.cond(
    #                tf.less_equal(tf.shape(sp)[0], 1),
    #                lambda: tf.matmul(tf.matmul(pg, -sg, transpose_b=True), pa),
    #                lambda: tf.matmul(tf.matmul(g, -svgd_gradient(sp, sg, kernel=self.config.kernel), transpose_b=True), a)
    #            )
    #            mixture_grads.append(tf.transpose(ss, [0,2,1]))
    #        ret = tf.dynamic_stitch(partitioned_indices, mixture_grads)
    #        ret.set_shape(x.get_shape())
    #        out_grads.append(ret)

    #    return out_grads


