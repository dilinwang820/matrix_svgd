from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from math import pi

import sys
from ops import svgd_gradient, sqr_dist, rbf_kernel
from gmm_models import mixture_weights_and_grads

class SVGD():

    def __init__(self, config):
        self.config = config

        W = tf.get_variable('W',
                    shape=(self.config.n_particles, self.config.dim,), dtype=tf.float32,
                    #initializer = tf.random_uniform_initializer(-0.1, 0.1) )
                    initializer = tf.glorot_uniform_initializer() )
        self.train_vars = [W]

        # create placeholders for the input
        self.X = tf.placeholder(
            name='X', dtype=tf.float32,
            shape=[None, self.config.dim],
        )

        # binary classification, 0 / 1
        self.y = tf.placeholder(
            name='y', dtype=tf.float32,
            shape=[None],
        )

        batch_size = tf.cast(tf.shape(self.X)[0], tf.float32)
        self.step = tf.placeholder_with_default(1., shape=(), name='step')

        z = tf.reduce_sum(tf.expand_dims(self.X, 0) * tf.expand_dims(W, 1), -1) # n_p * B
        y_prob = tf.sigmoid(z) #n_p * B

        y_expand = tf.expand_dims(self.y, 0) # 1 * B
        dy = (y_expand - y_prob) / (y_prob - y_prob**2) # n_p * B
        dz = tf.sigmoid(z) * (1. - tf.sigmoid(z)) # n_p * B
        dW = tf.reduce_mean(tf.expand_dims(self.X, 0) * tf.expand_dims(dz * dy, 2), 1)

        grad_loglik_z = (y_prob - y_expand) / (y_prob - y_prob**2) * dz 
        grad_loglik_W = tf.expand_dims(grad_loglik_z, 2) * tf.expand_dims(self.X, 0) # n_p * B * d

        mean_dW = tf.reduce_mean(grad_loglik_W, axis=1, keep_dims=True)
        diff_dW = grad_loglik_W - mean_dW
        cov_dW_ = tf.matmul(diff_dW, diff_dW, transpose_a=True) / batch_size

        W_grads = dW * self.config.n_train - W

        y_pred = tf.reduce_mean(y_prob, 0)
        self.ll = tf.reduce_mean( self.y * tf.log(y_pred + 1e-3) + (1. - self.y) * tf.log(1. - y_pred + 1e-3) )
        self.accuracy = tf.reduce_mean( tf.cast(tf.equal(
                                                    tf.cast(tf.greater(self.y, 0.5), tf.int32),
                                                    tf.cast(tf.greater(y_pred, 0.5), tf.int32)
                                                ), tf.float32) )


        # update covariance
        cov_dW = tf.Variable(tf.ones_like(cov_dW_), trainable=False, dtype=tf.float32)
        rho = tf.minimum(1. - 1./self.step, 0.95)
        self.cov_update_step = tf.assign(cov_dW, rho * cov_dW + (1. - rho) * cov_dW_)
        H_inv = tf.linalg.inv(cov_dW + 1e-2*tf.expand_dims(tf.eye(self.config.dim), 0))

        ##############################################################
        ### svgd ## 
        svgd_grad = svgd_gradient(W, W_grads, kernel=self.config.kernel)
        self.svgd_grads = [ -svgd_grad ]

        ### average hessian ###
        if config.method == 'svgd_kfac':
            kfac_grad = tf.matmul(svgd_grad, tf.reduce_mean(H_inv, 0))
            self.kfac_grads = [-kfac_grad]
        
        ### mixture hessians ###
        if config.method == 'mixture_kfac':
            self.mixture_grads = [self.mixture_kfac_gradient(W, W_grads, H_inv)]

        ##############################################################
        if self.config.method in ['SGLD', 'pSGLD']:
            # better stability
            self.acc = tf.Variable(tf.ones(W.get_shape()), trainable=False, dtype=tf.float32)

            self.moment_op = self.moment_inc_step(dW)
            G = tf.sqrt(self.acc + 1e-6) 

            self.psgld_grads = [ -self.config.learning_rate * (dW - W/self.config.n_train) / G + \
                        tf.sqrt(self.config.learning_rate / G) * 2. /self.config.n_train * tf.random_normal(W.get_shape()) ]

        self.log_prob = self.ll
        tf.summary.scalar("log_prob", self.log_prob)


    def moment_inc_step(self, in_grads, rho=0.95):
        assign_ops = []
        assign_ops.append(tf.assign(self.acc, rho*self.acc+ (1.-rho)*in_grads**2))
        return tf.group(*assign_ops)


    def get_feed_dict(self, batch_chunk, step=None):
        fd = {
            self.X: batch_chunk['X'],  
            self.y: batch_chunk['y'],  
        }
        if step is not None:
            fd[self.step] = step
        return fd


    def mixture_kfac_gradient(self, W, W_grads, H_inv):
        # for the \ell-th cluster
        def _weighted_svgd(x, d_log_pw, w):
            kxy, dxkxy = rbf_kernel(x, to3d=True)
            velocity = tf.reduce_sum(tf.expand_dims(tf.expand_dims(w, 0), 2) * tf.expand_dims(kxy, 2) * tf.expand_dims(d_log_pw, 0), axis=1) + \
                        tf.reduce_sum(tf.expand_dims(tf.expand_dims(w, 1), 2) * dxkxy, axis=0)
            # n * d , d x d
            return velocity


        def _mixture_svgd_grads(x, d_log_p, mix, mix_grads, H_inv):
            velocity = 0
            for i in range(self.config.n_particles):
                w_i_svgd = _weighted_svgd(x, d_log_p + mix_grads[i], mix[i])

                # H_\ell
                delta = tf.matmul(w_i_svgd, H_inv[i])
                velocity += tf.expand_dims(mix[i], 1) * delta
            return  velocity

        mix, mix_grads = mixture_weights_and_grads(W)  # c * n, c * n * d
        velocity = _mixture_svgd_grads(W, W_grads, mix, mix_grads, H_inv)
        return -velocity



