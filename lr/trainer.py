from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from pprint import pprint
import sys

from util import log
import os
import glob
import time

from model_svgd import  SVGD
from load_data import load_uci_dataset


#import pdb

class Trainer(object):

    def optimize_sgd(self, train_vars, loss=None, train_grads=None, lr=1e-2):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)  #adagrad with momentum
        if train_grads is not None:
            train_op = optimizer.apply_gradients(zip(train_grads, train_vars))
        else:
            train_op = optimizer.minimize(tf.reduce_mean(loss), var_list=train_vars, global_step=self.global_step)
        return train_op

    def optimize_adagrad(self, train_vars, loss=None, train_grads=None, lr=1e-2):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.9)  #adagrad with momentum
        if train_grads is not None:
            train_op = optimizer.apply_gradients(zip(train_grads, train_vars))
        else:
            train_op = optimizer.minimize(tf.reduce_mean(loss), var_list=train_vars, global_step=self.global_step)
        return train_op


    def optimize_adam(self, train_vars, loss=None, train_grads=None, lr=1e-2):
        assert (loss is not None) or (train_grads is not None), 'illegal inputs'
        optimizer = tf.train.AdamOptimizer(lr)
        if train_grads is not None:
            train_op = optimizer.apply_gradients(zip(train_grads, train_vars))
        else:
            train_op = optimizer.minimize(loss, var_list=train_vars, global_step=self.global_step)
        return train_op


    def __init__(self, config, dataset, session):
        self.config = config
        self.session = session
        self.dataset = dataset

        self.filepath = '%s' % (
            config.method,
        )

        self.train_dir = './train_dir/%s' % self.filepath
        #self.fig_dir = './figures/%s' % self.filepath

        #for folder in [self.train_dir, self.fig_dir]:
        for folder in [self.train_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)
            # clean train folder
            if self.config.clean:
                files = glob.glob(folder + '/*')
                for f in files: os.remove(f)

        #log.infov("Train Dir: %s, Figure Dir: %s", self.train_dir, self.fig_dir)

        # --- create model ---
        self.model = SVGD(config)

        # --- optimizer ---
        #self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.global_step = tf.Variable(0, name="global_step")

        self.learning_rate = config.learning_rate
        #self.learning_rate = tf.train.exponential_decay(
        #        self.learning_rate,
        #        global_step=self.global_step,
        #        decay_steps=500,
        #        decay_rate=0.5,
        #        staircase=True,
        #        name='decaying_learning_rate'
        #)

        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)
        self.checkpoint_secs = 300  # 5 min

        ##self.train_op = self.optimize_adam( self.model.kl_loss, lr=self.learning_rate)
        if self.config.method == 'svgd':
            self.train_op = self.optimize_adagrad( self.model.train_vars, train_grads=self.model.svgd_grads, lr=self.learning_rate)
        elif self.config.method == 'svgd_kfac':
            self.train_op = self.optimize_adagrad( self.model.train_vars, train_grads=self.model.kfac_grads, lr=self.learning_rate)
        elif self.config.method == 'mixture_kfac':
            self.train_op = self.optimize_adagrad( self.model.train_vars, train_grads=self.model.mixture_grads, lr=self.learning_rate)
        elif self.config.method in ['SGLD', 'pSGLD']:
            self.train_op = self.optimize_sgd( self.model.train_vars, train_grads=self.model.psgld_grads, lr=1.0)

        tf.global_variables_initializer().run()
        if config.checkpoint is not None:
            self.ckpt_path = tf.train.latest_checkpoint(self.config.checkpoint)
            if self.ckpt_path is not None:
                log.info("Checkpoint path: %s", self.ckpt_path)
                self.saver.restore(self.session, self.ckpt_path)
                log.info("Loaded the pretrain parameters from the provided checkpoint path")


    def evaluate(self, step):
        
        def get_lik_and_acc(X, y):
            n = len(X)
            ll, acc = [], []
            batch_size = 2000
            for i in range( n // batch_size +1 ):
                start = i * batch_size
                end = min((i+1)*batch_size, n)
                batch = {
                    'X': X[start:end],
                    'y': y[start:end],
                }
                ll_i, acc_i = self.session.run([self.model.ll, self.model.accuracy], feed_dict=self.model.get_feed_dict(batch, step))

                ll.append(ll_i)
                acc.append(acc_i)
            return np.mean(ll), np.mean(acc)

        train_ll, train_acc = get_lik_and_acc(self.dataset.x_train, self.dataset.y_train)
        valid_ll, valid_acc = get_lik_and_acc(self.dataset.x_valid, self.dataset.y_valid)
        test_ll, test_acc = get_lik_and_acc(self.dataset.x_test, self.dataset.y_test)

        return train_ll, train_acc, valid_ll, valid_acc, test_ll, test_acc



    def train(self):
        log.infov("Training Starts!")
        output_save_step = 1000
        buffer_save_step = 100
        self.session.run(self.global_step.assign(0)) # reset global step
        n_updates = 1

        for ep in xrange(1, 1+self.config.n_epoches):
            x_train, y_train = shuffle(self.dataset.x_train, self.dataset.y_train)
            max_batches = self.config.n_train // self.config.batch_size 

            #if self.config.n_train % self.config.batch_size != 0: max_batches += 1
            for bi in xrange(max_batches):
                start = bi * self.config.batch_size
                end = min((bi+1) * self.config.batch_size, self.config.n_train)

                batch_chunk = {
                    'X': x_train[start:end],
                    'y': y_train[start:end]
                }

                step, summary, log_prob, step_time = \
                        self.run_single_step(n_updates, batch_chunk)

                #if np.any(np.isnan(log_prob)): sys.exit(1)

                self.summary_writer.add_summary(summary, global_step=step)
                #if n_updates % 100 == 0:
                #    self.log_step_message(n_updates, log_prob, step_time)

                if n_updates % 50 == 0:
                    print (n_updates, self.evaluate(n_updates))

                n_updates+= 1


            #if ep % (self.config.n_epoches//10 + 1) == 0:
            #    rmse, ll = self.evaluate()
            #    print(ep, rmse, ll)

        #test_rmse, test_ll = self.evaluate()
        #write_time = time.strftime("%m-%d-%H:%M:%S")
        #with open(self.config.savepath + self.config.dataset + "_test_ll_rmse_%s.txt" % (self.filepath), 'a') as f:
        #    f.write(repr(self.config.trial) + ',' + write_time + ',' + repr(self.config.n_epoches) + ',' + repr(test_rmse) + ',' + repr(test_ll) + '\n')

        #if self.config.save:
        #    # save model at the end
        #    self.saver.save(self.session,
        #        os.path.join(self.train_dir, 'model'),
        #        global_step=step)


    def run_single_step(self, step, batch_chunk):
        _start_time = time.time()
        fetch = [self.global_step, self.summary_op, self.model.log_prob]
        if self.config.method in ['mixture_kfac', 'svgd_kfac']:
            fetch += [self.model.cov_update_step]

        if self.config.method == 'pSGLD':
            fetch += [self.model.moment_op]

        fetch += [self.train_op]

        fetch_values = self.session.run(
            fetch, feed_dict = self.model.get_feed_dict(batch_chunk, step)
        )

        [step, summary, log_prob] = fetch_values[:3]
        _end_time = time.time()
        return step, summary, log_prob, (_end_time - _start_time)


    def log_step_message(self, step, log_prob, step_time, is_train=True):
        if step_time == 0:
            step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                #"loss: {loss:.4f} " +
                "log_prob: {log_prob:.4f} " +
                "({sec_per_batch:.3f} sec/batch)"
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step, log_prob=log_prob,
                         sec_per_batch=step_time,
                         )
               )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoches', type=int, default=2, required=False)
    parser.add_argument('--method', type=str, default='svgd', choices=['SGLD', 'pSGLD', 'svgd', 'svgd_kfac', 'mixture_kfac'], required=True)
    parser.add_argument('--n_particles', type=int, default=20, required=False)
    parser.add_argument('--batch_size', type=int, default=256, required=False)
    parser.add_argument('--dataset', type=str, default='covtype', required=False, choices=['covtype'])
    parser.add_argument('--trial', type=int, default=1, required=False)
    parser.add_argument('--learning_rate', type=float, default=5e-3, required=False)
    parser.add_argument('--kernel', type=str, default='rbf', required=False)
    parser.add_argument('--clean', action='store_true', default=False)
    parser.add_argument('--savepath', type=str, default='results/', required=False)
    parser.add_argument('--checkpoint', type=str, default=None, required=False)
    parser.add_argument('--save', action='store_true', default=False)
    config = parser.parse_args()
    
    if not config.save:
        log.warning("nothing will be saved.")

    session_config = tf.ConfigProto(

        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True),
        device_count={'GPU': 0},
    )

    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:

        from collections import namedtuple
        dataStruct = namedtuple("dataStruct", "x_train, x_valid, x_test, y_train, y_valid, y_test")

        x_train, x_valid, x_test, y_train, y_valid, y_test = load_uci_dataset(dataset = config.dataset, random_state = config.trial)
        config.n_train, config.dim = x_train.shape
        dataset = dataStruct(x_train=x_train, x_valid=x_valid, x_test=x_test, \
                                 y_train=y_train, y_valid=y_valid, y_test=y_test)
        trainer = Trainer(config, dataset, sess)
        trainer.train()


if __name__ == '__main__':
    main()

