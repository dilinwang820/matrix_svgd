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

'''
    Augmented Lagrangian: for stable training
'''
class Trainer(object):

    def optimize_sgd(self, train_vars, loss=None, train_grads=None, lr=1e-2):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)  #adagrad with momentum
        if train_grads is not None:
            train_op = optimizer.apply_gradients(zip(train_grads, train_vars))
        else:
            train_op = optimizer.minimize(tf.reduce_mean(loss), var_list=train_vars, global_step=self.global_step)
        return train_op

    def optimize_adagrad(self, train_vars, loss=None, train_grads=None, lr=1e-2):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.95)  #adagrad with momentum
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

        self.filepath = '%s-%s' % (
            config.method,
            config.dataset,
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
        if config.lr_weight_decay:
            decay_step = int(0.1 * self.config.n_epoches * self.config.n_train // self.config.batch_size) 
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=decay_step,
                decay_rate=0.8,
                staircase=True,
                name='decaying_learning_rate'
            )

        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)
        self.checkpoint_secs = 300  # 5 min

        #self.train_op = self.optimize_adam( self.model.kl_loss, lr=self.learning_rate)
        if self.config.method == 'svgd':
            self.train_op = self.optimize_adagrad( self.model.train_vars, train_grads=self.model.svgd_grads, lr=self.learning_rate)
        elif self.config.method in ['svgd_kfac', 'map_kfac', 'mixture_kfac']:
            self.inc_op = self.model.inc_ops
            self.inv_op = self.model.inv_ops
            if self.config.method == 'svgd_kfac':
                self.train_op = self.optimize_adagrad( self.model.train_vars, train_grads=self.model.svgd_kfac_grads, lr=self.learning_rate)
            elif self.config.method == 'map_kfac':
                self.train_op = self.optimize_adagrad( self.model.train_vars, train_grads=self.model.map_kfac_grads, lr=self.learning_rate)
            elif self.config.method == 'mixture_kfac':
                self.train_op = self.optimize_adagrad( self.model.train_vars, train_grads=self.model.mixture_kfac_grads, lr=self.learning_rate)
        elif self.config.method in ['SGLD', 'pSGLD']:
                self.train_op = self.optimize_sgd( self.model.train_vars, train_grads=self.model.psgld_grads, lr=1.0)

        tf.global_variables_initializer().run()
        if config.checkpoint is not None:
            self.ckpt_path = tf.train.latest_checkpoint(self.config.checkpoint)
            if self.ckpt_path is not None:
                log.info("Checkpoint path: %s", self.ckpt_path)
                self.saver.restore(self.session, self.ckpt_path)
                log.info("Loaded the pretrain parameters from the provided checkpoint path")


    def evaluate(self,):

        dev_set = {
            'X':self.dataset.x_train[:1000],
            'y':self.dataset.y_train[:1000],
        }
        test_set = {
            'X':self.dataset.x_test,
            'y':self.dataset.y_test,
        }

        pred_y_dev = self.session.run(self.model.y_pred, self.model.get_feed_dict(dev_set))
        pred_y_dev = pred_y_dev * self.dataset.std_y_train + self.dataset.mean_y_train
        y_dev = dev_set['y'] * self.dataset.std_y_train + self.dataset.mean_y_train
        neg_log_var = -np.log(np.mean((pred_y_dev - y_dev) ** 2))

        y_test = test_set['y']
        pred_y_test = self.session.run(self.model.y_pred, self.model.get_feed_dict(test_set))
        pred_y_test = pred_y_test * self.dataset.std_y_train + self.dataset.mean_y_train
        prob = np.sqrt(np.exp(neg_log_var) / (2*np.pi)) * np.exp( -0.5*(pred_y_test - np.expand_dims(y_test, 0))**2 * np.exp(neg_log_var) )

        rmse = np.sqrt(np.mean((y_test - np.mean(pred_y_test, 0))**2))
        ll = np.mean( np.log(np.mean(prob, axis=0)) )
        return rmse, ll, neg_log_var


    def train(self):
        log.infov("Training Starts!")
        output_save_step = 1000
        buffer_save_step = 100
        self.session.run(self.global_step.assign(0)) # reset global step
        n_updates = 1

        x_train, y_train = self.dataset.x_train, self.dataset.y_train
        for ep in xrange(1, 1+self.config.n_epoches):
            x_train, y_train = shuffle(x_train, y_train)
            #x_train, y_train = self.dataset.x_train, self.dataset.y_train
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

                if np.any(np.isnan(log_prob)): sys.exit(1)

                self.summary_writer.add_summary(summary, global_step=step)
                #if n_updates % 500 == 0:
                #    self.log_step_message(n_updates, log_prob, step_time)
                
                #if n_updates % 50 == 0:
                #    rmse, ll, _ = self.evaluate()
                #    print(n_updates, rmse, ll)

                n_updates+= 1

            if ep % (self.config.n_epoches//10 + 1) == 0:
                rmse, ll, neg_log_var = self.evaluate()
                print(ep, neg_log_var, rmse, ll)

        test_rmse, test_ll, _ = self.evaluate()
        write_time = time.strftime("%m-%d-%H:%M:%S")
        
        pardir = "%s_%s/" % (self.config.method, repr(self.config.learning_rate))
        if not os.path.exists(self.config.savepath + pardir):
            os.makedirs(self.config.savepath + pardir)
        
        if self.config.trial  == 1:
            fm = 'w'
        else:
            fm = 'a'
        with open(self.config.savepath + pardir + self.config.dataset + "_test_ll_rmse_%s.txt" % (self.filepath), fm) as f:
            f.write(repr(self.config.trial) + ',' + write_time + ',' + repr(self.config.n_epoches) + ',' + repr(test_rmse) + ',' + repr(test_ll) + '\n')
        
        #if self.config.save:
        #    # save model at the end
        #    self.saver.save(self.session,
        #        os.path.join(self.train_dir, 'model'),
        #        global_step=step)


    def run_single_step(self, step, batch_chunk):
        _start_time = time.time()
        fetch = [self.global_step, self.summary_op, self.model.log_prob]
        if self.config.method in ['svgd_kfac', 'map_kfac', 'mixture_kfac']:
            fetch += [self.inc_op]
            if step % self.config.inverse_update_freq == 0:
                fetch += [self.inv_op]

        if self.config.method == 'pSGLD':
            fetch += [self.model.moment_op]

        fetch += [self.train_op]

        fetch_values = self.session.run(
            fetch, feed_dict = self.model.get_feed_dict(batch_chunk, step)
        )

        [step, summary, log_prob] = fetch_values[:3]
        _end_time = time.time()
        return step, summary, np.sum(log_prob), (_end_time - _start_time)


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
    parser.add_argument('--n_epoches', type=int, default=100, required=False)
    parser.add_argument('--method', type=str, default='svgd', choices=['SGLD', 'pSGLD', 'svgd', 'map_kfac', 'svgd_kfac', 'mixture_kfac'], required=True)
    parser.add_argument('--dataset', type=str, default='boston', required=True)
    parser.add_argument('--n_particles', type=int, default=10, required=False)
    parser.add_argument('--batch_size', type=int, default=100, required=False)
    parser.add_argument('--n_hidden', type=int, default=50, required=False)
    parser.add_argument('--eps', type=float, default=5e-3, required=False)
    parser.add_argument('--inverse_update_freq', type=int, default=20, required=False)
    parser.add_argument('--trial', type=int, default=1, required=False)
    parser.add_argument('--learning_rate', type=float, default=5e-3, required=False)
    parser.add_argument('--kernel', type=str, default='rbf', required=False)
    parser.add_argument('--clean', action='store_true', default=False)
    parser.add_argument('--savepath', type=str, default='results/', required=False)
    parser.add_argument('--checkpoint', type=str, default=None, required=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    config = parser.parse_args()
    
    if not config.save:
        log.warning("nothing will be saved.")
    
    if config.dataset == 'year':
        config.batch_size = 1000
    
    if config.dataset in ['wine', 'boston']:
        config.inverse_update_freq = 5
        config.eps = 5e-2

    session_config = tf.ConfigProto(

        allow_soft_placement=True,
        # intra_op_parallelism_threads=1,
        # inter_op_parallelism_threads=1,
        gpu_options=tf.GPUOptions(allow_growth=True),
        #device_count={'GPU': 1},
    )
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        with tf.device('/gpu:%d'% config.gpu):
            tf.set_random_seed(123)
            np.random.seed(123)

            from collections import namedtuple
            dataStruct = namedtuple("dataStruct", "x_train x_test y_train, y_test, mean_y_train, std_y_train")

            x_train, x_test, y_train, y_test, mean_y_train, std_y_train = load_uci_dataset(config.dataset, config.trial)
            config.n_train, config.dim = x_train.shape
            dataset = dataStruct(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, mean_y_train=mean_y_train, std_y_train=std_y_train)
            trainer = Trainer(config, dataset, sess)
            trainer.train()


if __name__ == '__main__':
    main()

