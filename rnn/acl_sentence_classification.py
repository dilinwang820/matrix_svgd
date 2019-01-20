'''
Scalable Bayesian Learning of Recurrent Neural Networks for Language Modeling 
https://arxiv.org/pdf/1611.08034.pdf
Developed by Zhe Gan, zg27@duke.edu, July, 12, 2016
'''
#import os
import time
import logging
import cPickle

import numpy as np
import theano
import theano.tensor as tensor

from sent_classifier import init_params, init_tparams
from sent_classifier import build_model

from optimizers import pSGLD, SVGD, RMSprop, SGLD
from utils import get_minibatches_idx
from utils import numpy_floatX

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mr', type=str, help='dataset', required=True)
parser.add_argument('--dropout', default=0.0, type=float, help='dropout', required=False)
parser.add_argument('--n_p', default=10, type=int, help='n_particles', required=True)
parser.add_argument('--eps', default=1e-3, type=float, help='matrix inverse', required=False)
parser.add_argument('--method', type=str, default='pSGLD', help='method', choices=['SGLD', 'pSGLD', 'SVGD', 'RMSprop', 'SVGD_KFAC', 'MIXTURE_KFAC'], required=True)
config = parser.parse_args()

save_prefix = '%s_%s_%.2f_%d' % (config.dataset, config.method, config.dropout, config.n_p)

""" Training the model. """

""" used to calculate the prediction error. """

def pred_probs(f_pred_prob, prepare_data, data, iterator, options, verbose=False):
    
    """ compute the probabilities of new examples.
    """
    n_samples = len(data[0])
    n_y = options['n_y']
    probs = np.zeros((n_samples, n_y)).astype(theano.config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs

def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    
    """ compute the prediction error. 
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = np.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err


""" used to preprocess the dataset. """
 

#import nltk
#nltk.download('stopwords')   
#from nltk.corpus import stopwords 
#stop_words = set(stopwords.words('english')) 
def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            #if word in stop_words: continue
            x.append(word_idx_map[word])
    return x

def make_idx_data_cv(revs, word_idx_map, cv):
    """
    Transforms sentences into a 2-d matrix.
    """
    train_x, train_y = [], []
    test_x, test_y = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map)   
        if rev["split"]==cv:            
            test_x.append(sent) 
            test_y.append(rev["y"]) 
        else:  
            train_x.append(sent) 
            train_y.append(rev["y"])    
    
    train = (train_x, train_y)
    test = (test_x, test_y)
    return train, test 
    
def create_valid(train_set,valid_portion=0.10):
    
    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)

    return train, valid
    
def prepare_data(seqs, labels, maxlen=None):
    
    # seqs: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int32')
    x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.
        
    labels = np.array(labels).astype('int32')
    
    return x, x_mask, labels


def train_classifier(train, valid, test, W, n_p=10, n_words=10000, n_x=300, n_h=200, 
    patience=10, max_epochs=50, lrate=0.001, n_train = 10000, optimizer='RMSprop', 
    batch_size=50, valid_batch_size=50, dispFreq=10, validFreq=100,
    saveFreq=500, eps=1e-3):
        
    """ train, valid, test : datasets
        W : the word embedding initialization
        n_words : vocabulary size
        n_x : word embedding dimension
        n_h : LSTM/GRU number of hidden units 
        n_z : latent embedding sapce for a sentence 
        patience : Number of epoch to wait before early stop if no progress
        max_epochs : The maximum number of epoch to run
        lrate : learning rate
        optimizer : methods to do optimization
        batch_size : batch size during training
        valid_batch_size : The batch size used for validation/test set
        dispFreq : Display to stdout the training progress every N updates
        validFreq : Compute the validation error after this number of update.
    """

    options = {}
    options['n_p'] = n_p
    options['n_words'] = n_words
    options['n_x'] = n_x
    options['n_h'] = n_h
    options['patience'] = patience
    options['max_epochs'] = max_epochs
    options['lrate'] = lrate
    options['optimizer'] = optimizer
    options['batch_size'] = batch_size
    options['valid_batch_size'] = valid_batch_size
    options['dispFreq'] = dispFreq
    options['validFreq'] = validFreq

    #if config.method in ['SVGD', 'SVGD_KFAC']: patience = 5

    logger.info('Model options {}'.format(options))
    
    logger.info('{} train examples'.format(len(train[0])))
    logger.info('{} valid examples'.format(len(valid[0])))
    logger.info('{} test examples'.format(len(test[0])))

    logger.info('Building model...')
    
    assert np.min(train[1]) == 0 and np.max(train[1]) == 1
    n_y = np.max(train[1]) + 1
    options['n_y'] = n_y
    
    params = init_params(options, W)
    tparams = init_tparams(params)

    (use_noise, x, mask, y, f_pred_prob, f_pred, cost, cache) = build_model(tparams,options)
    
    lr_theano = tensor.scalar(name='lr')
    ntrain_theano = tensor.scalar(name='ntrain')

    if config.method == 'pSGLD':
        f_grad_shared, f_update = pSGLD(tparams, cost, [x, mask, y], ntrain_theano, lr_theano)
    elif config.method == 'SGLD':
        f_grad_shared, f_update = SGLD(tparams, cost, [x, mask, y], ntrain_theano, lr_theano)
    elif config.method == 'RMSprop':
        f_grad_shared, f_update = RMSprop(tparams, cost, [x, mask, y], lr_theano)
    elif config.method == 'SVGD':
        f_grad_shared, f_update = SVGD(tparams, cost, [x, mask, y], ntrain_theano, lr_theano, kfac=False)
    elif config.method == 'SVGD_KFAC':
        f_grad_shared, f_update = SVGD(tparams, cost, [x, mask, y], ntrain_theano, lr_theano, kfac=True, average=True, cache=cache, eps=eps, n_p=n_p)
    elif config.method == 'MIXTURE_KFAC':
        f_grad_shared, f_update = SVGD(tparams, cost, [x, mask, y], ntrain_theano, lr_theano, kfac=True, average=False, cache=cache, eps=eps, n_p=n_p)

    #print 'Training model...'
    logger.info('Training model...')
    
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    estop = False  # early stop
    history_errs = []
    best_train_err, best_valid_err, best_test_err = 0., 0., 0.
    bad_counter = 0    
    uidx = 0  # the number of update done
    start_time = time.time()
    
    n_average = 0
    train_probs = np.zeros((len(train[0]),n_y))
    valid_probs = np.zeros((len(valid[0]),n_y))
    test_probs = np.zeros((len(test[0]),n_y))
    
    try:
        for eidx in xrange(max_epochs):
            print tparams.keys()
            from optimizers import sqr_dist
            ##['Wemb', 'lstm_encoder_W', 'lstm_encoder_U', 'lstm_encoder_rev_W', 'lstm_encoder_rev_U', 'Wy']
            tv = tensor.flatten(tparams['Wy'], 2)
            ftv = theano.function([], sqr_dist(tv, tv))
            otv = ftv()
            print(np.min(otv), np.max(otv), np.mean(otv), np.median(otv), np.sum(otv**2)/n_p)

            n_samples = 0
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                #use_noise.set_value(0.5)
                use_noise.set_value(config.dropout)

                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]
                                
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                if config.method == 'RMSprop':
                    f_update(lrate)
                elif config.method in ['SVGD', 'pSGLD', 'SGLD']:
                    f_update(lrate,n_train)
                elif config.method in ['SVGD_KFAC', 'MIXTURE_KFAC']:
                    f_update(lrate,n_train, x, mask, y)

                if np.isnan(cost) or np.isinf(cost):
                    
                    logger.info('NaN detected')
                    estop = True
                    break
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    logger.info('Epoch {} Update {} Cost {}'.format(eidx, uidx, cost))

                if np.mod(uidx, saveFreq) == 0:
                    logger.info('Saving ...')
                    saveto = 'results/%s.npz' % save_prefix
                    np.savez(saveto, history_errs=history_errs)
                    
                    logger.info('Done ...')

                if np.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    
                    if eidx < 1 :
                        train_err = pred_error(f_pred, prepare_data, train, kf)
                        valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
                        test_err = pred_error(f_pred, prepare_data, test, kf_test)
                        history_errs.append([valid_err, test_err, train_err])
                    else:
                        train_probs_curr = pred_probs(f_pred_prob, prepare_data, train, kf, options)
                        valid_probs_curr = pred_probs(f_pred_prob, prepare_data, valid, kf_valid, options)
                        test_probs_curr = pred_probs(f_pred_prob, prepare_data, test, kf_test, options)
                        train_probs = (n_average * train_probs + train_probs_curr)/(n_average+1)                       
                        valid_probs = (n_average * valid_probs + valid_probs_curr)/(n_average+1) 
                        test_probs = (n_average * test_probs + test_probs_curr)/(n_average+1) 
                        n_average += 1

                        train_pred = train_probs.argmax(axis=1)
                        valid_pred = valid_probs.argmax(axis=1)
                        test_pred = test_probs.argmax(axis=1)

                        train_err = (train_pred == np.array(train[1])).sum()
                        train_err = 1. - numpy_floatX(train_err) / len(train[0])

                        valid_err = (valid_pred == np.array(valid[1])).sum()
                        valid_err = 1. - numpy_floatX(valid_err) / len(valid[0])
                        
                        test_err = (test_pred == np.array(test[1])).sum()
                        test_err = 1. - numpy_floatX(test_err) / len(test[0])
                        history_errs.append([valid_err, test_err, train_err])
                    
                    if (uidx == 0 or
                        valid_err <= np.array(history_errs)[:,0].min()):

                        best_train_err = train_err
                        best_valid_err = valid_err
                        best_test_err = test_err
                        bad_counter = 0

                    logger.info('Train {} Valid {} Test {}'.format(train_err, valid_err, test_err))

                    if (len(history_errs) > patience and
                        valid_err >= np.array(history_errs)[:-patience,0].min()):
                        #valid_err >= np.array(history_errs)[:-patience,0].mean()):
                        bad_counter += 1
                        #valid_err >= np.array(history_errs)[:-patience,0].mean()):
                        if bad_counter > patience:
                            
                            logger.info('Early Stop!')
                            estop = True
                            break

            logger.info('Seen {} samples'.format(n_samples))

            if estop:
                break

    except KeyboardInterrupt:
        
        logger.info('Training interupted')

    end_time = time.time()
    logger.info('Train {} Valid {} Test {}'.format(best_train_err, best_valid_err, best_test_err))

    saveto = 'results/%s.npz' % save_prefix
    np.savez(saveto, train_err=best_train_err,
             valid_err=best_valid_err, test_err=best_test_err,
             history_errs=history_errs)
    
    logger.info('The code run for {} epochs, with {} sec/epochs'.format(eidx + 1, 
                 (end_time - start_time) / (1. * (eidx + 1))))
    
    #print >> sys.stderr, ('Training took %.1fs' %
    #                      (end_time - start_time))
    return best_train_err, best_valid_err, best_test_err
    

if __name__ == '__main__':

    # Set the random number generators' seeds for consistency
    SEED = 123  
    np.random.seed(SEED)

    # create logger with 'eval_bookcorpus'
    # https://docs.python.org/2/howto/logging-cookbook.html
    logger = logging.getLogger('eval_%s' % save_prefix)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('logs/eval_%s.log' % save_prefix, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('loading data...')
    x = cPickle.load(open("./benchmark/%s.p" % config.dataset,"rb"))

    if config.dataset == 'trec':
        train_x, test_x, train_y, test_y, W, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        train_x = [get_idx_from_sent(sent.lower(), word_idx_map) for sent in train_x]
        test_x = [get_idx_from_sent(sent.lower(), word_idx_map) for sent in test_x]

        train = (train_x, train_y)
        test = (test_x, test_y)
    else:
        revs, W, word_idx_map, vocab = x[0], x[1], x[2], x[3]

    del x

    logger.info('data loaded!')
    n_words = W.shape[0]

    patience, lrate = 10, 0.001

    config.eps = 5e-3 
    if 'KFAC' in config.method: lrate = 0.005

    results = []
    r = range(0, 10)
    for i in r:
        if config.dataset != 'trec':
            train, test = make_idx_data_cv(revs, word_idx_map, i)
        else:
            np.random.seed(i) #reset random seed

        train, valid = create_valid(train, valid_portion=0.1)
        max_epochs=20
        if config.method == 'SGLD': lrate=0.05
        #if config.method == 'SVGD_KFAC': lrate=5e-4
        [train_err, valid_err, test_err] = train_classifier(train, valid, test, 
            W, n_p=config.n_p, n_words=n_words, n_train=len(train[0]), max_epochs=max_epochs, batch_size=50, optimizer=config.method, validFreq=50,
            lrate=lrate, n_h=50, patience=patience, eps=config.eps)

        logger.info('cv: {} test err: {}'.format(i, test_err))
        results.append(test_err)

    logger.info('final test err: {} {}'.format(np.mean(results), np.std(results)))

    logger.info('Saving ...')             
    np.savez("results/%s.npz" % save_prefix, test_err=results)
