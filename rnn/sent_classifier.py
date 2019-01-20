import numpy as np
import theano
import theano.tensor as tensor
from theano import config

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import dropout, numpy_floatX
from utils import uniform_weight, zero_bias

from lstm_layers import param_init_encoder, lstm_encoder, gru_encoder

# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

""" init. parameters. """  

def init_params(options,W):
    
    n_p = options['n_p']
    n_h = options['n_h']
    n_y = options['n_y']
    
    params = OrderedDict()
    # W is initialized by the pretrained word embedding
    params['Wemb'] = W.astype(config.floatX)
    # otherwise, W will be initialized randomly
    # n_words = options['n_words']
    # n_x = options['n_x'] 
    # params['Wemb'] = uniform_weight(n_words,n_x)
    
    # bidirectional LSTM
    #params = param_init_encoder(options,params, prefix="lstm_encoder")
    #params = param_init_encoder(options,params, prefix="lstm_encoder_rev")
    params = param_init_encoder(options,params, prefix="gru_encoder")
    params = param_init_encoder(options,params, prefix="gru_encoder_rev")

    #params['Wy'] = uniform_weight(n_p, 2*n_h+1, scale=0.1)
    params['Wy'] = np.squeeze(uniform_weight(n_p, 2*n_h, 1))
    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
        #tparams[kk].tag.test_value = params[kk]
    return tparams
    
""" Building model... """
def build_model(tparams,options):

    n_p = options['n_p']
    trng = RandomStreams(SEED)
    
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))
    
    # input sentence: n_steps * n_samples
    x = tensor.matrix('x', dtype='int32')
    mask = tensor.matrix('mask', dtype=config.floatX) 
    
    # label: (n_samples,)
    y = tensor.vector('y',dtype='int32')

    n_steps = x.shape[0] # the length of the longest sentence in this minibatch
    n_samples = x.shape[1] # how many samples we have in this minibatch
    n_x = tparams['Wemb'].shape[1] # the dimension of the word-embedding
    
    emb = tparams['Wemb'][x.flatten()].reshape([n_steps,n_samples,n_x])  
    emb = tensor.tile(emb.dimshuffle(0, 'x', 1, 2), (1, n_p, 1, 1))
    #emb = dropout(emb, trng, use_noise) # n_steps * n_p * n_samples * n_x

    # encoding of the sentence, size of n_samples * n_h                                                               
    #h_encoder = lstm_encoder(tparams, emb, mask, prefix='lstm_encoder')
    #h_encoder_rev = lstm_encoder(tparams, emb[::-1], mask[::-1], prefix='lstm_encoder_rev')
    h_encoder = gru_encoder(tparams, emb, mask, prefix='gru_encoder')
    h_encoder_rev = gru_encoder(tparams, emb[::-1], mask[::-1], prefix='gru_encoder_rev')

    ## size of n_samples * (2*n_h) 
    #z = tensor.concatenate((h_encoder,h_encoder_rev),axis=1) 
    # size of n_p * n_samples * (2*n_h) 
    h = tensor.concatenate((h_encoder,h_encoder_rev),axis=-1) 
    #z = dropout(z, trng, use_noise)  # n_p x n_samples x (2*n_h)
    #z = tensor.concatenate( (z, tensor.ones((z.shape[0], z.shape[1], 1), config.floatX)), axis=-1 )

    z = tensor.sum(h * tparams['Wy'].dimshuffle(0, 'x', 1), -1) # n_p * B
    y_prob = tensor.nnet.nnet.sigmoid(z) #n_p * B

    y_expand = y.dimshuffle('x', 0).astype(config.floatX) # 1 * B
    dy = (y_expand - y_prob) / (y_prob - y_prob**2) # n_p * B
    dz = tensor.nnet.nnet.sigmoid(z) * (1. - tensor.nnet.nnet.sigmoid(z)) # n_p * B
    dW = tensor.mean(h * (dz * dy).dimshuffle(0, 1, 'x'), 1) # n_p * 2nh

    grad_loglik_z = (y_prob - y_expand) / (y_prob - y_prob**2) * dz 
    grad_loglik_W = grad_loglik_z.dimshuffle(0, 1, 'x') * h # n_p * B * d

    mean_dW = tensor.mean(grad_loglik_W, axis=1, keepdims=True) # n_p * 1 * d
    diff_dW = grad_loglik_W - mean_dW # n_p * B * d
    cov_dW = tensor.batched_dot(diff_dW.dimshuffle(0, 2, 1), diff_dW) / n_samples #n_p * d * d

    # n_p x n_samples
    pred = tensor.mean(y_prob, axis=0).dimshuffle(0, 'x') # n_samples, 1
    pred = tensor.concatenate((1.0-pred, pred), axis=1) # n_samples * ny

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], tensor.argmax(pred, 1), name='f_pred')

    # get the expression of how we calculate the cost function
    # i.e. corss-entropy loss
    index = tensor.arange(n_samples)

    #cost = -tensor.sum(tensor.mean(tensor.log(p_pred[:, index, y] + 1e-6), axis=1))
    cost = -tensor.mean(y_expand*tensor.log(y_prob+1e-6) + (1.-y_expand)*tensor.log(1.-y_prob+1e-6), axis=1).sum()

    cache = {}
    cache['Wy'] = cov_dW
    return use_noise, x, mask, y, f_pred_prob, f_pred, cost, cache
    

