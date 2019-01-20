import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano import config
from gmm_models import mixture_weights_and_grads


def sqr_dist(x, y, e=1e-8):
    if x.ndim == 2:
        xx = T.sqr(T.sqrt((x*x).sum(axis=1) + e))
        yy = T.sqr(T.sqrt((y*y).sum(axis=1) + e))
        dist = T.dot(x, y.T)
        dist *= -2.
        dist += xx.dimshuffle(0, 'x')
        dist += yy.dimshuffle('x', 0)
    else:
        raise NotImplementedError
    return dist


def median_distance(H, e=1e-6):
    if H.ndim != 2:
        raise NotImplementedError

    V = H.flatten()
    # median distance
    h = T.switch(T.eq((V.shape[0] % 2), 0),
        # if even vector
        T.mean(T.sort(V)[ ((V.shape[0] // 2) - 1) : ((V.shape[0] // 2) + 1) ]),
        # if odd vector
        T.sort(V)[V.shape[0] // 2])
    #h = h / T.log(H.shape[0] + 1).astype(theano.config.floatX)
    return h


def poly_kernel(x, e=1e-8):
    x = x - T.mean(x, axis=0)
    kxy = 1 + T.dot(x, x.T)
    dxkxy = x * x.shape[0].astype(theano.config.floatX)

    return kxy, dxkxy


def imq_kernel(x, h=-1):
    H = sqr_dist(x, x)
    if h == -1:
        h = median_distance(H)

    kxy = 1. / T.sqrt(1. + H / h) 

    dk = -.5 * kxy / (1. + H / h)
    dxkxy = T.dot(dk, x)
    sumkxy = T.sum(dk, axis=1, keepdims=True)
    dxkxy = (dxkxy - x * sumkxy) * 2. / h

    return kxy, dxkxy


def rbf_kernel(x, H=None, h='median', to3d=False):
    assert x.ndim == 2, 'rbf kernel, illegal shapes'
    if H is None:
        H = sqr_dist(x, x)

    if h == 'median':
        h = median_distance(H)

    kxy = T.exp(-H / h)
    dxkxy = -T.dot(kxy, x)
    sumkxy = T.sum(kxy, axis=1).dimshuffle(0, 'x')
    dxkxy = (dxkxy + x * sumkxy) * 2. / h

    if to3d: dxkxy = -(x.dimshuffle(0, 'x', 1) - x.dimshuffle('x', 0, 1)) * kxy.dimshuffle(0, 1, 'x') * 2. / h
    return kxy, dxkxy


def svgd_gradient(p, g, ntrain, h='median', kernel='rbf'):
    assert p.ndim == g.ndim, 'illegal shapes'
    p_shape = p.shape
    if p.ndim > 2:
        p = T.flatten(p, 2)
        g = T.flatten(g, 2)

    if kernel == 'rbf':
        kxy, dxkxy = rbf_kernel(p, h=h)
    elif kernel == 'imq':
        kxy, dxkxy = imq_kernel(p)
    else:
        raise NotImplementedError
    svgd_grad = (T.dot(kxy, g) + dxkxy/ntrain) / T.sum(kxy, axis=1).dimshuffle(0, 'x')
    #svgd_grad = (T.dot(kxy, g) + dxkxy/ntrain) / kxy.shape[0].astype(theano.config.floatX)
    return T.reshape(svgd_grad, p_shape)


def SGD(tparams, cost, inps, lr):
    """ default: lr=0.01 """
    
    grads = T.grad(cost, tparams.values())
    norm = T.sqrt(sum([T.sum(g**2) for g in grads]))
    if T.ge(norm, 5):
        grads = [g*5/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)
    
    updates = []

    for p, g in zip(tparams.values(), gshared):        
        updated_p = p - lr * g
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 


def RMSprop(tparams, cost, inps, lr, rho=0.9, epsilon=1e-6, clip_norm=5.):
    """ default: lr=0.001 
        This is the implementation of the RMSprop algorithm used in
        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf.
    """
    
    grads = T.grad(cost, tparams.values())
    norm = T.sqrt(sum([T.sum(g**2)/ T.cast(g.shape[0], theano.config.floatX) for g in grads]))
    if T.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []

    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc, acc_new))
        updated_p = p - lr * (g / T.sqrt(acc_new + epsilon))
        updates.append((p, updated_p))

    f_update = theano.function([lr], [], updates=updates)
    return f_grad_shared, f_update


#### from https://github.com/zhegan27/Bayesian_RNN ###
def SGLD(tparams, cost, inps, ntrain, lr):
    """ default: lr=0.01 """

    trng = RandomStreams(123)

    grads = T.grad(cost, tparams.values())
    norm = T.sqrt(sum([T.sum(g**2) for g in grads]))
    if T.ge(norm, 5):
        grads = [g*5/norm for g in grads]

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k)
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)

    updates = []

    for p, g in zip(tparams.values(), gshared):
        eps = trng.normal(p.get_value().shape, avg = 0.0, std = 1.0, dtype=theano.config.floatX)
        updated_p = p - lr * (g-p/ntrain) + T.sqrt(lr*2./ntrain) * eps
        updates.append((p, updated_p))

    f_update = theano.function([lr,ntrain], [], updates=updates)

    return f_grad_shared, f_update

      
#### from https://github.com/zhegan27/Bayesian_RNN ###
def pSGLD(tparams, cost, inps, ntrain, lr, rho=0.9, epsilon=1e-6, clip_norm=5):
    """ default: lr=0.001 """
    
    trng = RandomStreams(123)
    
    grads = T.grad(cost, tparams.values())
    norm = T.sqrt(sum([T.sum(g**2) / T.cast(g.shape[0], theano.config.floatX) for g in grads]))
    if T.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []
    
    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc, acc_new))
        
        G = T.sqrt(acc_new + epsilon)
        
        eps = trng.normal(p.get_value().shape, avg = 0.0, std = 1.0, 
                          dtype=theano.config.floatX)
        
        updated_p = p - lr * (g-p/ntrain) / G + T.sqrt(lr/G)*2./ntrain * eps 
        updates.append((p, updated_p))
    
    f_update = theano.function([lr,ntrain], [], updates=updates)
    
    return f_grad_shared, f_update



def SVGD(tparams, cost, inps, ntrain, lr, average=True, kfac=False, cache=None, rho=0.9, epsilon=1e-6, clip_norm=5, eps=1e-3, n_p=10):
    # for the \ell-th cluster
    def _weighted_svgd(x, d_log_pw, w, g_scale):
        kxy, dxkxy = rbf_kernel(x, to3d=True)
        velocity = T.sum(w.dimshuffle('x', 0, 'x') * kxy.dimshuffle(0, 1, 'x')  * d_log_pw.dimshuffle('x', 0, 1), axis=1) + \
                    T.sum(w.dimshuffle(0, 'x', 'x') * dxkxy / g_scale, axis=0)
        # n * d , d x d
        return velocity


    trng = RandomStreams(123)
    grads = T.grad(cost, tparams.values())
    norm = T.sqrt(sum([T.sum(g**2) / T.cast(g.shape[0], theano.config.floatX) for g in grads]))
    ratio = clip_norm / norm
    if T.ge(norm, clip_norm):
        grads = [g*ratio for g in grads]
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     

    updates = []
    step = theano.shared(np.int32(1))
    for n, p, g in zip(tparams.keys(), tparams.values(), gshared):
        if n == 'Wy':
            vg = svgd_gradient(p, -(g+p/ntrain), ntrain, h='median', kernel='rbf')
            if kfac:
                cov_dW = cache['Wy']
                cov_acc = theano.shared(1e-2 * np.tile(np.expand_dims(np.eye(p.get_value().shape[1]), 0), (n_p,1,1)).astype('float32'))
                cov_acc_new = T.cast(rho * cov_acc + (1.0 - rho) * cov_dW, theano.config.floatX)

                H_invs, _ = theano.scan( fn=lambda v:T.nlinalg.matrix_inverse(v+ eps*T.eye(v.shape[0])), sequences=cov_acc_new)
                updates.append((cov_acc, cov_acc_new))
                # average hessian
                if average:
                    vg = T.dot(vg, T.mean(H_invs, 0).astype(config.floatX))
                # mixture hessian
                else:
                    ps = T.squeeze(p) # n x d
                    gs = T.squeeze( -(g+p/ntrain) ) # n x d
                    assert ps.ndim == 2 and gs.ndim == 2
                    mix, mix_grads = mixture_weights_and_grads(ps)
                    velocity = 0.
                    for i in range(n_p):
                        w_i_grad = _weighted_svgd(ps, gs+ mix_grads[i] / ntrain, mix[i], ntrain)
                        velocity += mix[i].dimshuffle(0, 'x') * T.dot(w_i_grad, H_invs[i])
                    vg = T.reshape(velocity, p.shape).astype(theano.config.floatX)
        else:
            vg = -g # graident descent

        #RMSprop
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1. - rho) * vg ** 2
        updates.append((acc, acc_new))
        G = T.sqrt(acc_new) + epsilon
        updated_p = p + lr * vg / G
        updates.append((p, updated_p))

    updates.append((step, step+1))  # update step
    if cache is not None:
        f_update = theano.function([lr,ntrain] + inps, [], updates=updates)
    else:
        f_update = theano.function([lr,ntrain], [], updates=updates)
    return f_grad_shared, f_update


