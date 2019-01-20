import numpy as np
from scipy.linalg import sqrtm
from kernel import gaussian_kernel, rbf_kernel
from scipy.stats import ortho_group

# H: n*d*d
def SVN(x, Dlogp, H):
	n,d = x.shape
	A = np.mean(H, axis = 0)
	v = np.zeros_like(x)
	for i in range(n):
		diff = x[i,None,:] - x 			# n*d
		Adiff = np.matmul(diff, A)	# n*d
		kern = np.exp(-0.5 * np.sum(diff * Adiff, axis = -1))	#n
		Dkern = Adiff * kern[:,None]							#n*d

		tmpv = np.mean(Dlogp * kern[:,None] + Dkern, axis = 0)	#d
		Hv = np.mean(H * kern[:,None,None] ** 2, axis = 0) + np.matmul(Dkern.T, Dkern)	#d*d
		v[i,:] = np.linalg.solve(Hv, tmpv)
	return v

# calculate weight and its Dlog(weight) in mixture SVGD
def soft_max_H(x, H, beta = 1.):
	n,d = x.shape
	diff = x[:,None,:] - x[None,:,:]
	Hdiff = np.sum(diff[:,:,:,None] * H[:,None,:,:], axis = 2)
	# Hdiff = np.zeros_like(diff)
	# for i in range(n):
		# Hdiff[i,:,:] = np.matmul(diff[i,:,:], H[i,:,:]) - 0.5 * np.log(np.linalg.det(H[i,:,:]))
	dist2H = np.sum(Hdiff * diff, axis = -1)
	for i in range(n):
		dist2H[i,:] -= np.log(np.linalg.det(H[i,:,:]))	# minus normalization for gaussian
	dist2H -= np.min(dist2H, axis = 0)
	ww = np.exp(-0.5* beta * dist2H)
	w = ww / np.sum(ww, axis = 0)
	# w = np.eye(n)
	Dlogw = beta * np.sum((Hdiff[:,None,:,:] - Hdiff[None,:,:,:]) * ww[None,:,:,None], axis = 1)/np.sum(ww, axis = 0)[None,:,None]
	return w, Dlogw

# A_i = H(x_i) = D^2 log p(x_i)
def weighted_Hessian_SVGD(x, Dlogp, A, w):
	n,d = x.shape
	invA = np.linalg.inv(A)
	# K, grad_K = gaussian_kernel(A).calculate_kernel(x)
	K, grad_K = gaussian_kernel(A, adaptive = True).calculate_kernel(x)
	velocity = np.sum(w[None,:,None] * K[:,:,None] * Dlogp[None,:,:], axis = 1) + np.sum(w[:,None,None] * grad_K, axis = 0)
	velocity = np.matmul(velocity, invA)
	return velocity

def mixture_hessian_SVGD(x, Dlogp, H, alpha = 0.5):
	n,d = x.shape

	H = H + np.zeros((n,1,1))	# make sure the shape is n*d*d
	avg_H = np.mean(H, axis = 0)
	H = (1-alpha) * avg_H + alpha * H 	# Mixed with average hessian for robustness

	w, Dlogw = soft_max_H(x, H)
	velocity = np.zeros_like(x)
	for i in range(n):
		velocity += w[i,:,None] * weighted_Hessian_SVGD(x, Dlogp + Dlogw[i,:,:], H[i,:,:], w[i,:])
		
	return velocity

# This is for separable kernel B k(x,x'), where k is a scalar kernel
def matrix_SVGD(x, grad_logp, scalar_kernel, B, temperature = 0.0):
	n,d = x.shape
	K, grad_K = scalar_kernel.calculate_kernel(x)
	velocity = (np.matmul(K, grad_logp) + (1.0 + temperature) * np.sum(grad_K, axis = 0))/n
	velocity = np.matmul(velocity, B)
	return velocity

