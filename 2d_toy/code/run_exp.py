from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import argparse

import numpy as np
from environment import double_banana, sine, star_gaussian
from kernel import gaussian_kernel, rbf_kernel
from stein_samplers import matrix_SVGD, mixture_hessian_SVGD, SVN
from draw_fig import make_video

def Stein_sampler(model, n_particles, max_iter, step_size, seed = 44, 
				adagrad = True, kernel_type = 'mixture'):
	np.random.seed(seed)
	d = model.dimension
	sig = 1.5
	
	x_initial = sig * np.random.randn(n_particles, d)	# set initial particles as standard normal
	
	x = x_initial

	EVOLVEX = np.zeros([max_iter+1, n_particles, d])
	EVOLVEX[0,:,:] = x

	adag = np.zeros([n_particles, d])	# for adagrad
	for i in range(max_iter):
		grad_logp = model.grad_log_p(x)		# gradient information for each particles: n*d
		Hs = model.Hessian_log_p(x)			# Hessian information for each particles: n*d*d
		A = np.mean(Hs, axis = 0)			# Average Hessian: d*d

		if kernel_type == 'newton':			# SVN
			v = SVN(x, grad_logp, Hs)
		elif kernel_type == 'mixture':		# matrix SVGD(mixture)
			v = mixture_hessian_SVGD(x, grad_logp, Hs)
		else:
			if kernel_type == 'gaussian':	# matrix SVGD(average)
				kernel = gaussian_kernel(A)
				B = model.inv_avg_Hessian(A)
			else:							# vanilla SVGD
				if i > 30:
					kernel = rbf_kernel(d, decay = True)
				else:
					kernel = rbf_kernel(d)
				B = np.eye(d)
			v = matrix_SVGD(x, grad_logp, kernel, B)

		adag += v ** 2 		# update sum of gradient's square
		if adagrad:
			x = x + step_size * v / np.sqrt(adag + 1e-12)
		else:
			x = x + step_size * v
		EVOLVEX[i+1,:,:] = x
	
	return EVOLVEX


if __name__ == '__main__':
	# Pick one toy environment to play
	env_name = 'star'
	# env_name = 'sine'
	# env_name = 'double_banana'

	if env_name == 'star':
		env = star_gaussian(100, 5)								# star gaussian mixture example
	elif env_name == 'sine':
		env = sine(1., 0.003)									# unimodal sine shape example
	elif env_name == 'double_banana':
		env = double_banana(0.0, 100.0, 1.0, 0.09, np.log(30))	# bimodal double banana example

	kernel_type = ['rbf', 'gaussian', 'mixture', 'newton']
	n_methods = 4

	n_particles = 50
	dim = 2
	n_repeatition = 1
	test_iteration = 300
	
	for j in range(n_methods):
		print('~~~~~Current kernel type = '+kernel_type[j]+'~~~~~')
		EVOLVEXS = np.zeros([n_repeatition, test_iteration+1 , n_particles, dim])
		for seed in range(n_repeatition):
			print('####seed = {}####'.format(seed))
			EVOLVEXS[seed,:,:,:] = Stein_sampler(env, n_particles, test_iteration, 0.7, kernel_type = kernel_type[j], adagrad = True, seed = seed+40)
		np.save('../data/'+env_name+'_'+kernel_type[j]+'.npy', EVOLVEXS)

	for j in range(n_methods):
		make_video(env, '../data/'+env_name+'_'+kernel_type[j]+'.npy', kernel_type[j], seed = 0)
