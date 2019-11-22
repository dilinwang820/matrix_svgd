import numpy as np
import matplotlib

def sample_langevin(n, model, max_iter = 10000, a = 0.01, b = 1, c = 0.55):
	d = model.dimension
	x = np.zeros([n, d])
	for t in range(max_iter):
		epsilon = a*np.exp(np.log(b+t) *(- c))
		x += epsilon * model.grad_log_p(x)
		x += 2 * np.sqrt(epsilon) * np.random.randn(n,d)
	# plt.scatter(x[:,0], x[:,1])
	return x
