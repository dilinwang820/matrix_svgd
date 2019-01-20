import numpy as np

class gaussian_kernel(object):
	def __init__(self, Q, adaptive = False, decay = False):
		self.Q = 0.5*(Q+Q.T)
		self.d = Q.shape[0]
		self.adaptive = adaptive
		self.decay = decay

	def calculate_kernel(self, x):
		n,d = x.shape
		diff = x[:, None, :] - x[None, :, :]
		Qdiff = np.matmul(diff, self.Q)
		if self.adaptive:
			h = np.mean(np.sum(diff * Qdiff, axis = -1))	# faster calculation, for small number of particles should use median distant
			# h = np.median(np.sum(diff * Qdiff, axis = -1))
			if self.decay:
				h /= 10.
			else:
				h /= 2.
			h /= np.log(n)
		else:
			h = self.d
		K = np.exp(-np.sum(Qdiff * diff, axis = -1)/(2.*h))
		gradK = -Qdiff * K[:,:,None]/h
		return K, gradK

# scalar rbf kernel use median distant trick as bandwidth
class rbf_kernel(gaussian_kernel):
	def __init__(self, d, decay = False):
		gaussian_kernel.__init__(self, np.eye(d))
		self.d = d
		self.adaptive = True
		self.decay = decay
