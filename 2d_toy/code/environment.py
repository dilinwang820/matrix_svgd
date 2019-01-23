import numpy as np
from scipy.stats import multivariate_normal

class double_banana(object):
	dimension = 2
	def __init__(self, a, b, prior_var, y_var, y):
		self.a = a
		self.b = b
		self.prior_var = prior_var
		self.y_var = y_var
		self.y = y

	def F(self, x):
		x1 = x[:, 0]
		x2 = x[:, 1]
		expFx = np.square(self.a-x1) + self.b * np.square(x2 - x1**2) + 1e-10
		Fx = np.log(expFx)
		Jx1 = 2.0 * (x1-self.a) + 4.0 * self.b * x1 * (x1**2 - x2)
		Jx2 = 2.0 * self.b * (x2 - x1 ** 2)
		Jx = np.array([Jx1, Jx2]).T/expFx[:, None]
		return Fx, Jx

	def logp(self, x):
		Fx, _ = self.F(x)
		return -np.sum(x*x, axis = -1)/(2*self.prior_var) - (Fx - self.y)**2/(2*self.y_var)

	def grad_log_p(self, x):
		Fx, Jx = self.F(x)
		return -x/self.prior_var - Jx * (Fx - self.y)[:, None] / self.y_var
		# return np.clip(-x/self.prior_var - Jx * (Fx - self.y)[:, None] / self.y_var, -50., 50.)

	def Hessian_log_p(self, x):
		Fx, Jx = self.F(x)
		n,d = x.shape
		return np.eye(d)/self.prior_var + Jx[:,:,None]*Jx[:,None,:] / self.y_var

	def inv_avg_Hessian(self, Q):
		return np.linalg.inv(Q)

class sine(object):
	dimension = 2
	def __init__(self, prior_var, y_var):
		self.prior_var = prior_var
		self.y_var = y_var

	def F(self, x):
		x1 = x[:, 0]
		x2 = x[:, 1]
		Fx = np.square(x2 + np.sin(x1))

		Jx2 = 2 * (x2 + np.sin(x1)) 
		Jx1 = 2 * (x2 + np.sin(x1)) * np.cos(x1) 
		Jx = np.array([Jx1, Jx2]).T

		return Fx, Jx

	def logp(self, x):
		Fx, _ = self.F(x)
		return -np.sum(x*x, axis = -1)/(2*self.prior_var) - Fx **2/(2*self.y_var)

	def grad_log_p(self, x):
		Fx, Jx = self.F(x)
		return -x/self.prior_var - Jx * Fx[:, None] / self.y_var
		# return np.clip(-x/self.prior_var - Jx * Fx[:, None] / self.y_var, -50., 50.)

	# Use Gaussian-Newton approximation on Fx
	def Hessian_log_p(self, x):
		Fx, Jx = self.F(x)
		n,d = x.shape
		return np.eye(d)/self.prior_var + Jx[:,:,None]*Jx[:,None,:] / self.y_var

	def inv_avg_Hessian(self, Q):
		return np.linalg.inv(Q)

class star_gaussian(object):
	def __init__(self, skewness, n):
		self.d = 2
		self.dimension = 2
		self.K = n
		theta = 2*np.pi/n
		U = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])

		self.mu = np.zeros([self.K, self.d])
		self.sigma = np.zeros([self.K, self.d, self.d])
		self.inv_sigma = np.zeros_like(self.sigma)

		self.mu[0,:] = 1.5 * np.array([1.,0.])
		self.sigma[0,:,:] = np.diag([1.,1./skewness])
		self.inv_sigma[0,:,:] = np.diag([1.,skewness])

		for i in range(1,n):
			self.mu[i,:] = np.matmul(U,self.mu[i-1,:])
			self.sigma[i,:,:] = np.matmul(U, np.matmul(self.sigma[i-1,:,:],U.T))
			self.inv_sigma[i,:,:] = np.matmul(U, np.matmul(self.inv_sigma[i-1,:,:],U.T))

		self.mean = np.mean(self.mu)
		self.x2 = self.mean * self.mean * self.K
		for i in range(self.K):
			self.x2 += np.diag(self.sigma[i,:,:])
		self.x2 /= self.K

	def sample(self, n_samples):
		n = int(n_samples/self.K)
		x = np.zeros([self.K * n, self.d])
		for k in range(self.K):
			x[k*n:(k+1)*n,:] = np.random.multivariate_normal(self.mu[k,:], self.sigma[k,:,:], n)
		np.random.shuffle(x)
		return x

	def logp(self, x):
		n,d = x.shape
		Fx = np.zeros(n)
		for k in range(self.K):
			pdfi = multivariate_normal.pdf(x, mean = self.mu[k,:], cov = self.sigma[k,:,:])
			Fx += pdfi
		return np.log(Fx / self.K)

	def grad_log_p(self, x):
		n = x.shape[0]
		Fx = np.zeros(n)
		Jx = np.zeros_like(x)
		for k in range(self.K):
			pdfi = multivariate_normal.pdf(x, mean = self.mu[k,:], cov = self.sigma[k,:,:]) + 1e-20
			Fx += pdfi
			Jx += pdfi[:, None] * np.matmul(self.mu[k,:] - x, self.inv_sigma[k,:,:])
		return Jx/Fx[:, None]

	def Hessian_log_p(self, x):
		n, d = x.shape
		Fx = np.zeros(n)
		Hx = np.zeros([n,d,d])
		for k in range(self.K):
			pdfi = multivariate_normal.pdf(x, mean = self.mu[k,:], cov = self.sigma[k,:,:]) + 1e-20
			Fx += pdfi
			Hx += pdfi[:, None, None] * self.inv_sigma[k,:,:]
		return Hx/Fx[:, None, None]

	def inv_avg_Hessian(self, Q):
		return np.linalg.inv(Q)
