import numpy as np
import matplotlib
font = {'family' : 'normal',
        'size'   : 34}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from pylab import rcParams
rcParams['figure.figsize'] = 7, 7
from environment import star_gaussian, double_banana, sine
color_map = [(0.3016, 0.3016, 0.3016),  (0.1974, 0.5129, 0.7403),  (1, 0, 0), (0.75, 0.75, 0)]
lineWidth = 7.0
label_name = ['Vanilla SVGD', 'Matrix SVGD(average)', 'Matrix SVGD(mixture)', 'SVN']

def make_video(env, filename, kernel_type, seed = 0):
	Iteration = range(0,300,10)#[0,5,10,30,100,300]
	ngrid = 350
	# set the line space carefully to the region of your figure
	x = np.linspace(-3.3, 3.7, ngrid)
	y = np.linspace(-3.5, 3.5, ngrid)
	X, Y = np.meshgrid(x, y)
	Z = np.exp(env.logp(np.vstack( (np.ndarray.flatten(X), np.ndarray.flatten(Y))).T)).reshape(ngrid, ngrid)
	
	for iteration in Iteration:
		plt.xlim([-3.5, 3.5])
		plt.ylim([-3.3, 3.7])
		plt.axis('off')
		plt.contourf(Y, X, Z, 10, cmap = 'Blues')
		x = np.load(filename)
		plt.scatter(x[seed,iteration,:,1], x[seed,iteration,:,0], color = '#FF0000', marker = 'X', s = 300, alpha = 1.)
		plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.0, top = 0.9)
		plt.title(kernel_type + '_iter = {}'.format(iteration))
		# plt.show()
		plt.draw()
		plt.pause(0.1)
		plt.clf()
