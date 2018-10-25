import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

#method = ['svgd', 'map_kfac', 'svgd_kfac', 'mixture_kfac']
method = ['svgd', 'svgd_kfac', 'mixture_kfac']
datasets = ['boston', 'concrete', 'energy', 'kin8nm', 'naval', 'combined', 'wine', 'yacht', 'protein', 'year']
#method = ['svgd', 'svgd_kfac']
#datasets = ['boston']

for ds in datasets:
    #if ds not in table: table[ds] = {}
    rmse, ll = [], []
    # alpha approach
    for m in method:
        #if m not in table[ds]: table[ds][m] = {}
        res_file_path = '%s_test_ll_rmse_%s-%s.txt' % (ds, m, ds)
        test_ll_rmse = np.loadtxt(res_file_path, delimiter=',', usecols=[3,4])
        n_samples = len(test_ll_rmse)
        if ds in ['protein', 'year']:
            assert  n_samples == 5
        else:
            assert  n_samples == 20
        rmse.append((np.mean(test_ll_rmse[:,0]), np.std(test_ll_rmse[:,0]) / np.sqrt(n_samples)))
        ll.append((np.mean(test_ll_rmse[:,1]), np.std(test_ll_rmse[:,1])/ np.sqrt(n_samples)))

    print ds
    print method
    print ds,',', [ '%.3f#%.3f' % (v, s) for (v,s) in rmse]
    print ds,',', [ '%.3f#%.3f' % (v, s) for (v,s) in ll]

