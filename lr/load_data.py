import math
import time
import numpy as np

import sys
import numpy as np
from math import pi

import scipy.io
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_svmlight_file

def load_uci_dataset(dataset='covtype', random_state=1):
    data = scipy.io.loadmat('covertype.mat')
    if dataset == 'covtype':
        X_input = data['covtype'][:, 1:]
        y_input = data['covtype'][:, 0]
        y_input[y_input == 2] = 0

        X_input = np.hstack([X_input, np.ones([len(X_input), 1])])

        # split the dataset into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=random_state)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)
    else:
        raise NotImplementedError

    return X_train, X_valid, X_test, y_train, y_valid, y_test

