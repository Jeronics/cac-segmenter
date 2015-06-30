import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)
X_1 = np.ones([1, 100]) * 10
X_2 = np.ones([1, 100]) * 30
X = np.concatenate([X_1, X_2], axis=1)
X = np.concatenate([X, X, X * 2 / 3.]).T

print X.shape
# Fit a mixture of Gaussians with EM using five components
gmm = mixture.GMM(n_components=2, covariance_type='full')
gmm.fit(X)

# # Fit a Dirichlet process mixture of Gaussians using five components
# dpgmm = mixture.DPGMM(n_components=5, covariance_type='full')
# dpgmm.fit(X)

print gmm.means_
print gmm.covars_