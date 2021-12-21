import math

import scipy
import torch
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

import ot
from ot.datasets import make_1D_gauss as gauss

from utils.tools import to_numpy, is_point_in_sphere


def corrcoef(X, y):
	X, y = to_numpy(X), to_numpy(y)

	if len(X.shape) == 1:
		X = X.reshape(1, -1)

	Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
	ym = np.mean(y)
	r_num = np.sum((X - Xm) * (y - ym), axis=1)
	r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=1) * np.sum((y - ym) ** 2))
	r = r_num / (r_den + 1e-6)

	return r.mean()


def mse(X, y):
	if len(y.shape) == 1:
		y = np.tile(y, (X.shape[0], 1))
	return mean_squared_error(X, y)


def rmse(X, y):
	return np.sqrt(mse(X, y))


def kl_divergence(X, Y, verbose=False):
	BIN_SIZE = 1.0

	assert len(X.shape) == 2 and len(Y.shape) == 2
	assert X.shape[1] == Y.shape[1]

	tot_x = X.shape[0]
	tot_y = Y.shape[0]
	P = X.shape[1]
	result_list = []
	for i in range(P):
		min_bin = math.floor(Y[:, i].min())
		max_bin = math.floor(Y[:, i].max())
		res = 0
		for j in range(min_bin, max_bin + 1):
			# print((j <= X[:, i]) & (X[:, i] < j + 1), (j <= Y[:, i]) & (Y[:, i] < j + 1))
			cnt_x = ((j <= X[:, i]) & (X[:, i] < j + 1)).sum()
			cnt_y = ((j <= Y[:, i]) & (Y[:, i] < j + 1)).sum()
			if cnt_y == 0 or cnt_x == 0:
				continue
			res += (cnt_x / tot_x) * np.log((cnt_x / tot_x) / (cnt_y / tot_y))
		# if verbose and i == 1:
		#     for x in X[:, 1]:
		#         print(f'{x:.3f}', end=",")
		#     print("||", end="")
		#     for x in Y[:, 1]:
		#         print(f'{x:.3f}', end=",")
		#     print("||", end="")
		#     print(f'{res}')
		result_list.append(res)
	return np.mean(result_list)


def wasserstein_distance(X, Y, p=2, verbose=False):
	"""
	Test:
		metrics.wasserstein_distance(np.array([[1.0], [6.0]]), np.array([[4.0], [2.0]]))
		2.5
		---
		wasserstein_distance(np.array([[1.0], [6.0]]), np.array([[4.0], [2.0], [7.0]]))
		2.833
		---
		metrics.wasserstein_distance(np.array([[1.0, 2.0], [6.0, 1.0]]), np.array([[4.0, -1.0], [2.0, 3.0], [7.0, 0.0]]))
		2.380
		---
	"""
	X, Y = to_numpy(X), to_numpy(Y)

	if len(X.shape) == 1:
		X = X.reshape(-1, 1)
	if len(Y.shape) == 1:
		Y = Y.reshape(-1, 1)

	assert len(X.shape) == 2 and len(Y.shape) == 2
	assert X.shape[1] == Y.shape[1]

	# X = X[:, :1]
	# Y = Y[:, :1]

	if p == 2:
		M = ot.dist(X, Y, 'sqeuclidean')
		M /= X.shape[1]
	elif p == 1:
		M = ot.dist(X, Y, 'euclidean')
		M /= X.shape[1]
	else:
		raise NotImplementedError

	# print(M)

	nx = X.shape[0]
	ny = Y.shape[0]
	res = ot.emd2(np.ones(nx) / nx, np.ones(ny) / ny, M)

	if p == 2:
		res = res ** (1 / p)
	elif p == 1:
		res = res
	else:
		raise NotImplementedError

	assert isinstance(res, float)
	return res

def precision(G, L):
	"""
	calculate the ratio of the number of label samples which is covered over the total number of labeled samples
	:param G: generated samples
	:param L: labeled samples
	:return: int, the precision metrics \in [0, 1]
	"""
	G, L = to_numpy(G), to_numpy(L)
	assert isinstance(G, np.ndarray) and isinstance(L, np.ndarray)
	assert G.ndim == 2 and L.ndim == 2
	assert G.shape[1] == L.shape[1]
	Ng, P = G.shape
	Nl, _ = L.shape
	counter = 0
	radius2 = scipy.stats.chi2.ppf(0.01, df=P)
	# print(radius2)
	for i in range(Nl):
		center = L[i]
		flag = False
		for j in range(Ng):
			if is_point_in_sphere(G[j], center, radius2):
				flag = True
				break
		if flag is True:
			counter += 1
	return counter / Nl

def recall(G, L):
	return precision(L, G)