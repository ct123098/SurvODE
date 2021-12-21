import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_gene(X):
	if len(X) == 0:
		return X
	scaler_X = StandardScaler()
	# print(X[:10, :10])
	X_log1p = np.log1p(X)
	# X = scaler_X.fit_transform(X)
	X = scaler_X.fit_transform(X_log1p)
	# print(X[:10, :10])
	return X, scaler_X

def scale_time(t):
	if len(t) == 0:
		return t
	scaler_t = StandardScaler(with_mean=False)
	t = scaler_t.fit_transform(t.reshape(-1, 1)).reshape(-1)
	min_t = t.min()
	t -= min_t
	return t, scaler_t
