from utils import metrics
import numpy as np
from sklearn.linear_model import ElasticNet

def select_feature_rand(X, t, c, max_number=100, random_state=0, verbose=False):
	N, P = X.shape
	rng = np.random.default_rng(random_state)
	key_gene_indices = rng.choice(P, max_number, replace=False)
	return key_gene_indices


def select_features_reg(X, t, c, max_number=100, random_state=0, verbose=False):
	params = {"alpha": 0.1, "l1_ratio": 0.95}
	R = 10

	N, P = X.shape
	accumulated_coef = np.zeros(P)
	rng = np.random.default_rng(random_state)
	round_index = rng.permutation([i % R for i in range(N)])
	# print(round_index.shape)
	for i in range(R):
		train_mask = round_index != i
		valid_mask = round_index == i
		# print(train_mask.shape)
		# print(X[train_mask].shape, t[train_mask].shape)
		reg_model = ElasticNet(**params).fit(X[train_mask], t[train_mask])
		for indices, data in zip(reg_model.sparse_coef_.indices, reg_model.sparse_coef_.data):
			accumulated_coef[indices] += data
		score_train = reg_model.score(X[train_mask], t[train_mask])
		score_test = reg_model.score(X[valid_mask], t[valid_mask])
		if verbose:
			print(score_train, score_test)
		# print(reg_model.sparse_coef_)

	key_gene_indices = np.argsort(-accumulated_coef)[:max_number]
	key_gene_indices = key_gene_indices[accumulated_coef[key_gene_indices] > 0.0]

	return key_gene_indices

def select_features_wdist(X, t, c, max_number=100, verbose=False):
	N, P = X.shape
	order = np.argsort(t)
	left_indices, right_indices = order[:N // 2], order[N // 2:]
	# left_indices, right_indices = (t <= 1.0), (t > 1.0)
	X_left, X_right = X[left_indices], X[right_indices]
	# print(t[left_indices], t[right_indices])
	weight = np.zeros(P)
	for i in range(P):
		bound = np.abs(X[:, i]).max()
		if bound > 4.0:
			continue
		weight[i] = metrics.wasserstein_distance(X_left[:, i], X_right[:, i])
	key_gene_indices = np.argsort(-weight)[:max_number]
	# print(weight[key_gene_indices])

	return key_gene_indices


def select_features_nwdist(X, t, c, max_number=100, verbose=False):
	X_ob, t_ob = X[c == True], t[c == True]
	N, P = X_ob.shape
	order = np.argsort(t_ob)
	left_indices, right_indices = order[:N // 2], order[N // 2:]
	# left_indices, right_indices = (t <= 1.0), (t > 1.0)
	X_left, X_right = X_ob[left_indices], X_ob[right_indices]
	# print(t[left_indices], t[right_indices])
	weight = np.zeros(P)
	for i in range(P):
		bound = np.abs(X_ob[:, i]).max()
		if bound > 4.0:
			continue
		weight[i] = metrics.wasserstein_distance(X_left[:, i], X_right[:, i])
	key_gene_indices = np.argsort(-weight)[:max_number]
	# print(weight[key_gene_indices])

	# print(np.sum(weight > 0))
	# for i in range(0, max_number, max_number // 10):
	# 	print(i, key_gene_indices[i], weight[key_gene_indices[i]])

	return key_gene_indices


# def select_features_cox(dataset):
# 	X_train, X_test, y_train, y_test = dataset.train_test_split(survival=True, seed=0)
#
# 	if X_train is None or X_train.shape[0] < 100:
# 		return None
#
# 	transformation = GeneDataGeneralTransformation(enable_y=True).fit(X_train, y_train)
# 	X_train_trans, y_train_trans = transformation.transform(X_train, y_train)
# 	X_test_trans, y_test_trans = transformation.transform(X_test, y_test)
#
# 	assert type(y_train_trans) is np.ndarray and len(y_train_trans.dtype) == 2
#
# 	n_components = min(100, X_train_trans.shape[0])
# 	pca = PCA(n_components=n_components, whiten=True).fit(X_train_trans)
# 	X_train_reduced = pca.transform(X_train_trans)
# 	X_test_reduced = pca.transform(X_test_trans)
#
# 	cox_model = CoxPHSurvivalAnalysis(alpha=1e5).fit(X_train_reduced, y_train_trans)
#
# 	score_train = cox_model.score(X_train_reduced, y_train_trans)
# 	score_test = cox_model.score(X_test_reduced, y_test_trans)
#
# 	# print(score_train, score_test)
# 	# print(type(pca.components_))
# 	# print(pca.components_.shape)
#
# 	# Unit Test
# 	full_coef = pca.components_.T @ np.diag(1 / np.sqrt(pca.explained_variance_)) @ cox_model.coef_
#
# 	# print(full_coef.shape)
# 	# print(full_coef)
# 	# print((X_train_trans @ full_coef).shape)
#
# 	# print((X_train_reduced @ cox_model.coef_)[:10])
# 	# print((X_train_trans @ full_coef)[:10])
#
# 	cox_model_tmp = CoxPHSurvivalAnalysis(alpha=0.0).fit((X_train_trans @ full_coef).reshape(-1, 1), y_train_trans)
#
# 	# print(cox_model_tmp.score((X_train_trans @ full_coef).reshape(-1, 1), y_train_trans), score_train)
# 	# print(cox_model_tmp.score((X_test_trans @ full_coef).reshape(-1, 1), y_test_trans), score_test)
#
# 	assert abs(cox_model_tmp.score((X_train_trans @ full_coef).reshape(-1, 1), y_train_trans) - score_train) <= 1e-3
# 	assert abs(cox_model_tmp.score((X_test_trans @ full_coef).reshape(-1, 1), y_test_trans) - score_test) <= 1e-3
#
# 	gene_order = np.argsort(-full_coef)
# 	# print([(gene_id, full_coef[gene_id]) for gene_id in gene_order][:10])
#
# 	sparse_coef = np.zeros_like(full_coef)
# 	key_gene_indices = gene_order[:100]
# 	for gene_id in key_gene_indices:
# 		sparse_coef[gene_id] = full_coef[gene_id]
# 	cox_model_tmp = CoxPHSurvivalAnalysis(alpha=0.0).fit((X_train_trans @ sparse_coef).reshape(-1, 1), y_train_trans)
# 	# print(cox_model_tmp.score((X_train_trans @ sparse_coef).reshape(-1, 1), y_train_trans))
# 	# print(cox_model_tmp.score((X_test_trans @ sparse_coef).reshape(-1, 1), y_test_trans))
#
# 	return key_gene_indices


# def evaluate_dataset(dataset):
# 	result = {}
# 	X_train, X_test, y_train, y_test = dataset.train_test_split(survival=False)
#
# 	reg = LinearRegression(normalize=True).fit(X_train, y_train)
# 	result["linear_regression"] = (reg.score(X_train, y_train), reg.score(X_test, y_test))
#
# 	reg_en = ElasticNet(alpha=0.3, l1_ratio=1.0, normalize=True).fit(X_train, y_train)
# 	result["linear_regression_elastic_net"] = (reg_en.score(X_train, y_train), reg_en.score(X_test, y_test))
#
# 	X_train, X_test, y_train, y_test = dataset.train_test_split(survival=True)
#
# 	scaler = StandardScaler().fit(X_train)
# 	X_train_normalized = scaler.transform(X_train)
# 	X_test_normalized = scaler.transform(X_test)
# 	n_components = min(200, X_train.shape[0])
# 	pca = PCA(n_components=n_components, whiten=True).fit(X_train_normalized)
# 	X_train_reduced = pca.transform(X_train_normalized)
# 	X_test_reduced = pca.transform(X_test_normalized)
#
# 	# print(X_train.sum())
# 	# print(X_train)
# 	# print(X_train_normalized)
# 	# print(X_train_reduced)
#
# 	estimator = CoxPHSurvivalAnalysis(alpha=0.01).fit(X_train_reduced, y_train)
# 	result["cox_PH_survival_analysis"] = (
# 	estimator.score(X_train_reduced, y_train), estimator.score(X_test_reduced, y_test))
#
# 	return result
