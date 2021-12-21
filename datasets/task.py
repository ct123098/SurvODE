import math

import numpy as np

from datasets.expression import ExpressionDataBase


class Task:
	def __init__(self, data: ExpressionDataBase):
		self.data = data
		self.train_data = None
		self.valid_data_dict = None
		self.test_data_dict = None

	def get_train(self):
		return self.train_data

	def get_valid(self):
		return self.valid_data_dict

	def get_test(self):
		return self.test_data_dict


class ExtrapolationTask(Task):
	def __init__(self, data: ExpressionDataBase, train_ratio=0.8, train_use_censored_data=False, valid_bins=8,
				 test_bins=2):
		super(ExtrapolationTask, self).__init__(data)
		self.train_ratio = train_ratio
		self.train_use_censored_data = train_use_censored_data
		self.valid_bins = valid_bins
		self.test_bins = test_bins

		self._calc_train()
		self._calc_valid()
		self._calc_test()

	def _calc_train(self):
		data = self.data
		X, t, c = data.X, data.t, data.c
		mask_observed = (c == True)
		# [0, ratio) || [ratio, 1)
		sep_num = math.ceil(self.train_ratio * np.sum(mask_observed))
		sep = np.partition(t[mask_observed], sep_num)[sep_num] if sep_num < sum(mask_observed) else float("inf")
		mask_train = t < sep
		if self.train_use_censored_data is False:
			mask_train &= (c == True)
		X, t, c = X[mask_train], t[mask_train], c[mask_train]
		self.train_data = ExpressionDataBase(X, t, c, data.get_gene_name(), data.get_patient_name()[mask_train])

	def _get_single_case(self, total, index):
		data = self.data
		X, t, c = data.X, data.t, data.c
		mask_observed = (c == True)
		X, t, c = X[mask_observed], t[mask_observed], c[mask_observed]
		rank_indices = np.argsort(np.argsort(t))
		N = X.shape[0]
		st, ed = index / total * N, (index + 1) / total * N
		mask_selected = (st <= rank_indices) & (rank_indices < ed)
		X, t, c = X[mask_selected], t[mask_selected], c[mask_selected]
		return ExpressionDataBase(X, t, c, data.get_gene_name(), data.get_patient_name()[mask_observed][mask_selected])

	def _calc_valid(self):
		valid_data_list = [
			self._get_single_case(self.valid_bins + self.test_bins, i)
			for i in range(0, self.valid_bins)
		]
		self.valid_data_dict = {valid_data[:][1].mean().item(): valid_data for valid_data in valid_data_list}

	def _calc_test(self):
		test_data_list = [
			self._get_single_case(self.valid_bins + self.test_bins, i)
			for i in range(self.valid_bins, self.valid_bins + self.test_bins)
		]
		self.test_data_dict = {test_data[:][1].mean().item(): test_data for test_data in test_data_list}


class InterpolationTask(Task):
	def __init__(self, data: ExpressionDataBase, train_ratio=0.8, train_use_censored_data=False, valid_bins=8,
				 test_bins=2, random_state=0):
		super(InterpolationTask, self).__init__(data)
		self.train_ratio = train_ratio
		self.train_use_censored_data = train_use_censored_data
		self.valid_bins = valid_bins
		self.test_bins = test_bins

		observed_indices = np.where(self.data.c == True)[0]
		rng = np.random.default_rng(random_state)

		perm = rng.permutation(observed_indices.shape[0])
		train_mask_observed = perm < int(self.train_ratio * observed_indices.shape[0])
		valid_mask_observed = perm < int(valid_bins / (valid_bins + test_bins) * observed_indices.shape[0])
		test_mask_observed = ~valid_mask_observed
		self.mask_test = np.zeros(self.data.X.shape[0], dtype=bool)
		self.mask_test[observed_indices[test_mask_observed]] = True
		self.mask_valid = np.zeros(self.data.X.shape[0], dtype=bool)
		self.mask_valid[observed_indices[valid_mask_observed]] = True
		self.mask_train = np.ones(self.data.X.shape[0], dtype=bool)
		self.mask_train[observed_indices[~train_mask_observed]] = False
		# print(self.mask_train.sum())
		# print(self.mask_valid.sum())
		# print(self.mask_test.sum())

		self._calc_train()
		self._calc_valid()
		self._calc_test()

	def _calc_train(self):
		data = self.data
		X, t, c = data.X, data.t, data.c
		mask_train = self.mask_train
		if self.train_use_censored_data is False:
			mask_train &= (c == True)
		X, t, c = X[mask_train], t[mask_train], c[mask_train]
		self.train_data = ExpressionDataBase(X, t, c, data.get_gene_name(), data.get_patient_name()[mask_train])

	def _get_single_case(self, total, index, is_test=False):
		data = self.data
		mask_selected = self.mask_test if is_test else self.mask_valid
		mask_selected &= (data.c == True)
		X, t, c = data.X[mask_selected], data.t[mask_selected], data.c[mask_selected]
		rank_indices = np.argsort(np.argsort(t))
		N = X.shape[0]
		st, ed = index / total * N, (index + 1) / total * N
		mask_range = (st <= rank_indices) & (rank_indices < ed)
		X, t, c = X[mask_range], t[mask_range], c[mask_range]

		return ExpressionDataBase(X, t, c, data.get_gene_name(), data.get_patient_name()[mask_selected][mask_range])

	def _calc_valid(self):
		valid_data_list = [self._get_single_case(self.valid_bins, i, is_test=False) for i in range(self.valid_bins)]
		self.valid_data_dict = {valid_data[:][1].mean().item(): valid_data for valid_data in valid_data_list}

	def _calc_test(self):
		test_data_list = [self._get_single_case(self.test_bins, i, is_test=True) for i in range(self.test_bins)]
		self.test_data_dict = {test_data[:][1].mean().item(): test_data for test_data in test_data_list}
