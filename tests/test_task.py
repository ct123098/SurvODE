import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from datasets.expression import ExpressionData
from datasets.task import ExtrapolationTask, InterpolationTask


def check_isin(data_small, data_big, neg_flag=False):
	X, t, _ = data_big[:]
	for xi, ti, _ in zip(*data_small[:]):
		res = torch.any(torch.all(X == xi, dim=1)) and torch.any(t == ti)
		if res.item() is (False ^ neg_flag):
			return False
	return True


def test_extrapolation_task():
	cancer_name = "CESC"
	data = ExpressionData(cancer_name, load_data=True, save_data=True, feature_selection="reg10", enable_scaling=True)
	task1 = ExtrapolationTask(data, train_ratio=0.8, train_use_censored_data=False, valid_bins=8, test_bins=2)
	train_data, valid_data_dict, test_data_dict = task1.get_train(), task1.get_valid(), task1.get_test()
	for valid_data in valid_data_dict.values():
		assert check_isin(valid_data, train_data)
	for test_data in test_data_dict.values():
		assert not check_isin(test_data, train_data)
		assert check_isin(test_data, train_data, neg_flag=True)

	task2 = ExtrapolationTask(data, train_ratio=0.9, train_use_censored_data=False, valid_bins=8, test_bins=2)
	train_data, valid_data_dict, test_data_dict = task2.get_train(), task2.get_valid(), task2.get_test()
	for valid_data in valid_data_dict.values():
		assert check_isin(valid_data, train_data)
	assert check_isin(list(test_data_dict.values())[0], train_data)
	assert check_isin(list(test_data_dict.values())[1], train_data, neg_flag=True)

def test_same_range_task():
	cancer_name = "CESC"
	data = ExpressionData(cancer_name, load_data=True, save_data=True, feature_selection="reg10", enable_scaling=True)
	task = InterpolationTask(data, train_ratio=0.8, train_use_censored_data=False, valid_bins=8, test_bins=2)
	train_data, valid_data_dict, test_data_dict = task.get_train(), task.get_valid(), task.get_test()
	for valid_data in valid_data_dict.values():
		assert check_isin(valid_data, train_data)
	for test_data in test_data_dict.values():
		assert not check_isin(test_data, train_data)
		assert check_isin(test_data, train_data, neg_flag=True)


if __name__ == '__main__':
	test_extrapolation_task()
	test_same_range_task()
