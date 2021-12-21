from pathlib import Path

import numpy as np
from datetime import date

def evaluate(data_fn, task_fn, model_fn, experiment_name=None, verbose=False, save_model=False):
	if verbose:
		print(experiment_name)
	data = data_fn()
	# print(data.get_gene_name())
	task = task_fn(data)
	train_data, valid_data_dict, test_data_dict = task.get_train(), task.get_valid(), task.get_test()
	# print(train_data, valid_data_dict, test_data_dict)
	# print(next(iter(test_data_dict.values()))[:][1])
	model = model_fn(train_data)
	model.fit(train_data, valid_data_dict, test_data_dict,
			  experiment_name=f'{date.today().strftime("%m%d")}/{experiment_name}',
			  verbose=False, enable_tensorboard=True)
	if save_model:
		path = f'log/{date.today().strftime("%m%d")}/{experiment_name}'
		Path(path).parent.mkdir(parents=True, exist_ok=True)
		model.save(path)
	info_eval = model.score(test_data_dict)
	ts = list(info_eval.keys())
	score = {key: np.mean([info_eval[t][key] for t in ts]) for key in info_eval[ts[0]].keys()}
	# print(info_eval)
	return score

