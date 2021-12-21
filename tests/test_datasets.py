import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from datasets.expression import ExpressionData


def test_expression_dataset():
	cancer_name = "CESC"
	# data = ExpressionData(cancer_name, save_data=False, feature_selection="reg", enable_scaling=True)
	# exit(0)

	num_sample_of_this_dataset = 193
	project_root = Path(__file__).parent.parent
	save_path = project_root / 'data' / 'preprocessed' / f'{cancer_name}.npz'
	if save_path.exists():
		save_path.unlink()
	for load_data, save_data in [(True, False), (True, True), (True, False)]:
		data = ExpressionData(cancer_name, load_data=load_data, save_data=save_data)
		assert len(data) == num_sample_of_this_dataset
		for i in range(len(data)):
			assert isinstance(data[i][0], torch.Tensor) and data[i][0].dtype == torch.float32
			assert isinstance(data[i][1], torch.Tensor) and data[i][1].dtype == torch.float32
			assert isinstance(data[i][2], torch.Tensor) and data[i][2].dtype == torch.bool
	# print(len(data))
	# print(data[0])


def test_reproducibility():
	cancer_name = "CESC"
	data1 = ExpressionData(
		cancer_name, load_data=True, save_data=True,
		feature_selection="reg10", enable_scaling=True,
	)
	data2 = ExpressionData(
		cancer_name, load_data=True, save_data=True,
		feature_selection="reg10", enable_scaling=True,
	)
	assert (data1.get_gene_name() == data2.get_gene_name()).all()
	assert (data1.get_patient_name() == data2.get_patient_name()).all()


if __name__ == '__main__':
	test_expression_dataset()
	test_reproducibility()
