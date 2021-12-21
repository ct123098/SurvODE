import sys
from pathlib import Path
import numpy as np
import torch


sys.path.append(str(Path(__file__).parent.parent.absolute()))

from datasets.expression import ExpressionData
from datasets.train_test import ExpressionTrainData, ExpressionTestData
from models.cvae_with_independent_cox import CVAEWithIndependentCox
from models.cvae import ConditionalVAE
from models.vae import VAE


def test_cvae():
	cancer_name = "GBMLGG"
	R = 10
	data = ExpressionData(
		cancer_name, load_data=False, save_data=False,
		feature_selection="reg10", enable_scaling=True,
	)
	train_data = ExpressionTrainData(data, 0.8, use_censored_data=False)
	valid_data_list = [ExpressionTestData(data, R, i) for i in range(R)]
	valid_data_dict = {valid_data[:][1].mean().item(): valid_data for valid_data in valid_data_list}
	N, P = data[:][0].shape

	# model_vae = VAE(input_dim=P, latent_dim=4, hidden_dims=[32, 32], device="cpu", num_epochs=1000, kld_weight=1.0)
	# model_vae.fit(train_data, valid_data_dict=valid_data_dict)

	model_cvae = ConditionalVAE(input_dim=P, label_dim=1, latent_dim=4, hidden_dims=[32, 32], device="cpu", num_epochs=1000, kld_weight=1.0)
	model_cvae.fit(train_data, valid_data_dict=valid_data_dict)

	# model_ind_cox_cvae = IndependentCoxConditionalVAE(input_dim=P, label_dim=1, latent_dim=4, hidden_dims=[32, 32], device="cpu", num_epochs=1000, kld_weight=1.0)
	# model_ind_cox_cvae.fit(train_data, valid_data_dict=valid_data_dict)


if __name__ == '__main__':
	test_cvae()
