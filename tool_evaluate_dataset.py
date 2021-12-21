import argparse
import os
from typing import Union

from models.odevae_distillation import ODEVAEDistillationIndCox
from models.odevae_with_independent_cox import ODEVAEWithIndependentCox

import numpy as np

from datasets.expression import ExpressionData
from datasets.task import ExtrapolationTask, InterpolationTask
from models.cvae import ConditionalVAE
from models.cvae_with_independent_cox import CVAEWithIndependentCox
from models.odevae import ODEVAE
from models.odevae_with_independent_cox_Iter import ODEVAEWithIndependentCoxIter
from models.vae import VAE
from utils.evaluation import evaluate
from utils.tools import seed_everything


def main(args):
	print(">>> begin")

	device = "cuda:0"
	cancer_name = args.dataset
	feature_method = args.feature

	data_fn = lambda: ExpressionData(
		cancer_name, load_data=True, save_data=True,
		feature_selection=feature_method, enable_scaling=True,
	)

	task_cls: type = {"extrapolation": ExtrapolationTask, "interpolation": InterpolationTask}.get(args.task)
	task_observed_fn = lambda data: task_cls(
		data, train_ratio=0.8, train_use_censored_data=False, valid_bins=8, test_bins=2
	)
	task_censored_fn = lambda data: task_cls(
		data, train_ratio=0.8, train_use_censored_data=True, valid_bins=8, test_bins=2
	)
	# task_range_fn = lambda data: SameRangeTask(
	# 	data, train_ratio=0.8, train_use_censored_data=False, valid_bins=8, test_bins=2
	# )
	task_train_all_fn = lambda data: task_cls(
		data, train_ratio=1.0, train_use_censored_data=False, valid_bins=8, test_bins=2
	)

	learning_rate = 1e-3
	num_epoch = 5000
	beta = 0.04
	vae_fn = lambda train_data: VAE(
		input_dim=train_data[:][0].shape[1], latent_dim=4, hidden_dims=[32, 32], device=device,
		learning_rate=learning_rate, num_epochs=num_epoch, kld_weight=beta,
	)
	cvae_fn = lambda train_data: ConditionalVAE(
		input_dim=train_data[:][0].shape[1], latent_dim=4, hidden_dims=[32, 32], device=device,
		learning_rate=learning_rate, num_epochs=num_epoch, kld_weight=beta,
	)
	cvae_ind_cox_fn = lambda train_data: CVAEWithIndependentCox(
		input_dim=train_data[:][0].shape[1], latent_dim=4, hidden_dims=[32, 32], device=device,
		learning_rate=learning_rate, num_epochs=num_epoch, kld_weight=beta,
	)
	odevae_fn = lambda train_data: ODEVAE(
		input_dim=train_data[:][0].shape[1], latent_dim=4, encoder_hidden_dims=[32, 32], ode_hidden_dims=[16, 16],
		device=device,
		learning_rate=learning_rate, num_epochs=num_epoch, kld_0_weight=beta / 2, kld_t_weight=beta / 2,
	)
	odevae_without_kld_t_fn = lambda train_data: ODEVAE(
		input_dim=train_data[:][0].shape[1], latent_dim=4, encoder_hidden_dims=[32, 32], ode_hidden_dims=[16, 16],
		device=device,
		learning_rate=learning_rate, num_epochs=num_epoch, kld_0_weight=beta, kld_t_weight=0.0,
	)
	odevae_ind_cox_fn = lambda train_data: ODEVAEWithIndependentCox(
		input_dim=train_data[:][0].shape[1], latent_dim=4, encoder_hidden_dims=[32, 32], ode_hidden_dims=[16, 16],
		device=device,
		learning_rate=learning_rate, num_epochs=num_epoch, kld_0_weight=beta / 2, kld_t_weight=beta / 2,
	)
	odevae_ind_cox_quarter_fn = lambda train_data: ODEVAEWithIndependentCox(
		input_dim=train_data[:][0].shape[1], latent_dim=4, encoder_hidden_dims=[32, 32], ode_hidden_dims=[16, 16],
		device=device,
		learning_rate=learning_rate, num_epochs=num_epoch, kld_0_weight=beta / 4 * 3, kld_t_weight=beta / 4,
	)
	odevae_ind_cox_l2_fn = lambda train_data: ODEVAEWithIndependentCox(
		input_dim=train_data[:][0].shape[1], latent_dim=4, encoder_hidden_dims=[32, 32], ode_hidden_dims=[16, 16],
		device=device,
		learning_rate=learning_rate, num_epochs=num_epoch, kld_0_weight=beta / 2, kld_t_weight=beta / 2, kld_t_type="l2",
	)
	odevae_ind_cox_l2_quarter_fn = lambda train_data: ODEVAEWithIndependentCox(
		input_dim=train_data[:][0].shape[1], latent_dim=4, encoder_hidden_dims=[32, 32], ode_hidden_dims=[16, 16],
		device=device,
		learning_rate=learning_rate, num_epochs=num_epoch, kld_0_weight=beta / 4 * 3, kld_t_weight=beta / 4, kld_t_type="l2",
	)
	odevae_distill_indcox_fn = lambda train_data: ODEVAEDistillationIndCox(
		input_dim=train_data[:][0].shape[1], latent_dim=4, ode_dim=4,
		encoder_hidden_dims=[32, 32], ode_hidden_dims=[16, 16], ode_init_hidden_dims=[16, 16],
		cvae_hidden_dims=[32, 32],
		device=device,
		learning_rate=learning_rate, num_epochs=num_epoch, num_epochs_distill=num_epoch,
		weight_recons_cvae=1.0, weight_recons_ode=1.0, weight_kld=beta, weight_zt=1.0,
	)
	odevae_ind_cox_iter_fn = lambda train_data: ODEVAEWithIndependentCoxIter(
		input_dim=train_data[:][0].shape[1], latent_dim=4, encoder_hidden_dims=[32, 32], ode_hidden_dims=[16, 16],
		device=device,
		learning_rate=learning_rate, num_epochs=num_epoch, kld_0_weight=beta / 2, kld_t_weight=beta / 2,
		cox_num_sample=8, cox_freq=num_epoch // 10,
	)

	exp_prefix = f"eval/{args.task}/{cancer_name}{feature_method}"
	evaluate(data_fn, task_train_all_fn, cvae_fn, f"{exp_prefix}/opt", verbose=True)
	evaluate(data_fn, task_observed_fn, vae_fn, f"{exp_prefix}/vae", verbose=True)
	evaluate(data_fn, task_observed_fn, cvae_fn, f"{exp_prefix}/cvae", verbose=True)
	evaluate(data_fn, task_observed_fn, cvae_ind_cox_fn, f"{exp_prefix}/cvaeIndCox", verbose=True)
	evaluate(data_fn, task_censored_fn, cvae_ind_cox_fn, f"{exp_prefix}/cvaeIndCoxCensored", verbose=True)
	evaluate(data_fn, task_observed_fn, odevae_fn, f"{exp_prefix}/odevae", verbose=True)
	evaluate(data_fn, task_observed_fn, odevae_ind_cox_fn, f"{exp_prefix}/odevaeIndCox", verbose=True)
	evaluate(data_fn, task_censored_fn, odevae_ind_cox_fn, f"{exp_prefix}/odevaeIndCoxCensored", verbose=True)
	evaluate(data_fn, task_observed_fn, odevae_without_kld_t_fn, f"{exp_prefix}/odevaeWithoutKLDt", verbose=True)
	evaluate(data_fn, task_censored_fn, odevae_ind_cox_quarter_fn, f"{exp_prefix}/odevaeIndCoxCensoredQuarter", verbose=True)
	evaluate(data_fn, task_censored_fn, odevae_ind_cox_l2_fn, f"{exp_prefix}/odevaeIndCoxCensoredL2", verbose=True)
	evaluate(data_fn, task_censored_fn, odevae_ind_cox_l2_quarter_fn, f"{exp_prefix}/odevaeIndCoxCensoredL2Quarter", verbose=True)
	evaluate(data_fn, task_censored_fn, odevae_distill_indcox_fn, f"{exp_prefix}/odevaeDistillIndCoxCensored", verbose=True)
	evaluate(data_fn, task_censored_fn, odevae_ind_cox_iter_fn, f"{exp_prefix}/odevaeIndCoxCensoredIter", verbose=True)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Genegen')
	parser.add_argument("--gpu", type=int, default=None)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--dataset", type=str, default="GBMLGG")
	parser.add_argument("--feature", type=str, default="wdist10")
	parser.add_argument("--task", type=str, default="extrapolation")
	args = parser.parse_args()
	if isinstance(args.gpu, int) and args.gpu >= 0:
		print(f"use gpu {args.gpu}")
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	if isinstance(args.seed, int):
		seed_everything(args.seed)
	main(args)
