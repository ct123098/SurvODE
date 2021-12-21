from typing import Tuple

import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from torch import nn
from torch.nn import functional as F
from models.base import BaseModel
from models.mlp import MLP
from models.vae import VAEBaseModel
from utils import metrics
from matplotlib import pyplot as plt

from utils.plotting import plot_hidden_dynamics
from utils.tools import to_numpy

from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.util import Surv
from utils.tools import random_survival_function, to_numpy


class CVAEBaseModelEncoderWithoutT(VAEBaseModel):
	def __init__(self):
		super().__init__()

		self.sample_method = None

	def sample(self, num_samples: int, t: float, return_z=False):
		self.eval()
		if self.train_data is None or self.sample_method == "pz":
			if self.train_data is None:
				print("[warning] the model is not trained")
			z_repeated = torch.randn(num_samples, self.latent_dim).to(self.device)
		# elif self.sample_method == "qz":
		# 	X_train, t_train, c_train = self.train_data[:]
		# 	X_train, t_train, c_train = X_train.to(self.device), t_train.to(self.device).reshape(-1, 1), c_train.to(
		# 		self.device).reshape(-1, 1)
		# 	sampling_indices = torch.randint(0, X_train.shape[0], (num_samples,))
		# 	mu_sampled, log_var_sampled = self.encode(X_train[sampling_indices], t_train[sampling_indices])
		# 	assert isinstance(mu_sampled, torch.Tensor) and mu_sampled.ndim == 2
		# 	assert isinstance(log_var_sampled, torch.Tensor) and log_var_sampled.ndim == 2
		# 	z_repeated = self.reparameterize(mu_sampled, log_var_sampled)
		# # print(num_samples, t, mu_sampled.shape, mu_sampled[:2, :2], log_var_sampled[:2, :2].exp(), z_repeated[:2, :2])
		else:
			raise NotImplementedError
		t_repeated = torch.full((num_samples, 1), t).to(self.device)
		samples = self.decode(z_repeated, t_repeated)
		assert len(samples.shape) == 2 and samples.shape[0] == num_samples
		if not return_z:
			return samples
		else:
			return samples, z_repeated


class ConditionalVAEEncoderWithoutT(CVAEBaseModelEncoderWithoutT):
	def __init__(self, input_dim: int, latent_dim: int, hidden_dims=None, device="cpu", learning_rate=1e-3,
				 num_epochs=1000, kld_weight=1.0, sample_method="pz"):
		super().__init__()

		if hidden_dims is None:
			hidden_dims = [32, 64, 128, 256, 512]

		self.learning_rate = learning_rate
		self.num_epochs = num_epochs
		self.kld_weight = kld_weight
		self.latent_dim = latent_dim
		self.sample_method = sample_method

		self.encoder = MLP(input_dim, None, hidden_dims)
		# hidden Space
		self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
		self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
		# Decoder
		self.decoder = MLP(latent_dim + 1, input_dim, hidden_dims[::-1])

		self.device = device
		self.to(device)

	# print(self)

	def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		o = self.encoder(x)
		mu = self.fc_mu(o)
		log_var = self.fc_var(o)
		return mu, log_var

	def decode(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		o = torch.cat([z, t.reshape(-1, 1)], dim=1)
		o = self.decoder(o)
		return o

	def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)
		return eps * std + mu

	def forward(self, x: torch.Tensor, t: torch.Tensor):
		mu, log_var = self.encode(x)
		z = self.reparameterize(mu, log_var)
		recons = self.decode(z, t)
		return recons, x, mu, log_var, z

	def training_step(self, train_batch, epoch_idx):
		self.train()
		X, t, c = train_batch[:]
		recons, inputs, mu, log_var, _ = self(X.to(self.device), t.to(self.device))
		recons_loss = F.mse_loss(recons, inputs)
		kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
		loss = recons_loss + self.kld_weight * kld_loss
		return {'loss': loss, 'loss_rec': recons_loss, 'loss_kld': kld_loss}

	# def callback_eval_plot(self, epoch, train_data, valid_data_dict=None, test_data_dict=None):
	# 	if self.tb_logger:
	# 		fig = plot_hidden_dynamics(self, valid_data_dict=valid_data_dict, test_data_dict=test_data_dict,
	# 								   display_mode="tensorboard")
	# 		self.tb_logger.add_figure('Hidden Dynamics', fig, epoch)


class CVAEWithIndependentCoxEncoderWithoutT(ConditionalVAEEncoderWithoutT):
	def __init__(self, input_dim: int, latent_dim: int, hidden_dims=None, device="cpu", kld_weight=1.0, num_epochs=1000,
				 learning_rate=1e-3, sample_method="pz"):
		super().__init__(input_dim, latent_dim, hidden_dims=hidden_dims, device=device, kld_weight=kld_weight,
						 num_epochs=num_epochs, learning_rate=learning_rate, sample_method=sample_method)

		self.cox_regression = None

	def training_step(self, train_batch, epoch_idx):
		self.train()
		X, t, c = train_batch[:]
		if self.cox_regression is None:
			assert epoch_idx == 0
			self.cox_regression = CoxnetSurvivalAnalysis(fit_baseline_model=True).fit(X, Surv.from_arrays(c, t))
		# self.cox_regression = CoxPHSurvivalAnalysis(alpha=0.1).fit(X, Surv.from_arrays(c, t))
		surv_funcs = self.cox_regression.predict_survival_function(X)
		t_sampled = torch.tensor([random_survival_function(fn) for fn in surv_funcs]).float().to(self.device)
		recons, inputs, mu, log_var, _ = self(X.to(self.device), t_sampled)
		recons_loss = F.mse_loss(recons, inputs)
		kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
		loss = recons_loss + self.kld_weight * kld_loss
		return {'loss': loss, 'loss_rec': recons_loss, 'loss_kld': kld_loss}

