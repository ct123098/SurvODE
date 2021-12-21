from typing import Tuple

import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from torch import nn
from torch.nn import functional as F
from models.base import BaseModel
from models.cvae import CVAEBaseModel
from models.mlp import MLP
from models.vae import VAEBaseModel
from utils import metrics
from matplotlib import pyplot as plt

from utils.plotting import plot_hidden_dynamics
from utils.tools import to_numpy, random_survival_function

"""
	use a dynamics network to merge z_0 and t
"""

class ConditionalVAE3(CVAEBaseModel):
	def __init__(self, input_dim: int, latent_dim: int, latent_dim_dynamics: int,
				 hidden_dims=None, hidden_dims_dynamics=None,
				 device="cpu",
				 learning_rate=1e-3, num_epochs=1000, kld_weight=1.0, sample_method="pz"):
		super().__init__()

		if hidden_dims is None:
			hidden_dims = [32, 64, 128, 256, 512]
		if hidden_dims_dynamics is None:
			hidden_dims_dynamics = [32, 32]

		self.learning_rate = learning_rate
		self.num_epochs = num_epochs
		self.kld_weight = kld_weight
		self.latent_dim = latent_dim
		self.sample_method = sample_method

		self.encoder = MLP(input_dim + 1, None, hidden_dims)
		# hidden Space
		self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
		self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

		# Decoder
		self.dynamics_net = MLP(latent_dim + 1, latent_dim_dynamics, hidden_dims_dynamics)
		self.decoder = MLP(latent_dim_dynamics, input_dim, hidden_dims[::-1])

		self.device = device
		self.to(device)

	# print(self)

	def encode(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		o = torch.cat([x, t.reshape(-1, 1)], dim=1)
		o = self.encoder(o)
		mu = self.fc_mu(o)
		log_var = self.fc_var(o)
		return mu, log_var

	def decode(self, z: torch.Tensor, t: torch.Tensor, return_z=False) -> Tuple[torch.Tensor, torch.Tensor]:
		z_t = self.dynamics_net(torch.cat([z, t.reshape(-1, 1)], dim=1))
		o = self.decoder(z_t)
		if return_z:
			return o, z_t
		else:
			return o

	def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)
		return eps * std + mu

	def forward(self, x: torch.Tensor, t: torch.Tensor):
		mu, log_var = self.encode(x, t)
		z_0 = self.reparameterize(mu, log_var)
		recons, z_t = self.decode(z_0, t, return_z=True)
		return recons, x, mu, log_var, z_0, z_t

	def training_step(self, train_batch, epoch_idx):
		self.train()
		X, t, c = train_batch[:]
		recons, inputs, mu, log_var, _, _ = self(X.to(self.device), t.to(self.device))
		recons_loss = F.mse_loss(recons, inputs)
		kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
		loss = recons_loss + self.kld_weight * kld_loss
		return {'loss': loss, 'loss_rec': recons_loss, 'loss_kld': kld_loss}

	def callback_eval_plot(self, epoch, train_data, valid_data_dict=None, test_data_dict=None):
		if self.tb_logger:
			fig = plot_hidden_dynamics(self, valid_data_dict=valid_data_dict, test_data_dict=test_data_dict,
									   display_mode="tensorboard")
			self.tb_logger.add_figure('Hidden Dynamics', fig, epoch)


class CVAEWithIndependentCox3(ConditionalVAE3):
	def __init__(self, input_dim: int, latent_dim: int, latent_dim_dynamics: int,
				 hidden_dims=None, hidden_dims_dynamics=None,
				 device="cpu",
				 learning_rate=1e-3, num_epochs=1000, kld_weight=1.0, sample_method="pz"):
		super().__init__(input_dim, latent_dim, latent_dim_dynamics,
						 hidden_dims=hidden_dims, hidden_dims_dynamics=hidden_dims_dynamics,
						 device=device, kld_weight=kld_weight,
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
		recons, inputs, mu, log_var, _, _ = self(X.to(self.device), t_sampled.to(self.device))
		recons_loss = F.mse_loss(recons, inputs)
		kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
		loss = recons_loss + self.kld_weight * kld_loss
		return {'loss': loss, 'loss_rec': recons_loss, 'loss_kld': kld_loss}
