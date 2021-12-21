from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from models.cvae import CVAEBaseModel
from models.mlp import MLP
from models.vae import VAEBaseModel


class CVAEWithJointCox(CVAEBaseModel):
	def __init__(self, input_dim: int, latent_dim: int, hidden_dims=None, survival_hidden_dims=None,
				 device="cpu", kld_weight=1.0, cox_weight=1.0, reg_weight=1.0, num_epochs=1000, learning_rate=1e-3):
		super().__init__()

		if hidden_dims is None:
			hidden_dims = [32, 64, 128, 256, 512]
		if survival_hidden_dims is None:
			survival_hidden_dims = [32, 32]

		self.learning_rate = learning_rate
		self.num_epochs = num_epochs
		self.kld_weight = kld_weight
		self.cox_weight = cox_weight
		self.reg_weight = reg_weight

		self.latent_dim = latent_dim

		# Encoder
		self.encoder = MLP(input_dim + 1 + 1, None, hidden_dims)
		# hidden Space
		self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
		self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
		# Hazard Ratio
		self.fc_phi = nn.Linear(hidden_dims[-1], 1)
		# Decoder
		self.decoder = MLP(latent_dim + 1, input_dim, hidden_dims[::-1])
		# Build Survival Function
		self.survival_function = MLP(1, 1, survival_hidden_dims, coefficient=-1.0)

		self.device = device
		self.to(device)
		# print(self)

	def encode(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		o = torch.cat([x, t.reshape(-1, 1), c.reshape(-1, 1)], dim=1)
		assert isinstance(x, torch.Tensor) and x.ndim == 2
		o = self.encoder(o)
		mu = self.fc_mu(o)
		log_var = self.fc_var(o)
		phi = self.fc_phi(o)
		return mu, log_var, phi

	def decode(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		o = torch.cat([z, t.reshape(-1, 1)], dim=1)
		o = self.decoder(o)
		return o

	def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor, phi: torch.Tensor) -> Tuple[
		torch.Tensor, torch.Tensor]:
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)
		z = eps * std + mu
		exponent = 1 / torch.exp(phi)
		u = torch.zeros_like(phi).uniform_(0, 1)
		u = torch.pow(u, exponent)
		assert ((u >= 0) & (u <= 1)).all(), f'u: {u.reshape(-1)}'
		t = self.survival_function(u)
		return z, t

	def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> Tuple[
		torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		mu, log_var, phi = self.encode(x, t, c)
		z, t = self.reparameterize(mu, log_var, phi)
		recons = self.decode(z, t)
		return recons, x, mu, log_var, phi, z, t

	def training_step(self, train_batch, epoch_idx):
		self.train()
		X, y_value, y_observed = train_batch[:]
		y_value = y_value.to(self.device)
		y_observed = y_observed.to(self.device)
		recons, inputs, mu, log_var, phi, z, t = self(X.to(self.device), y_value, y_observed)
		phi = phi.reshape(-1)
		t = t.reshape(-1)
		# Reconstruction Loss
		recons_loss = F.mse_loss(recons, inputs)
		# KL Divergence Loss
		kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
		# Cox Model Loss
		y_censored = ~y_observed
		y_order_matrix = y_value.reshape(-1, 1) <= y_value.reshape(1, -1)
		# print(y_order_matrix)
		num_observed = torch.sum(y_observed)
		exp_phi = torch.exp(phi)
		# print(phi)
		# print(exp_phi)
		y_order_sum = torch.sum((exp_phi.reshape(1, -1) * y_order_matrix), dim=1)
		# print(y_order_sum)
		cox_loss = -(torch.sum((phi - torch.log(y_order_sum)) * y_observed) / num_observed)
		# exit(0)
		# T-distribution Loss
		tmp1 = torch.abs(y_value - t)
		tmp2 = torch.maximum(y_value - t, torch.zeros_like(y_value))
		reg_loss = torch.mean(tmp1 * y_observed + tmp2 * y_censored) + phi.norm(2)

		loss = recons_loss + self.kld_weight * kld_loss + self.cox_weight * cox_loss + self.reg_weight * reg_loss

		# print(cox_loss.item(), reg_loss.item(), phi[:4].tolist(), t[:4].tolist())

		if epoch_idx % 500 == 0:
			print(phi.norm(2))
			print(phi[:10].tolist())
			print(t[:10].tolist())

		return {'loss': loss, 'loss_rec': recons_loss, 'loss_kld': kld_loss, 'loss_cox': cox_loss, 'loss_reg': reg_loss}