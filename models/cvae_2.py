from typing import Tuple

import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from torch import nn
from torch.distributions import MultivariateNormal
from torch.nn import functional as F
from models.base import BaseModel
from models.cvae import CVAEBaseModel
from models.mlp import MLP
from models.vae import VAEBaseModel
from utils import metrics
from matplotlib import pyplot as plt

from utils.tools import to_numpy

"""
	calculating KL Divergence by sampling
"""
class ConditionalVAE2(CVAEBaseModel):
	def __init__(self, input_dim: int, latent_dim: int, hidden_dims=None, device="cpu",
				 learning_rate=1e-3, num_epochs=1000, kld_weight_analytical=0.0, kld_weight_sampling=1.0):
		super().__init__()

		if hidden_dims is None:
			hidden_dims = [32, 64, 128, 256, 512]

		self.kld_weight_analytical = kld_weight_analytical
		self.kld_weight_sampling = kld_weight_sampling
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		self.latent_dim = latent_dim

		self.mvn = MultivariateNormal(torch.zeros(self.latent_dim).to(device), covariance_matrix=torch.eye(self.latent_dim).to(device))

		self.encoder = MLP(input_dim + 1, None, hidden_dims)
		# hidden Space
		self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
		self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
		# Decoder
		self.decoder = MLP(latent_dim + 1, input_dim, hidden_dims[::-1])

		self.device = device
		self.to(device)
		# print(self)

	def encode(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		o = torch.cat([x, t.reshape(-1, 1)], dim=1)
		o = self.encoder(o)
		mu = self.fc_mu(o)
		log_var = self.fc_var(o)
		return mu, log_var

	def decode(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		o = torch.cat([z, t.reshape(-1, 1)], dim=1)
		o = self.decoder(o)
		return o

	def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)
		log_prob = self.mvn.log_prob(eps) - 0.5 * log_var.sum(dim=1)
		log_prob = log_prob.reshape(-1, 1)
		return eps * std + mu, log_prob

	def forward(self, x: torch.Tensor, t: torch.Tensor):
		mu, log_var = self.encode(x, t)
		z, r = self.reparameterize(mu, log_var)
		recons = self.decode(z, t)
		return recons, x, mu, log_var, z, r

	def training_step(self, train_batch, epoch_idx):
		self.train()
		X, t, c = train_batch[:]
		recons, inputs, mu, log_var, z, r = self(X.to(self.device), t.to(self.device))
		recons_loss = F.mse_loss(recons, inputs)
		kld_loss_std = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
		rr = self.mvn.log_prob(z)		# N
		kld_loss_sampling = (r - rr).mean()
		# print(kld_loss.item(), kld_loss_std.item())
		loss = recons_loss + self.kld_weight_analytical * kld_loss_std + self.kld_weight_sampling * kld_loss_sampling
		return {'loss': loss, 'loss_rec': recons_loss, 'loss_kld': kld_loss_std, "loss_kld_sampling": kld_loss_sampling}
