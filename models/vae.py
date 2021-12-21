import io
import time
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.base import BaseModel
from models.mlp import MLP
from utils import metrics
from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import PIL.Image
from torchvision.transforms import ToTensor

from utils.plotting import plot_generated_samples, plot_gene_vs_time, plot_hidden_space, plot_hidden_dynamics


class VAEBaseModel(BaseModel):
	def __init__(self):
		super().__init__()

		self.learning_rate = None
		self.num_epochs = None
		self.encoder = None
		self.fc_mu = None
		self.fc_var = None
		self.decoder = None
		self.tb_logger = None
		self.train_data = None
		self.valid_data_dict = None
		self.test_data_dict = None


	def encode(self, *args):
		raise NotImplementedError

	def decode(self, *args):
		raise NotImplementedError

	def reparameterize(self, *args):
		raise NotImplementedError

	def forward(self, *args):
		raise NotImplementedError

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		return optimizer

	def choose_optimizer(self, optimizers, epoch):
		return optimizers

	def fit(self, train_data, valid_data_dict=None, test_data_dict=None, experiment_name=None, enable_tensorboard=True, verbose=False):
		self.train_data = train_data
		self.valid_data_dict = valid_data_dict
		self.test_data_dict = test_data_dict

		info_history = []
		if enable_tensorboard:
			tb_log_dir = f'runs/{experiment_name}' if experiment_name is not None else None
			self.tb_logger = SummaryWriter(log_dir=tb_log_dir)

		optimizers = self.configure_optimizers()

		# Logging before fitting
		if self.tb_logger:
			fig = plot_gene_vs_time(train_data, valid_data_dict, test_data_dict, n_row=2, n_col=4, display_mode="tensorboard")
			self.tb_logger.add_figure('Dataset', fig)
			# self.tb_logger.add_graph(self)

		clock_st = time.time()
		for epoch in range(self.num_epochs):
			info = {}
			# Evaluation
			eval_freq = max(min(self.num_epochs // 10, 100), 1)
			plot_freq = (self.num_epochs // eval_freq // 10) * eval_freq
			print_freq = (self.num_epochs // eval_freq // 10) * eval_freq
			# print_freq = eval_freq * max((self.num_epochs // eval_freq // 10), 1)
			if valid_data_dict is not None and test_data_dict is not None and epoch % eval_freq == 0:
				if verbose and print_freq and epoch % print_freq == 0:
					print(f'epoch {epoch}')
					print("train time", time.time() - clock_st)
				clock_st = time.time()

				# Evaluation
				self.eval()
				info_valid = self.score(valid_data_dict)
				info_test = self.score(test_data_dict)
				info["valid"] = info_valid
				info["test"] = info_test
				if verbose and print_freq and epoch % print_freq == 0:
					print("eval time", time.time() - clock_st)
				clock_st = time.time()

				# print(epoch)
				# plot_hidden_space(self, valid_data_dict=valid_data_dict, test_data_dict=test_data_dict, display_mode="show")
				# plot_hidden_dynamics(self, valid_data_dict=valid_data_dict, test_data_dict=test_data_dict, display_mode="show")
				# if epoch >= 200:
				# 	assert False

				# Logging after evaluation
				if self.tb_logger:
					for valid_name, valid_result in info_valid.items():
						self.tb_logger.add_scalar(f"Score/valid-{valid_name:.3f}", valid_result.get("score"), epoch)
					for test_name, test_result in info_test.items():
						self.tb_logger.add_scalar(f"Score/test-{test_name:.3f}", test_result.get("score"), epoch)
					# fig = plot_hidden_space(self, valid_data_dict=valid_data_dict, test_data_dict=test_data_dict, display_mode="tensorboard")
					# self.tb_logger.add_figure('Hidden Space', fig, epoch)
					for name, param in self.named_parameters():
						name_slash = name.replace(".", "/")
						if param.grad is not None:
							self.tb_logger.add_histogram(name_slash + '/grad', param.grad, epoch)
						self.tb_logger.add_histogram(name_slash + '/data', param, epoch)
					if plot_freq and epoch % plot_freq == 0:
						fig = plot_generated_samples(self, valid_data_dict=valid_data_dict, test_data_dict=test_data_dict, display_mode="tensorboard")
						self.tb_logger.add_figure('Generated Samples', fig, epoch)
						self.callback_eval_plot(epoch, train_data, valid_data_dict, test_data_dict)

				if verbose and print_freq and epoch % print_freq == 0:
					print("log time", time.time() - clock_st)
				clock_st = time.time()

			# Training
			self.train()
			loss_dict = self.training_step(train_data, epoch)
			info["train"] = loss_dict
			loss = loss_dict.get("loss")
			optimizer = self.choose_optimizer(optimizers, epoch)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			info_history.append(info)

			# Logging after an epoch
			if self.tb_logger:
				for loss_name, loss_tensor in loss_dict.items():
					self.tb_logger.add_scalar(f"Loss/{loss_name}", loss_tensor, epoch)

			self.callback_epoch(epoch, train_data, valid_data_dict, test_data_dict)

		# print(self.score(valid_data_dict))
		return info_history


class VAE(VAEBaseModel):
	def __init__(self, input_dim: int, latent_dim: int, hidden_dims=None, device="cpu", kld_weight=1.0,
				 num_epochs=1000, learning_rate=1e-3):
		super().__init__()

		self.learning_rate = learning_rate
		self.num_epochs = num_epochs
		self.kld_weight = kld_weight

		self.latent_dim = latent_dim

		if hidden_dims is None:
			hidden_dims = [32, 64, 128, 256, 512]

		self.encoder = MLP(input_dim, None, hidden_dims)
		# hidden Space
		self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
		self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
		# Decoder
		self.decoder = MLP(latent_dim, input_dim, hidden_dims[::-1])

		self.device = device
		self.to(device)
		# print(self)

	def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		o = self.encoder(x)
		mu = self.fc_mu(o)
		log_var = self.fc_var(o)
		return mu, log_var

	def decode(self, z: torch.Tensor) -> torch.Tensor:
		# result = result.view(-1, 512, 2, 2)
		o = self.decoder(z)
		return o

	def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)
		return eps * std + mu

	def forward(self, x: torch.Tensor):
		mu, log_var = self.encode(x)
		z = self.reparameterize(mu, log_var)
		recons = self.decode(z)
		return recons, x, mu, log_var, z

	def sample(self, num_samples: int, t: float, return_z=False):
		self.eval()
		z_repeated = torch.randn(num_samples, self.latent_dim).to(self.device)
		samples = self.decode(z_repeated)
		assert len(samples.shape) == 2 and samples.shape[0] == num_samples
		if not return_z:
			return samples
		else:
			return samples, z_repeated

	def training_step(self, train_batch, epoch_idx):
		self.train()
		X, t, c = train_batch[:]
		recons, inputs, mu, log_var, _ = self(X.to(self.device))
		recons_loss = F.mse_loss(recons, inputs)
		kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
		loss = recons_loss + self.kld_weight * kld_loss
		return {'loss': loss, 'loss_rec': recons_loss, 'loss_kld': kld_loss}
