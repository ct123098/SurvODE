from datetime import time
from typing import Tuple, Union

import torch
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from torch import nn
from torch.distributions import MultivariateNormal
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint

from models.mlp import MLP
from models.vae import VAEBaseModel
from utils.plotting import plot_hidden_dynamics, plot_gene_vs_time, plot_generated_samples
from utils.tools import random_survival_function


class ODEVAEDistillation(VAEBaseModel):
	def __init__(self, input_dim: int, latent_dim: int, ode_dim: int,
				 encoder_hidden_dims: list = None, ode_hidden_dims: list = None, ode_init_hidden_dims: list = None,
				 cvae_hidden_dims: list = None,
				 device: str = "cpu",
				 learning_rate=1e-3, num_epochs=1000, num_epochs_distill=1000, weight_recons_cvae=1.0, weight_recons_ode=1.0,
				 weight_kld=1.0, weight_zt=1.0, sample_method="pz"):
		super().__init__()

		encoder_hidden_dims = encoder_hidden_dims if encoder_hidden_dims is not None else [32, 32]
		ode_hidden_dims = ode_hidden_dims if ode_hidden_dims is not None else [16, 16]
		ode_init_hidden_dims = ode_init_hidden_dims if ode_init_hidden_dims is not None else [16, 16]
		cvae_hidden_dims = cvae_hidden_dims if cvae_hidden_dims is not None else [16, 16]

		self.learning_rate = learning_rate
		self.num_epochs_train = num_epochs
		self.num_epochs = num_epochs + num_epochs_distill

		self.weight_recons_cvae = weight_recons_cvae
		self.weight_recons_ode = weight_recons_ode
		self.weight_kld = weight_kld
		self.weight_zt = weight_zt

		self.latent_dim = latent_dim
		self.sample_method = sample_method

		# self.ode_method = "rk4"
		# self.ode_rtol = None
		# self.ode_atol = None
		# self.ode_options = {"step_size": 1e-2}

		self.ode_method = "dopri5"
		self.ode_rtol = 1e-3
		self.ode_atol = 1e-3
		self.ode_options = None

		self.mvn = MultivariateNormal(torch.zeros(self.latent_dim).to(device),
									  covariance_matrix=torch.eye(self.latent_dim).to(device))

		self.encoder = MLP(input_dim + 1, None, encoder_hidden_dims)  # X, t, c -> features
		self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)  # features -> E(state)
		self.fc_var = nn.Linear(encoder_hidden_dims[-1], latent_dim)  # features -> V(state)

		self.cvae_net = MLP(latent_dim + 1, ode_dim, cvae_hidden_dims)
		self.ode_init = MLP(latent_dim, ode_dim, ode_init_hidden_dims)
		self.ode_net = MLP(ode_dim, ode_dim, ode_hidden_dims, activation="tanh")  # state -> d(state)/dt
		self.decoder = MLP(ode_dim, input_dim, encoder_hidden_dims[::-1])  # state -> X

		self.device = device
		self.to(device)

	def encode(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		assert x.ndim == 2 and t.ndim == 2
		o = torch.cat([x, t], dim=1)
		o = self.encoder(o)
		mu = self.fc_mu(o)
		log_var = self.fc_var(o)
		return mu, log_var

	def decode(self, z_t: torch.Tensor) -> torch.Tensor:
		o = self.decoder(z_t)
		return o

	def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)
		# log_prob = self.mvn.log_prob(eps) - 0.5 * log_var.sum(dim=1)
		# log_prob = log_prob.reshape(-1, 1)
		return eps * std + mu

	def ode_core(self, z_0: torch.Tensor, t: torch.Tensor) -> Union[torch.Tensor]:
		assert z_0.ndim == 2 and t.ndim == 1
		t_table = torch.cat([torch.tensor([0.0]).to(self.device), t.reshape(-1)])  # Append 0.0 to the front
		t_table, inverse_indices = torch.unique(t_table, sorted=True, return_inverse=True)
		inverse_indices = inverse_indices[1:]  # Remove the indices of 0.0
		assert inverse_indices.ndim == 1 and inverse_indices.shape[0] == t.shape[0]

		ode_func = lambda t, x: self.ode_net(x)
		ode_result = odeint(ode_func, z_0, t_table, method=self.ode_method,
							rtol=self.ode_rtol, atol=self.ode_atol, options=self.ode_options)  # T x N x H
		assert ode_result.ndim == 3 and ode_result.shape[0] == t_table.shape[0] and ode_result.shape[1] == z_0.shape[0]
		ret = ode_result[inverse_indices, :, :]
		assert ret.ndim == 3 and ret.shape[0] == t.shape[0] and ret.shape[0] == z_0.shape[0]
		return ret

	def ode(self, z: torch.Tensor, t: torch.Tensor) -> Union[torch.Tensor]:
		assert z.ndim == 2 and t.ndim == 2 and z.shape[0] == t.shape[0]
		z_0 = self.ode_init(z)
		ode_result = self.ode_core(z_0, t.reshape(-1))
		# print(ode_result.shape)
		assert ode_result.shape[0] == t.shape[0] and ode_result.shape[1] == z_0.shape[0] and ode_result.shape[2] == \
			   z_0.shape[1]
		gather_indices = torch.tensor([[[j for k in range(z_0.shape[1])] for j in range(z_0.shape[0])] for i in range(1)]).to(self.device)
		ret = torch.gather(ode_result, dim=0, index=gather_indices).squeeze(dim=0)
		assert ret.ndim == 2 and ret.shape == z_0.shape
		return ret

	def forward(self, x: torch.Tensor, t: torch.Tensor, skip_ode=False):
		assert x.ndim == 2 and t.ndim == 2
		mu, log_var = self.encode(x, t)
		z = self.reparameterize(mu, log_var)
		z_t_cvae = self.cvae_net(torch.cat([z, t], dim=1))
		recons_cvae = self.decode(z_t_cvae)
		if skip_ode:
			z_t_ode = None
			recons_ode = None
		else:
			z_t_ode = self.ode(z, t)
			recons_ode = self.decode(z_t_ode)
		return x, mu, log_var, z, z_t_cvae, z_t_ode, recons_cvae, recons_ode

	def sample(self, num_samples: int, t: float, return_z=False):
		self.eval()
		if self.train_data is None or self.sample_method == "pz":
			if self.train_data is None:
				print("[warning] the model is not trained")
			z_0_repeated = torch.randn(num_samples, self.latent_dim).to(self.device)
		elif self.sample_method == "qz":
			X_train, t_train, c_train = self.train_data[:]
			X_train, t_train, c_train = X_train.to(self.device), t_train.to(self.device).reshape(-1, 1), c_train.to(
				self.device).reshape(-1, 1)
			sampling_indices = torch.randint(0, X_train.shape[0], (num_samples,))
			mu_sampled, log_var_sampled = self.encode(X_train[sampling_indices], t_train[sampling_indices])
			assert isinstance(mu_sampled, torch.Tensor) and mu_sampled.ndim == 2
			assert isinstance(log_var_sampled, torch.Tensor) and log_var_sampled.ndim == 2
			z_0_repeated = self.reparameterize(mu_sampled, log_var_sampled)
		# print(num_samples, t, mu_sampled.shape, mu_sampled[:2, :2], log_var_sampled[:2, :2].exp(), z_repeated[:2, :2])
		else:
			raise NotImplementedError
		t_repeated = torch.full((num_samples, 1), t).to(self.device)
		z_t_repeated = self.ode(z_0_repeated, t_repeated)
		assert z_t_repeated.ndim == 2 and z_t_repeated.shape[0] == num_samples
		samples = self.decode(z_t_repeated)
		assert samples.ndim == 2 and samples.shape[0] == num_samples
		if not return_z:
			return samples
		else:
			return samples, z_t_repeated

	def _calculate_loss_train(self, x_gt, mu, log_var, z, z_t_cvae, z_t_ode, recons_cvae, recons_ode):
		loss_rec_cvae = F.mse_loss(recons_cvae, x_gt)
		loss_kld = ((-0.5 * log_var + 0.5 * mu ** 2 + 0.5 * log_var.exp() - 0.5).sum(dim=1)).mean()
		loss = self.weight_recons_cvae * loss_rec_cvae + self.weight_kld * loss_kld
		return {'loss': loss, 'loss_rec': loss_rec_cvae, 'loss_kld': loss_kld}

	def _calculate_loss_distill(self, x_gt, mu, log_var, z, z_t_cvae, z_t_ode, recons_cvae, recons_ode):
		loss_rec_ode = F.mse_loss(recons_ode, x_gt)
		loss_zt = F.mse_loss(z_t_ode, z_t_cvae)
		loss = self.weight_recons_ode * loss_rec_ode + self.weight_zt * loss_zt
		return {'loss': loss, 'loss_rec': loss_rec_ode, 'loss_zt': loss_zt}

	def training_step(self, train_batch, epoch_idx):
		self.train()
		X, t, c = train_batch[:]
		X, t, c = X.to(self.device), t.to(self.device).reshape(-1, 1), c.to(self.device).reshape(-1, 1)
		if epoch_idx < self.num_epochs_train:
			x_gt, mu, log_var, z, z_t_cvae, z_t_ode, recons_cvae, recons_ode = self(X, t, skip_ode=True)
			loss_dict = self._calculate_loss_train(x_gt, mu, log_var, z, z_t_cvae, z_t_ode, recons_cvae, recons_ode)
		else:
			x_gt, mu, log_var, z, z_t_cvae, z_t_ode, recons_cvae, recons_ode = self(X, t, skip_ode=False)
			loss_dict = self._calculate_loss_distill(x_gt, mu, log_var, z, z_t_cvae, z_t_ode, recons_cvae, recons_ode)
		return loss_dict

	def callback_eval_plot(self, epoch, train_data, valid_data_dict=None, test_data_dict=None):
		if self.tb_logger:
			fig = plot_hidden_dynamics(self, valid_data_dict=valid_data_dict, test_data_dict=test_data_dict,
									   display_mode="tensorboard")
			self.tb_logger.add_figure('Hidden Dynamics', fig, epoch)

	def configure_optimizers(self):
		optimizer_cvae = torch.optim.Adam(list(self.encoder.parameters()) +
										  list(self.fc_mu.parameters()) +
										  list(self.fc_var.parameters()) +
										  list(self.cvae_net.parameters()) +
										  list(self.decoder.parameters()), lr=self.learning_rate)
		optimizer_ode = torch.optim.Adam(list(self.ode_init.parameters()) +
										  list(self.ode_net.parameters()) +
										  list(self.decoder.parameters()), lr=self.learning_rate)
		return optimizer_cvae, optimizer_ode

	def choose_optimizer(self, optimizers, epoch):
		assert isinstance(optimizers, tuple) and len(optimizers) == 2
		return optimizers[0] if epoch < self.num_epochs_train else optimizers[1]


class ODEVAEDistillationIndCox(ODEVAEDistillation):
	def __init__(self, input_dim: int, latent_dim: int, ode_dim: int, encoder_hidden_dims: list = None,
				 ode_hidden_dims: list = None, ode_init_hidden_dims: list = None, cvae_hidden_dims: list = None,
				 device: str = "cpu", learning_rate=1e-3, num_epochs=1000, num_epochs_distill=1000,
				 weight_recons_cvae=1.0, weight_recons_ode=1.0, weight_kld=1.0, weight_zt=1.0, sample_method="pz"):

		super().__init__(input_dim, latent_dim, ode_dim, encoder_hidden_dims, ode_hidden_dims, ode_init_hidden_dims,
						 cvae_hidden_dims, device, learning_rate, num_epochs, num_epochs_distill, weight_recons_cvae,
						 weight_recons_ode, weight_kld, weight_zt, sample_method)

		self.cox_regression = None


	def training_step(self, train_batch, epoch_idx):
		self.train()
		X, t, c = train_batch[:]
		if self.cox_regression is None:
			assert epoch_idx == 0
			self.cox_regression = CoxnetSurvivalAnalysis(fit_baseline_model=True).fit(X, Surv.from_arrays(c, t))
			# self.cox_regression = CoxPHSurvivalAnalysis(alpha=0.1).fit(X, Surv.from_arrays(c, t))
		surv_funcs = self.cox_regression.predict_survival_function(X)
		t_sampled = torch.tensor([random_survival_function(fn) for fn in surv_funcs]).float().to(self.device).reshape(-1, 1)
		X, _, _ = X.to(self.device), t.to(self.device).reshape(-1, 1), c.to(self.device).reshape(-1, 1)
		assert isinstance(t_sampled, torch.Tensor) and t_sampled.ndim == 2 and t_sampled.shape[0] == X.shape[0]
		if epoch_idx < self.num_epochs_train:
			x_gt, mu, log_var, z, z_t_cvae, z_t_ode, recons_cvae, recons_ode = self(X, t_sampled, skip_ode=True)
			loss_dict = self._calculate_loss_train(x_gt, mu, log_var, z, z_t_cvae, z_t_ode, recons_cvae, recons_ode)
		else:
			x_gt, mu, log_var, z, z_t_cvae, z_t_ode, recons_cvae, recons_ode = self(X, t_sampled, skip_ode=False)
			loss_dict = self._calculate_loss_distill(x_gt, mu, log_var, z, z_t_cvae, z_t_ode, recons_cvae, recons_ode)
		return loss_dict

