from typing import Tuple, Union

import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.nn import functional as F
from torchdiffeq import odeint

from models.mlp import MLP
from models.vae import VAEBaseModel
from utils.plotting import plot_hidden_dynamics


class ODEVAEOneSequence(VAEBaseModel):
	def __init__(self, input_dim: int, latent_dim: int, encoder_hidden_dims: list = None, ode_hidden_dims: list = None,
				 device: str = "cpu",
				 learning_rate=1e-3, num_epochs=1000, kld_weight_t=1.0, sample_method="pz"):
		super().__init__()

		self.learning_rate = learning_rate
		self.num_epochs = num_epochs
		self.kld_weight_t = kld_weight_t
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

		self.mvn = MultivariateNormal(torch.zeros(self.latent_dim).to(device), covariance_matrix=torch.eye(self.latent_dim).to(device))

		# self.ode_init = MLP(latent_dim, latent_dim, encoder_hidden_dims)
		self.ode_f = MLP(latent_dim, latent_dim, ode_hidden_dims, activation="tanh")		# state -> d(state)/dt
		self.decoder = MLP(latent_dim, input_dim, encoder_hidden_dims[::-1]) 	# state -> X

		self.device = device
		self.to(device)


	def encode(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		mu = torch.ones((x.shape[0], self.latent_dim)).to(self.device)
		log_var = torch.zeros((x.shape[0], self.latent_dim)).to(self.device)
		return mu, log_var

	def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)
		z = eps * std + mu
		z.requires_grad = True
		log_prob = self.mvn.log_prob(eps) - 0.5 * log_var.sum(dim=1)
		log_prob = log_prob.reshape(-1, 1)
		return z, log_prob

	def decode(self, z_t: torch.Tensor) -> torch.Tensor:
		o = self.decoder(z_t)
		return o

	def ode_func_test(self, t, x):
		return self.ode_f(x)

	def ode_func_train(self, t, xr):
		x, r = xr
		dx = self.ode_f(x)
		i = 0
		dr = torch.stack([
			torch.autograd.grad(outputs=dx[:, i], inputs=x, grad_outputs=torch.ones_like(dx[:, i]),
								retain_graph=True, create_graph=True)[0][:, i]
			for i in range(x.shape[1])
		], dim=1)
		dr = torch.sum(dr, dim=1)
		return dx, -dr

	def ode(self, z_0: torch.Tensor, t: torch.Tensor, r_0: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
		assert z_0.ndim == 2 and t.ndim == 2
		t_table = torch.cat([torch.tensor([0.0]).to(self.device), t.reshape(-1)])		# Append 0.0 to the front
		t_table, inverse_indices = torch.unique(t_table, sorted=True, return_inverse=True)
		inverse_indices = inverse_indices[1:]		# Remove the indices of 0.0
		# print(t_table)
		# print(inverse_indices)

		if r_0 is None:
			func, x_0 = self.ode_func_test, z_0
		else:
			func, x_0 = self.ode_func_train, (z_0, r_0)

		ode_result = odeint(func, x_0, t_table, method=self.ode_method, rtol=self.ode_rtol, atol=self.ode_atol, options=self.ode_options)
		ode_result = (ode_result, ) if r_0 is None else ode_result
		ret = []
		for res in ode_result:
			val = res.transpose(0, 1)[:, inverse_indices, :]
			assert all([(val[:, i, :] == res[inverse_indices[i], :, :]).all() for i in range(t.shape[0])])
			ret.append(val)
		return ret[0] if r_0 is None else tuple(ret)

	def forward(self, x: torch.Tensor, t: torch.Tensor):
		mu, log_var = self.encode(x, t)
		z_0, r_0 = self.reparameterize(mu, log_var)		# N x H
		z_t, r_t = self.ode(z_0, t, r_0)		# N x N x H
		r_t = r_t.squeeze(2)
		assert z_t.ndim == 3 and r_t.ndim == 2
		# print(z_t.shape, r_t.shape)
		recons = self.decode(z_t)		# N x N x H
		X_gt = x.reshape(1, x.shape[0], -1).repeat(x.shape[0], 1, 1)
		assert X_gt.ndim == 3 and X_gt.shape[0] == x.shape[0] and X_gt.shape[1] == x.shape[0]
		return recons, X_gt, z_0, z_t, r_0, r_t

	def sample(self, num_samples: int, t: float, return_z=False):
		self.eval()
		if self.train_data is None or self.sample_method == "pz":
			if self.train_data is None:
				print("[warning] the model is not trained")
			z_0_repeated = torch.randn(num_samples, self.latent_dim).to(self.device)
		# elif self.sample_method == "qz":
		# 	X_train, t_train, c_train = self.train_data[:]
		# 	X_train, t_train, c_train = X_train.to(self.device), t_train.to(self.device).reshape(-1, 1), c_train.to(
		# 		self.device).reshape(-1, 1)
		# 	sampling_indices = torch.randint(0, X_train.shape[0], (num_samples,))
		# 	mu_sampled, log_var_sampled = self.encode(X_train[sampling_indices], t_train[sampling_indices])
		# 	assert isinstance(mu_sampled, torch.Tensor) and mu_sampled.ndim == 2
		# 	assert isinstance(log_var_sampled, torch.Tensor) and log_var_sampled.ndim == 2
		# 	z_0_repeated, _ = self.reparameterize(mu_sampled, log_var_sampled)
		# # print(num_samples, t, mu_sampled.shape, mu_sampled[:2, :2], log_var_sampled[:2, :2].exp(), z_repeated[:2, :2])
		else:
			raise NotImplementedError
		t_repeated = torch.full((1, 1), t).to(self.device)
		z_t_repeated = self.ode(z_0_repeated, t_repeated)
		assert z_t_repeated.ndim == 3 and z_t_repeated.shape[0] == num_samples and z_t_repeated.shape[1] == 1
		z_t_repeated = z_t_repeated.squeeze(1)
		samples = self.decode(z_t_repeated)
		assert samples.ndim == 2 and samples.shape[0] == num_samples
		if not return_z:
			return samples
		else:
			return samples, z_t_repeated

	def _calculate_loss(self, recons, X_gt, z_0, z_t, r_0, r_t):
		recons_loss = F.mse_loss(recons, X_gt)
		rr_t = self.mvn.log_prob(z_t)		# S x N
		assert rr_t.ndim == 2 and r_t.ndim == 2
		kld_t_loss = (r_t - rr_t).mean()
		loss = recons_loss + self.kld_weight_t * kld_t_loss
		return {'loss': loss, 'loss_rec': recons_loss, 'loss_kld_t': kld_t_loss}

	def training_step(self, train_batch, epoch_idx):
		self.train()
		X, t, c = train_batch[:]
		X, t, c = X.to(self.device), t.to(self.device).reshape(-1, 1), c.to(self.device).reshape(-1, 1)
		recons, X_gt, z_0, z_t, r_0, r_t = self(X, t)
		loss_dict = self._calculate_loss(recons, X_gt, z_0, z_t, r_0, r_t)
		return loss_dict

	def callback_eval_plot(self, epoch, train_data, valid_data_dict=None, test_data_dict=None):
		if self.tb_logger:
			fig = plot_hidden_dynamics(self, valid_data_dict=valid_data_dict, test_data_dict=test_data_dict, display_mode="tensorboard")
			self.tb_logger.add_figure('Hidden Dynamics', fig, epoch)
