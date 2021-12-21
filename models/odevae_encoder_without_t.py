from typing import Tuple, Union

import torch
from sksurv.linear_model import CoxnetSurvivalAnalysis
from torch import nn
from torch.distributions import MultivariateNormal
from torch.nn import functional as F
from torchdiffeq import odeint

from models.mlp import MLP
from models.vae import VAEBaseModel
from utils.plotting import plot_hidden_dynamics
from utils.tools import random_survival_function


class ODEVAEEncoderWithoutT(VAEBaseModel):
	def __init__(self, input_dim: int, latent_dim: int, encoder_hidden_dims: list = None, ode_hidden_dims: list = None,
				 device: str = "cpu",
				 learning_rate=1e-3, num_epochs=1000, kld_weight_0=1.0, kld_weight_t=1.0, sample_method="pz"):
		super().__init__()

		self.learning_rate = learning_rate
		self.num_epochs = num_epochs
		self.kld_weight_0 = kld_weight_0
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

		self.encoder = MLP(input_dim, None, encoder_hidden_dims)		# X, t, c -> features
		self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)		# features -> E(state)
		self.fc_var = nn.Linear(encoder_hidden_dims[-1], latent_dim)		# features -> V(state)
		self.ode_f = MLP(latent_dim, latent_dim, ode_hidden_dims, activation="tanh")		# state -> d(state)/dt
		self.decoder = MLP(latent_dim, input_dim, encoder_hidden_dims[::-1]) 	# state -> X

		self.device = device
		self.to(device)


	def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		o = self.encoder(x)
		mu = self.fc_mu(o)
		log_var = self.fc_var(o)
		return mu, log_var

	def decode(self, z_t: torch.Tensor) -> torch.Tensor:
		o = self.decoder(z_t)
		return o

	def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)
		log_prob = self.mvn.log_prob(eps) - 0.5 * log_var.sum(dim=1)
		log_prob = log_prob.reshape(-1, 1)
		return eps * std + mu, log_prob

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

		ode_result = odeint(func, x_0, t_table,
							method=self.ode_method, rtol=self.ode_rtol, atol=self.ode_atol, options=self.ode_options)
		ode_result = (ode_result, ) if r_0 is None else ode_result
		ret = []
		for res in ode_result:
			# print(res.shape)
			val = res[inverse_indices, :, :].diagonal(dim1=0, dim2=1).transpose(dim0=0, dim1=1)
			ret.append(val)
		return ret[0] if r_0 is None else tuple(ret)

	def forward(self, x: torch.Tensor, t: torch.Tensor):
		mu, log_var = self.encode(x)
		z_0, r_0 = self.reparameterize(mu, log_var)
		z_t, r_t = self.ode(z_0, t, r_0)
		recons = self.decode(z_t)
		return recons, x, mu, log_var, z_0, z_t, r_0, r_t

	def sample(self, num_samples: int, t: float, return_z=False):
		self.eval()
		if self.train_data is None or self.sample_method == "pz":
			if self.train_data is None:
				print("[warning] the model is not trained")
			z_0_repeated = torch.randn(num_samples, self.latent_dim).to(self.device)
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

	def _calculate_loss(self, recons, inputs, mu, log_var, z_0, z_t, r_0, r_t):
		recons_loss = F.mse_loss(recons, inputs)
		rr_0 = self.mvn.log_prob(z_0)		# N
		rr_t = self.mvn.log_prob(z_t)		# N
		assert rr_0.ndim == 1 and rr_t.ndim == 1
		kld_0_loss = (r_0 - rr_0).mean()
		# kld_0_std = ((-0.5 * log_var + 0.5 * mu ** 2 + 0.5 * log_var.exp() - 0.5).sum(dim=1)).mean()
		# print(kld_0_std.item(), kld_0_loss.item())
		kld_t_loss = (r_t - rr_t).mean()
		loss = recons_loss + self.kld_weight_0 * kld_0_loss + self.kld_weight_t * kld_t_loss
		return {'loss': loss, 'loss_rec': recons_loss, 'loss_kld': kld_0_loss, 'loss_kld_t': kld_t_loss}

	def training_step(self, train_batch, epoch_idx):
		self.train()
		X, t, c = train_batch[:]
		X, t, c = X.to(self.device), t.to(self.device).reshape(-1, 1), c.to(self.device).reshape(-1, 1)
		recons, inputs, mu, log_var, z_0, z_t, r_0, r_t = self(X, t)
		loss_dict = self._calculate_loss(recons, inputs, mu, log_var, z_0, z_t, r_0, r_t)
		return loss_dict

	# def callback_eval_plot(self, epoch, train_data, valid_data_dict=None, test_data_dict=None):
	# 	if self.tb_logger:
	# 		fig = plot_hidden_dynamics(self, valid_data_dict=valid_data_dict, test_data_dict=test_data_dict, display_mode="tensorboard")
	# 		self.tb_logger.add_figure('Hidden Dynamics', fig, epoch)


class ODEVAEWithIndependentCoxEncoderWithoutT(ODEVAEEncoderWithoutT):
	def __init__(self, input_dim: int, latent_dim: int, encoder_hidden_dims: list = None, ode_hidden_dims: list = None,
				 device: str = "cpu",
				 learning_rate=1e-3, num_epochs=1000, kld_weight_0=1.0, kld_weight_t=1.0):
		super().__init__(input_dim, latent_dim, encoder_hidden_dims=encoder_hidden_dims, ode_hidden_dims=ode_hidden_dims,
						 device=device, learning_rate=learning_rate, num_epochs=num_epochs, kld_weight_0=kld_weight_0,
						 kld_weight_t=kld_weight_t)

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
		recons, inputs, mu, log_var, z_0, z_t, r_0, r_t = self(X, t_sampled)
		loss_dict = self._calculate_loss(recons, inputs, mu, log_var, z_0, z_t, r_0, r_t)
		return loss_dict

