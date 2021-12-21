import torch
from torch.nn import functional as F

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

from models.odevae import ODEVAE
from utils.tools import random_survival_function, to_numpy


class ODEVAEWithIndependentCox(ODEVAE):
	def __init__(self, input_dim: int, latent_dim: int, encoder_hidden_dims: list = None, ode_hidden_dims: list = None,
				 device: str = "cpu", learning_rate=1e-3, num_epochs=1000, kld_0_weight=1.0, kld_t_weight=1.0,
				 kld_t_type="kld", sample_method="pz"):

		super().__init__(input_dim, latent_dim, encoder_hidden_dims, ode_hidden_dims, device, learning_rate, num_epochs,
						 kld_0_weight, kld_t_weight, kld_t_type, sample_method)
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

