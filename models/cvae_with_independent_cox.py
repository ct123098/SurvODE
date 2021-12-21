import numpy as np
import sksurv
import torch
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.util import Surv
from torch import nn
from torch.nn import functional as F
from models.base import BaseModel
from models.cvae import ConditionalVAE
from models.mlp import MLP
from models.vae import VAEBaseModel
from utils import metrics
from matplotlib import pyplot as plt
from utils.tools import random_survival_function, to_numpy


class CVAEWithIndependentCox(ConditionalVAE):
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
