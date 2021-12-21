# Downloading from an Open Source Implementation
# URL: https://raw.githubusercontent.com/AntixK/PyTorch-VAE/master/models/cvae.py
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from datasets.expression import ExpressionDataBase
from utils import metrics


class BaseModel(nn.Module):
	def __init__(self):
		super().__init__()

	def configure_optimizers(self) -> torch.optim.Optimizer:
		# optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		# return optimizer
		raise NotImplementedError

	def choose_optimizer(self, optimizers, epoch):
		raise NotImplementedError

	def training_step(self, train_batch: ExpressionDataBase, epoch_idx: int) -> Dict[str, torch.Tensor]:
		# x, y = train_batch
		# x = x.view(x.size(0), -1)
		# z = self.encoder(x)
		# x_hat = self.decoder(z)
		# loss = F.mse_loss(x_hat, x)
		# self.log('train_loss', loss)
		# return loss
		raise NotImplementedError

	def fit(self, train_data: ExpressionDataBase, valid_data_dict: Dict[float, ExpressionDataBase] = None,
			test_data_dict: Dict[float, ExpressionDataBase] = None, verbose=False) -> List[dict]:
		raise NotImplementedError

	def sample(self, num_samples: int, t: float) -> torch.Tensor:
		raise NotImplementedError

	def score(self, data_dict: dict) -> Dict[float, dict]:
		self.eval()
		NUM_SAMPLES = 128
		info_eval = {}
		for t, data in data_dict.items():
			X_generated = self.sample(NUM_SAMPLES, t)
			X, _, _ = data[:]
			wdist = metrics.wasserstein_distance(X_generated, X)
			precision = metrics.precision(X_generated, X)
			recall = metrics.recall(X_generated, X)
			info_eval[t] = {"score": wdist, "wdist": wdist, "precision": precision, "recall": recall}
		return info_eval

	def callback_eval_plot(self, epoch, train_data, valid_data_dict=None, test_data_dict=None):
		pass

	def callback_epoch(self, epoch, train_data, valid_data_dict=None, test_data_dict=None):
		pass

	def save(self, path):
		torch.save(self.state_dict(), path)
		pass

	def load(self, path):
		self.load_state_dict(torch.load(path))
		self.eval()


# class BaseVAE(nn.Module):

#     def __init__(self) -> None:
#         super().__init__()

#     def encode(self, input: Tensor) -> Tuple[Tensor]:
#         raise NotImplementedError

#     def decode(self, input: Tensor) -> Any:
#         raise NotImplementedError

#     def sample(self, batch_size:int, *args, **kwargs) -> Tensor:
#         raise RuntimeWarning()

#     def generate(self, x: Tensor, **kwargs) -> Tensor:
#         raise NotImplementedError

#     @abstractmethod
#     def forward(self, *inputs: Any, **kwargs) -> Tuple[Tensor]:
#         pass

#     @abstractmethod
#     def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
#         pass
