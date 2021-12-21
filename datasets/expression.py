import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

from typing import Tuple

from datasets.feature import select_features_reg, select_features_wdist, select_features_nwdist
from datasets.transformation import scale_gene, scale_time


class ExpressionDataBase(Dataset):
	def __init__(self, X: np.ndarray = None, t: np.ndarray = None, c: np.ndarray = None, gene_name: np.ndarray = None,
				 patient_name: np.ndarray = None):
		super(ExpressionDataBase, self).__init__()
		self.X = X
		self.t = t
		self.c = c
		self.gene_name = gene_name
		self.patient_name = patient_name

	def __len__(self) -> int:
		return self.X.shape[0]

	def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		X, t, c = self.X[idx], self.t[idx], self.c[idx]
		return torch.tensor(X).float(), torch.tensor(t).float(), torch.tensor(c).bool()

	def get_gene_name(self) -> np.ndarray:
		return self.gene_name

	def get_patient_name(self) -> np.ndarray:
		return self.patient_name


class ExpressionData(ExpressionDataBase):
	def __init__(self, cancer_name, load_data=True, save_data=True, feature_selection=None, enable_scaling=None,
				 verbose=False):
		super(ExpressionData, self).__init__()
		self.verbose = verbose
		project_root = Path(__file__).parent.parent
		save_path = project_root / 'data' / 'preprocessed' / f'{cancer_name}.npz'
		gene_path = project_root / 'data' / 'expression' / f'{cancer_name}.txt'
		survival_path = project_root / 'data' / 'raw_survival' / f'{cancer_name}_surv.txt_clean'
		if load_data:
			try:
				self._load(save_path)
			except Exception as e:
				if self.verbose:
					print(f"Failed to load preprocessing data ({e})")
				self._process_data(gene_path, survival_path)
		else:
			self._process_data(gene_path, survival_path)
		if save_data:
			self._save(save_path)
		self._transformation(enable_scaling)
		self._select_feature(feature_selection)

	def _process_data(self, gene_path, survival_path):
		if self.verbose:
			print("Processing Data ... ")
		timestamp_0 = time.process_time()
		self._load_data_frame(gene_path, survival_path)
		timestamp_1 = time.process_time()
		self._clean_data()
		timestamp_2 = time.process_time()
		self._generate_table()
		timestamp_3 = time.process_time()
		self._tidy_up()
		if self.verbose:
			print(
				f'read {timestamp_1 - timestamp_0:.3f}s, clean {timestamp_2 - timestamp_1:.3f}s, '
				f'create {timestamp_3 - timestamp_2:.3f}s'
			)

	def _select_feature(self, feature_selection: str):
		if feature_selection == None:
			return
		if feature_selection.startswith("reg"):
			max_number = int(feature_selection[len("reg"):])
			key_gene_indices = select_features_reg(self.X, self.t, self.c, max_number, verbose=self.verbose)
			assert len(key_gene_indices) <= max_number
		elif feature_selection.startswith("wdist"):
			max_number = int(feature_selection[len("wdist"):])
			key_gene_indices = select_features_wdist(self.X, self.t, self.c, max_number, verbose=self.verbose)
			assert len(key_gene_indices) <= max_number
		elif feature_selection.startswith("nwdist"):
			max_number = int(feature_selection[len("nwdist"):])
			key_gene_indices = select_features_nwdist(self.X, self.t, self.c, max_number, verbose=self.verbose)
			assert len(key_gene_indices) <= max_number
		else:
			raise NotImplementedError
		self.X = self.X[:, key_gene_indices]
		self.gene_name = self.gene_name[key_gene_indices]

	# plot_gene_vs_time(self.X, self.t, gene_indices=key_gene_indices[:20], plot_indices=self.gene_name[key_gene_indices[:20]])
	# print(key_gene_indices)

	def _transformation(self, enable_scaling):
		if enable_scaling:
			self.X, self.scaler_X = scale_gene(self.X)
			self.t, self.scaler_t = scale_time(self.t)

	# print(self.X[:4, :4])
	# print(self.t[:4])

	def _load_data_frame(self, gene_path, survival_path):
		self.df_gene_raw = pd.read_csv(str(gene_path), sep="\t", header=None)
		self.df_surv_raw = pd.read_csv(str(survival_path), sep="\t")

	def _clean_data(self):
		self.df_gene_clean = self.df_gene_raw.dropna()[(self.df_gene_raw[1] != "?") & (self.df_gene_raw[2] > 0.0)]
		self.df_surv_clean = self.df_surv_raw.dropna()
		self.df_gene_clean = self.df_gene_clean[self.df_gene_clean[0].isin(self.df_surv_clean["ID"])]
		self.df_surv_clean = self.df_surv_clean[self.df_surv_clean["ID"].isin(self.df_gene_clean[0])]
		N = self.df_surv_clean.shape[0]
		counts = self.df_gene_clean[1].value_counts()
		gene_name_index = counts[counts > 0.95 * N].index
		self.df_gene_clean = self.df_gene_clean[self.df_gene_clean[1].isin(gene_name_index)]

	def _generate_table(self):
		name_array = self.df_surv_clean["ID"].value_counts().index
		gene_array = self.df_gene_clean[1].value_counts().index
		# print(name_array)
		# print(gene_array)
		# print(self.df_gene_clean[1].value_counts())
		# print(self.df_gene_clean[1].value_counts()["C20orf166"])
		name_mapping = {name: i for i, name in enumerate(name_array)}
		gene_mapping = {gene: i for i, gene in enumerate(gene_array)}
		# print(name_mapping)
		# print(gene_mapping)
		matrix = np.zeros([name_array.shape[0], gene_array.shape[0]])
		# print(matrix.shape)
		# import time
		# time_st = time.clock()
		for name, gene, value in self.df_gene_clean.to_numpy():
			if name in name_mapping:
				assert gene in gene_mapping
				name_index = name_mapping[name]
				gene_index = gene_mapping[gene]
				matrix[name_index][gene_index] = max(matrix[name_index][gene_index], value)
		# print(time.clock() - time_st)
		self.df_gene = pd.DataFrame(matrix, columns=gene_array, index=name_array)
		# df_gene = df_gene.loc[name_array, :]
		assert self.df_gene.index.equals(name_array)
		# print(df_gene)
		self.df_surv = self.df_surv_clean[["ID", "OS_STATUS", "OS_MONTHS"]].set_index("ID")
		self.df_surv["OS_STATUS"] = self.df_surv["OS_STATUS"].astype(bool)
		self.df_surv = self.df_surv.loc[name_array, :]
		assert self.df_surv.index.equals(name_array)
		assert len(self.df_gene) == len(self.df_surv)

	def _tidy_up(self):
		gene = self.df_gene.to_numpy()
		surv = self.df_surv.to_numpy()
		self.X = gene.copy()
		self.c = surv[:, 0].astype(np.bool)
		self.t = surv[:, 1].astype(np.float64)
		self.gene_name = self.df_gene.columns.to_numpy().astype(str)
		self.patient_name = self.df_gene.index.to_numpy().astype(str)
		if self.verbose:
			print(self.X.shape, self.X.dtype)
			print(self.t.shape, self.t.dtype)
			print(self.c.shape, self.c.dtype)
			print(self.gene_name.shape, self.gene_name.dtype, self.gene_name[:2])
			print(self.patient_name.shape, self.patient_name.dtype, self.patient_name[:2])

		del self.df_gene_raw, self.df_surv_raw
		del self.df_gene_clean, self.df_surv_clean
		del self.df_gene, self.df_surv

	def _save(self, save_path):
		save_path.parent.mkdir(parents=True, exist_ok=True)
		np.savez(str(save_path), X=self.X, t=self.t, c=self.c, gene_name=self.gene_name, patient_name=self.patient_name)
		if self.verbose:
			print(f'Save preprocessed data to {str(save_path)}')

	def _load(self, save_path):
		data = np.load(str(save_path), allow_pickle=True)
		self.X, self.t, self.c = data["X"], data["t"], data["c"]
		self.gene_name, self.patient_name = data["gene_name"], data["patient_name"]
		if self.verbose:
			print(f'Load preprocessed data from {str(save_path)}')
