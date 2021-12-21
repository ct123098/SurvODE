# SurvODE: Extrapolating Gene Expression Distribution for Early Cancer Identification

## Introduction

SurvODE provides a new perspective of prognostic gene identification by modeling the gene expression distribution at a given time. Please see our paper for detail: https://arxiv.org/abs/2111.15080. 

If you would like to cite our paper, the BibTex citation is as follows:

```bibtex
@misc{chen2021survode,
   title={SurvODE: Extrapolating Gene Expression Distribution for Early Cancer Identification}, 
   author={Tong Chen and Sheng Wang},
   year={2021},
   eprint={2111.15080},
   archivePrefix={arXiv},
   primaryClass={q-bio.GN}
}
```

## Setup

We used `conda` for package management. Please use the following command to create an environment. 

```bash
conda env create -f environment.yml
```

`environment_brief.yml` is a simplified package list. You can check it if needed. 

## Run

`tool_evaluate_dataset.py` evaluates the methods for generating gene expression at a given time. The parameters of the code is as follows:

```bash
python tool_evaluate_dataset.py --dataset {DATASET_NAME} --feature {FEATURE_NAME} --task {TASK_NAME} --gpu {GPU_ID}
```

### Example

If you want to evaluate methods on extrapolation task in BLCA dataset using 100 genes selected by Wasserstein distance, you can type:

```bash
python tool_evaluate_dataset.py --dataset BLCA --feature wdist100 --task extrapolation  --gpu 0
```