import matplotlib
import numpy
import numpy as np
import torch
from sklearn.decomposition import PCA

from datasets.expression import ExpressionDataBase
from utils import metrics
from utils.tools import to_numpy, to_value
import matplotlib.pyplot as plt


def plot_training_loss_curve(info_history):
    plt.figure(figsize=(20, 8))
    names = [name for name in info_history[-1].get("train").keys() if name.startswith("loss")]
    assert len(names) <= 10, f"cannot plot so many loss ({len(names)})"
    for i, name in enumerate(names):
        plt.subplot(2, 5, 1 + i)
        points = [(j, to_value(info.get("train").get(name))) for j, info in enumerate(info_history) if name in info.get("train")]
        x, y = tuple(zip(*points))
        plt.plot(x, y)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(name)
    plt.tight_layout()
    plt.show()

def plot_training_evaluation(info_history, metric_name="wdist"):
    plt.figure(figsize=(20, 8))
    points = []
    type_dict = {}
    for j, info in enumerate(info_history):
        if "valid" in info or "test" in info:
            info_valid = info.get("valid", {})
            info_test = info.get("test", {})
            info_all = {**info_valid, **info_test}
            for i, (t, value_dict) in enumerate(info_all.items()):
                type_dict[i] = "valid" if i < len(info_valid) else "test"
                points.append((i, t, j, value_dict[metric_name]))
    cnt = max(i for i, _, _, _ in points) + 1
    assert cnt <= 10, f"cannot plot so many cases ({cnt})"
    for i in range(cnt):
        plt.subplot(2, 5, 1 + i)
        t, x, y = tuple(zip(*[(tt, j, value) for ii, tt, j, value in points if ii == i]))
        plt.plot(x, y)
        plt.xlabel("epoch")
        plt.ylabel("wdist")
        plt.title(f'{type_dict[i]} | t = {t[0]:.3f}')
    plt.tight_layout()
    plt.show()

def merge_data_dict(data_dict: dict):
    X, t, c, patient_name = [numpy.concatenate(array_list, axis=0) for array_list in list(zip(*[
        (dataset.X, dataset.t, dataset.c, dataset.patient_name) for dataset in data_dict.values()
    ]))]
    gene_name = next(iter(data_dict.values())).get_gene_name()
    return ExpressionDataBase(X=X, t=t, c=c, gene_name=gene_name, patient_name=patient_name)

def plot_generated_samples(model, valid_data_dict=None, test_data_dict=None, display_mode="show"):
    valid_data_dict = {} if valid_data_dict is None else valid_data_dict
    test_data_dict = {} if test_data_dict is None else test_data_dict
    fig = plt.figure(figsize=(16, 12), dpi=50)
    NUM_SAMPLES = 128
    all_data_dict = {**valid_data_dict, **test_data_dict}
    X_all, _, _ = merge_data_dict(all_data_dict)[:]
    pca = PCA(n_components=2).fit(to_numpy(X_all))
    X_all_reduced = pca.transform(X_all)
    for i, (tt, data) in enumerate(all_data_dict.items()):
        plt.subplot(3, 4, 1 + i)
        X_generated = model.sample(NUM_SAMPLES, tt)
        X, t, _ = data[:]
        X_generated_reduced = pca.transform(to_numpy(X_generated))
        X_reduced = pca.transform(to_numpy(X))
        # wdict = metrics.wasserstein_distance(X_generated, X)
        # info_eval[t] = wdict

        plt.scatter(X_generated_reduced[:, 0], X_generated_reduced[:, 1], alpha=0.5, color="black")
        cmap = lambda x: matplotlib.cm.get_cmap('hsv')((x + 1.0) / 4.0)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.1, c=cmap(t))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cmap(t))
        plt.xlim(X_all_reduced[:, 0].min(), X_all_reduced[:, 0].max())
        plt.ylim(X_all_reduced[:, 1].min(), X_all_reduced[:, 1].max())
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        wdist = metrics.wasserstein_distance(X_generated, X)
        eval_type = "valid" if i < len(valid_data_dict) else "test"
        plt.title(f'{eval_type} | t = {tt:.3f} | wdist={wdist:.3f}')
    plt.tight_layout()
    if display_mode == "show":
        plt.show()
        plt.close(fig)
    elif display_mode == "tensorboard":
        return fig
    else:
        raise NotImplementedError


def plot_hidden_space(model, valid_data_dict=None, test_data_dict=None, display_mode="show"):
    valid_data_dict = {} if valid_data_dict is None else valid_data_dict
    test_data_dict = {} if test_data_dict is None else test_data_dict
    fig = plt.figure(figsize=(16, 12), dpi=50)
    NUM_SAMPLES = 128
    all_data_dict = {**valid_data_dict, **test_data_dict}
    X_all, _, _ = merge_data_dict(all_data_dict)[:]
    zmax = 0.0
    z_generated_list = []
    for i, (tt, data) in enumerate(all_data_dict.items()):
        X_generated, z_generated = model.sample(NUM_SAMPLES, tt, return_z=True)
        z_generated_list.append(to_numpy(z_generated[:, :2]))
        zmax = max(zmax, np.max(np.abs(z_generated_list[-1])))
    for i, (tt, data) in enumerate(all_data_dict.items()):
        plt.subplot(3, 4, 1 + i)
        plt.scatter(z_generated_list[i][:, 0], z_generated_list[i][:, 1], alpha=0.5, color="black")
        plt.xlim(-zmax, zmax)
        plt.ylim(-zmax, zmax)
        plt.xlabel("DIM 1")
        plt.ylabel("DIM 2")
        eval_type = "valid" if i < len(valid_data_dict) else "test"
        plt.title(f'{eval_type} | t = {tt:.3f}')
    plt.tight_layout()
    if display_mode == "show":
        plt.show()
        plt.close(fig)
    elif display_mode == "tensorboard":
        return fig
    else:
        raise NotImplementedError


def plot_hidden_dynamics(model, valid_data_dict=None, test_data_dict=None, display_mode="show"):
    fig = plt.figure(figsize=(8, 4), dpi=50)

    all_data_dict = {**valid_data_dict, **test_data_dict}
    X_all, t_all, c_all = merge_data_dict(all_data_dict)[:]
    mu_0_all, log_var_0_all = model.encode(X_all.to(model.device), t_all.to(model.device).reshape(-1, 1))
    std_0_all = torch.exp(0.5 * log_var_0_all)
    z_0_all = mu_0_all + torch.randn_like(std_0_all) * std_0_all

    stride = z_0_all.shape[0] // 25

    z_0_list = []
    color_list = []
    z_0_list.append(z_0_all[::stride])
    color_list.extend(["blue"] * z_0_all[::stride].shape[0])
    z_0_list.append(mu_0_all[::stride])
    color_list.extend(["cyan"] * mu_0_all[::stride].shape[0])
    z_0_list.append(torch.zeros_like(z_0_all[:1]))
    color_list.append("green")
    z0_array = to_numpy(torch.cat(z_0_list, dim=0))

    plt.subplot(1, 2, 1)
    pca_z0 = PCA(n_components=2).fit(to_numpy(z0_array))
    z0_array_reduced = pca_z0.transform(z0_array)
    for z_0_reduced_np, color_init in zip(z0_array_reduced, color_list):
        plt.scatter(z_0_reduced_np[0], z_0_reduced_np[1], color=color_init)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f'distribution of z_0 space')

    plt.subplot(1, 2, 2)
    tmax = max(max(valid_data_dict.keys()), max(test_data_dict.keys()))
    T = 20
    t_array = torch.linspace(0, tmax, T).to(model.device)
    z_array_list = []
    for z_0_np in z0_array:
        z_0 = torch.tensor(z_0_np).to(model.device)
        z_list = []
        # z_t = model.cvae_net(torch.cat([z_0.reshape(1, -1).repeat(T, 1), t_array.reshape(-1, 1)], dim=1))
        if hasattr(model, "ode"):
            z_t = model.ode(z_0.reshape(1, -1).repeat(T, 1), t_array.reshape(-1, 1))
        elif hasattr(model, "cvae_net"):
            z_t = model.dynamics_net(torch.cat([z_0.reshape(1, -1).repeat(T, 1), t_array.reshape(-1, 1)], dim=1))
        else:
            z_t = z_0.reshape(1, -1).repeat(T, 1)
        z_list.append(to_numpy(z_t[:, :2]))
        z_array = np.concatenate(z_list, axis=0)
        z_array_list.append(z_array)
    z_all = np.stack(z_array_list, axis=0)
    pca_z = PCA(n_components=2).fit(z_all.reshape(-1, z_all.shape[-1]))
    z0_array_reduced = pca_z.transform(z_all.reshape(-1, z_all.shape[-1])).reshape(z_all.shape[0], z_all.shape[1], -1)

    for z_array, color_init in zip(z0_array_reduced, color_list):
        cmap = lambda x: matplotlib.cm.get_cmap('hsv')((x + 1.0) / 4.0)
        plt.scatter(z_array[-1, 0], z_array[-1, 1], alpha=0.5, color="pink")
        plt.scatter(z_array[:, 0], z_array[:, 1], alpha=0.1, c=cmap(to_numpy(t_array)))
        plt.scatter(z_array[0, 0], z_array[0, 1], alpha=0.5, color=color_init)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f'dynamics of z_t space')
    plt.tight_layout()
    if display_mode == "show":
        plt.show()
        plt.close(fig)
    elif display_mode == "tensorboard":
        return fig
    else:
        raise NotImplementedError


def plot_gene_vs_time(train_data, valid_data_dict=None, test_data_dict=None, n_row=2, n_col=4, display_mode="show"):
    valid_data = merge_data_dict(valid_data_dict)
    test_data = merge_data_dict(test_data_dict)
    N = train_data[:][0].shape[0]
    train_data_order = torch.argsort(train_data[:][1])
    # print(train_data[:][1][train_data_order[:N // 2]])
    # print(train_data[:][1][train_data_order[N // 2:]])
    fig = plt.figure(figsize=(4 * n_col, 4 * n_row), dpi=50)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, 1 + i)
        # COLOR: faee1c, f3558e, 581b98
        plt.scatter(test_data[:][1], test_data[:][0][:, i], color="blue", alpha=0.25, label="test")
        plt.scatter(valid_data[:][1], valid_data[:][0][:, i], color="green", alpha=0.25, label="valid")
        plt.scatter(train_data[:][1], train_data[:][0][:, i], color="red", alpha=0.25, label="train")
        gene_name = train_data.get_gene_name()[i]
        assert train_data.get_gene_name()[i] == valid_data.get_gene_name()[i] == test_data.get_gene_name()[i]
        corrcoef = metrics.corrcoef(train_data[:][0][:, i], train_data[:][1])
        wdist = metrics.wasserstein_distance(train_data[:][0][train_data_order[:N // 2], i], train_data[:][0][train_data_order[N // 2:], i])
        plt.title(f'{gene_name} | r={corrcoef:.3f} | wd={wdist:.3f}')
        plt.axhline(0, color="gray")
        plt.xlabel("Time")
        plt.ylabel("Gene Expression")
        plt.legend()
    plt.tight_layout()
    if display_mode == "show":
        plt.show()
        plt.close(fig)
    elif display_mode == "tensorboard":
        return fig
    else:
        raise NotImplementedError


# def plot_gene_vs_time(X_train, y_train, X_test=None, y_test=None, X_line=None, y_line=None, gene_indices=None, plot_indices=None):
#     # assert gene_indices is not None, "gene indices should no be None"
#     if gene_indices is None:
#         gene_indices = range(min(20, X_train.shape[1]))
#     assert len(gene_indices) <= 20, "the lengthe should not be larger than 20"
#     plt.figure(figsize=(20, 16))
#     for plot_id, gene_id in enumerate(gene_indices):
#         plt.subplot(4, 5, 1 + plot_id)
#         plt.scatter(y_train, X_train[:, gene_id], color="#99d5ff", label="train")
#         if X_test is not None and y_test is not None:
#             plt.scatter(y_test, X_test[:, gene_id], color="#ffd699", label="test")
#         if X_line is not None and y_line is not None:
#             plt.scatter(y_line, X_line[:, gene_id], color="gray", label="line")
#         corrcoef = np.corrcoef(X_train[:, gene_id], y_train)[0, 1]
#         plt.title(f"{gene_id}" if plot_indices is None else f'{plot_indices[plot_id]}')
#         plt.axhline(0, color="gray")
#         plt.legend()
#         plt.xlabel(f"Time | r^2={corrcoef:.3f}")
#         plt.ylabel("Gene Exp.")
#     plt.tight_layout()
#     plt.show()
#     # plt.savefig("tmp.jpg")
#     # plt.close()

def plot_init():
    matplotlib.rcParams['pdf.fonttype'] = 42
    # Lemur 1 = Bernard (male); Lemur 2 = Stumpy (female); Lemur 3 = Martine (female); Lemur 4 = Antoine (male)
    MEDIUM_SIZE = 8
    SMALLER_SIZE = 6
    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('xtick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
    plt.rc('figure', titlesize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)