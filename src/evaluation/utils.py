import numpy as np
import torch
import os
from torchmetrics import MetricCollection
import numcodecs

from .metrics import ConfusionStatsOverall, BinaryStatsOverallWindow

compressor = numcodecs.Blosc(
        cname="zstd",      # best compression ratio
        clevel=7,          # 1–5 is usually optimal
        shuffle=numcodecs.Blosc.BITSHUFFLE
    )

def detach_all(z):
    if isinstance(z, dict):
        z_ = {}
        for k, v in z.items():
            z_[k] = detach_all(v)
        z = z_
    elif isinstance(z, list):
        z = [z_.detach().cpu().numpy() for z_ in z]
    else:
        z = z.detach().cpu().numpy()
    return z

def collate_all_list(z, z_):
    if isinstance(z_, dict):
        for k, v in z_.items():
            collate_all_list(z[k], v)
    elif isinstance(z_,list):
        for i, z_i in enumerate(z_):
            z[i].append( z_i )
    else:
        z.append(z_) 

def object_to_list(z):
    if isinstance(z, dict):
        for k, v in z.items():
            z[k] = object_to_list(v)
        return z
    elif isinstance(z,list):
        return [ [z_i] for z_i in z]
    else:
        return [z]

def stack_all(z_list):
    if isinstance(z_list, dict):
        for k, v in z_list.items():
            z_list[k] = stack_all(v)
    elif isinstance(z_list[0], list):
        for i, v in enumerate(z_list):
            z_list[i] = stack_all(v)
    elif isinstance(z_list, list):
        z_list = np.concatenate(z_list, axis = 0)
    else:
        print(type(z_list))
        pass
    return z_list

def get_max_value(target_dtype):
    try:
        return  np.finfo(target_dtype).max
    except:
        return  np.iinfo(target_dtype).max
    
def numpy_to_torch(arr, device="cpu"):
    return torch.from_numpy(arr).to(device)

def move_numpys(*arrays, device="cpu"):
    return tuple(numpy_to_torch(arr, device=device) for arr in arrays)

def get_metricscollection(DIS_NO_DATA_VALUE, N_LABELS, confusion_metrics=[], tolerance_flag=False, device="cpu"):
    if tolerance_flag:
        return MetricCollection([
            BinaryStatsOverallWindow(no_data_val= DIS_NO_DATA_VALUE, metrics_calculate=confusion_metrics)]).to(device)
    else:
        return MetricCollection([ 
            ConfusionStatsOverall(num_classes=N_LABELS, no_data_val= DIS_NO_DATA_VALUE, metrics_calculate=confusion_metrics).to(device),
            ])

def add_summary_metrics(cube):
    for m in ["metrics", "metricsA"]:
        if m in cube:
            cube[f"{m}_yearly"] = cube[f"{m}"].mean(dim="cube", skipna=True)
            cube[f"{m}_cube"] = cube[f"{m}"].mean(dim="year", skipna=True)
            cube[f"{m}_overall"] = cube[f"{m}"].mean(dim=["cube", "year"], skipna=True)
        
            cube[f"{m}_yearly_std"] = cube[f"{m}"].std(dim="cube", skipna=True)
            cube[f"{m}_cube_std"] = cube[f"{m}"].std(dim="year", skipna=True)
            cube[f"{m}_overall_std"] = cube[f"{m}"].std(dim=["cube", "year"], skipna=True)
        
        if "pixel"+m in cube:
            cube[f"pixel{m}_overall"] = cube[f"pixel{m}"].mean(dim=["year"], skipna=True)
            cube[f"pixel{m}_overall_std"] = cube[f"pixel{m}"].std(dim=["year"], skipna=True)

    for var in ["metrics_yearly", "metrics_cube", "metrics_overall", "pixelmetrics_overall", 
                "metricsA_yearly", "metricsA_cube", "metricsA_overall", "pixelmetricsA_overall"]:
        if var not in cube:
            continue
        
        for coord in cube[var].coords:
            cube[var].coords[coord].encoding = cube[var].coords[coord].encoding.copy()
            cube[f"{var}_std"].coords[coord].encoding = cube[var].coords[coord].encoding.copy()
        cube[var].encoding = cube[var.split("_")[0]].encoding.copy()
        cube[f"{var}_std"].encoding = cube[var.split("_")[0]].encoding.copy()
        
def store_nc_compressed(cube_data, stored_name, compressor):
    for var in cube_data.data_vars:
        cube_data[var].encoding["compressor"] = compressor
    os.makedirs(os.path.dirname(stored_name), exist_ok=True)
    print("Storing into:", os.path.dirname(stored_name))
    cube_data.to_netcdf(stored_name, compute=True, engine='netcdf4')


def ece_binary(y_true, y_prob, n_bins=15):
    valid = ~torch.isnan(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    bins = torch.linspace(0, 1, n_bins + 1, device=y_prob.device)
    bin_ids = torch.bucketize(y_prob, bins[1:-1])

    ece = torch.tensor(0., device=y_prob.device)
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue

        conf = y_prob[mask].mean()
        acc = y_true[mask].float().mean()
        ece += mask.float().mean() * (acc - conf).abs()

    return ece


def ece_binary_batch(y_true, y_prob, n_bins=15):
    """
    Compute binary ECE for all years simultaneously.

    Parameters
    ----------
    y_true : (n_years, N)
        Binary labels (0/1)

    y_prob : (n_years, N)
        Predicted probabilities (may contain NaNs)

    Returns
    -------
    ece : (n_years,)
    """

    device = y_prob.device
    n_years, N = y_prob.shape
    valid = ~torch.isnan(y_prob)

    # assign each probability to a bin
    bins = torch.linspace(0, 1, n_bins + 1, device=device)
    bin_ids = torch.bucketize(y_prob.nan_to_num(0.), bins[1:-1])

    # build indices for scatter_add
    year_ids = (
        torch.arange(n_years, device=device)
        .unsqueeze(1)
        .expand_as(bin_ids)
    )

    flat_index = (year_ids * n_bins + bin_ids).flatten()

    # allocate accumulators
    counts = torch.zeros(n_years * n_bins, device=device)
    prob_sums = torch.zeros_like(counts)
    true_sums = torch.zeros_like(counts)

    valid_flat = valid.flatten()

    counts.scatter_add_(
        0,
        flat_index[valid_flat],
        torch.ones_like(flat_index[valid_flat], dtype=torch.float),
    )

    prob_sums.scatter_add_(
        0,
        flat_index[valid_flat],
        y_prob.flatten()[valid_flat],
    )

    true_sums.scatter_add_(
        0,
        flat_index[valid_flat],
        y_true.float().flatten()[valid_flat],
    )

    counts = counts.view(n_years, n_bins)
    prob_sums = prob_sums.view(n_years, n_bins)
    true_sums = true_sums.view(n_years, n_bins)
    nonzero = counts > 0

    mean_conf = torch.zeros_like(prob_sums)
    mean_acc = torch.zeros_like(true_sums)
    mean_conf[nonzero] = prob_sums[nonzero] / counts[nonzero]
    mean_acc[nonzero] = true_sums[nonzero] / counts[nonzero]

    total = counts.sum(dim=1, keepdim=True)
    weights = counts / total.clamp_min(1)

    ece = (weights * (mean_acc - mean_conf).abs()).sum(dim=1)
    return ece