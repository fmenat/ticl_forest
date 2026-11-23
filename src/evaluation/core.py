import torch
from tqdm import tqdm
import numpy as np
import xarray as xr

from .utils import detach_all, object_to_list, collate_all_list, stack_all, get_max_value, ece_binary, ece_binary_batch
from .globalmetrics import temporal_distance_to_prediction,relative_area_error

def calculate_metrics(aggregation, pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, metrics, probas=True, ref_tolerance_pos= None):
    if probas:
        pred_class = pred.argmax(axis=2)
    else:
        pred_class = pred
    if ref_tolerance_pos is not None:
        ref_tolerance_pos = ref_tolerance_pos.to(pred.device)

    metrics_return = {}
    if aggregation == "overall":
        metrics.reset()

        print("Calculating overall metrics...")
        if ref_tolerance_pos is not None:
            metrics.update(pred_class, ref, mask, ref_tolerance_pos)
        else:
            metrics.update(pred_class, ref, mask)
        values_ = metrics.compute()
        
        for v in values_:
            metrics_return[v] = values_[v].detach()
        
    else:
        for year in tqdm(range(pred_class.shape[0]), desc=f"Calculating per-year metrics in aggregation {aggregation}", total=pred_class.shape[0]):
            if aggregation == "per_year":
                metrics.reset()

                if ref_tolerance_pos is not None:
                    metrics.update(pred_class[year], ref[year], mask[year], ref_tolerance_pos[year])
                else:
                    metrics.update(pred_class[year], ref[year], mask[year])
                values_ = metrics.compute()

                for v in values_:
                    metrics_return.setdefault(v, []).append(values_[v].detach())

            if aggregation == "per_year_sample":     
                for i in range(pred_class.shape[1]):
                    metrics.reset()

                    if ref_tolerance_pos is not None:
                        metrics.update(pred_class[year, [i]], ref[year, [i]], mask[year, [i]], ref_tolerance_pos[year, [i]])
                    else:
                        metrics.update(pred_class[year, [i]], ref[year, [i]], mask[year, [i]])
                    values_ = metrics.compute()

                    for v in values_:
                        metrics_return.setdefault(v, []).append(values_[v].detach())
    
    for m in metrics_return:
        metrics_return[m] = torch.Tensor(metrics_return[m]).cpu().detach()
    return metrics_return

def calculate_further_metrics(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, probas=True, no_data_value=-1, max_distance=15.0):
    if probas:
        pred_class = pred.argmax(axis=2)
    else:
        pred_class = pred
    valid_mask = mask*(ref != no_data_value)
    valid_pred = pred_class*valid_mask
    valid_ref = ref*valid_mask
    
    metrics_return = {}

    per_pixel_distances, per_pixel_r2 = temporal_distance_to_prediction(valid_pred, valid_ref, max_distance=max_distance)
    metrics_return["distance"] = {
        "overall": per_pixel_distances
    }
    metrics_return["distance_r2"] = {
        "overall": per_pixel_r2
    }

    area_error_yearly, area_error, r2_area = relative_area_error(valid_pred, valid_ref, maximum_error_ratio=max_distance)
    metrics_return["area"] = {
        "per_year": area_error_yearly,
        "overall": area_error
    }
    metrics_return["area_r2"] = {
        "overall": r2_area
    }

    return metrics_return


def predict_numpy(model, dataloader, device, include_mask=True):    
    model.to(device)
    model.eval() 
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Predicting"):
            views_dict, refs = model.prepare_batch(batch)
            
            outputs_ = model(views_dict)
            outputdata_ = { "main": model.apply_softmax(outputs_["logits"], dim=1), }  
            
            if include_mask:
                masks_ = model.mask_calculation(reference_labels=refs)
                outputdata_["valid_mask"] = masks_["valid_mask"]
                outputdata_["forest_mask"] = masks_["forest"]            
            outputdata_ = detach_all(outputdata_)

            outputdata_["cube_name"] = np.asarray(batch["cube_name"])
            outputdata_["year"] = batch["year"]

            if batch_idx == 0:
                outputdata = object_to_list(outputdata_) #to start append values
            else:
                collate_all_list(outputdata, outputdata_) #add to list 
    #model.train()    
    return stack_all(outputdata) #stack


def reshape_predictions(output_data, years_predict):
    for k in output_data.keys():
        if k in ["cube_name", "year", "idx"]:
            output_data[k] = output_data[k].reshape(years_predict, -1)
        else:
            output_data[k] = output_data[k].reshape(years_predict, -1, *output_data[k].shape[1:])


def create_xarray_assets(assets_dict, partition="test", probas=True):
    encoded_vars = []
    cube = xr.Dataset()
    if partition == "test": 
        cube["prediction"] = xr.DataArray(assets_dict["main"].argmax(axis=2) if probas else assets_dict["main"], 
            dims=["year", "cube", "y", "x",],
            coords={
                "year": assets_dict["year"][:,0],
                "cube": assets_dict["cube_name"][0],
                }
        ).astype(np.uint8)
        cube["prediction"].attrs = {
            "long_name": "Predicted disturbance classes",
            "flag_meanings": ['non_disturbance', 'disturbance'],
            "flag_values": [0, 1]
            }

        if probas:
            cube["prediction_scores"] = xr.DataArray(assets_dict["main"][:,:,1], 
                dims=["year", "cube", "y", "x"],
                coords={
                    "year": assets_dict["year"][:,0],
                    "cube": assets_dict["cube_name"][0]
                }
            ).astype(np.float32)
            cube["prediction_scores"].attrs = {
                "long_name": "Predicted disturbance (positive) probabilities (scores)",
                "flag_meanings": ["Disturbance probability"],
            }
            encoded_vars.append("prediction_scores")

        if "valid_mask" in assets_dict:
            cube["model_mask"] = xr.DataArray(assets_dict["valid_mask"], 
                dims=["year", "cube", "y", "x"],
                coords={
                    "year": assets_dict["year"][:,0],
                    "cube": assets_dict["cube_name"][0]
                    }
            ).astype(np.uint8)
            cube["model_mask"].attrs = {
                "long_name": "Valid data mask used for model training",
                "flag_meanings": ['non_valid', 'valid'],
                "flag_values": [0, 1]
            }

    for var in ["pixelmetrics", "pixelmetricsF", "pixelmetricsFs", "pixelmetricsFr"]:
        if var not in assets_dict:
            continue
        cube[var] = xr.DataArray(np.stack(list(assets_dict[var].values())),
            dims=["metric_name"],
            coords={
                "metric_name": list(assets_dict[var].keys()),
                }
        ).astype(np.float32)
        encoded_vars.append(var)
        for var_asset in assets_dict:
            if f"{var}_" in var_asset and var_asset != var:
                print("Adding asset", var_asset, "for metrics", var)
                cube[var_asset] = xr.DataArray(np.stack(list(assets_dict[var_asset].values())),
                    dims=["metric_name"],
                    coords={
                        "metric_name": list(assets_dict[var_asset].keys()),
                        }
                ).astype(np.float32)
                encoded_vars.append(var_asset)

    for var in ["yearmetrics", "yearmetricsF", "yearmetricsFs", "yearmetricsFr"]:
        if var not in assets_dict:
            continue
        cube[var] = xr.DataArray(np.stack(list(assets_dict[var].values())),
            dims=["metric_name", "year"],
            coords={
                "metric_name": list(assets_dict[var].keys()),
                "year": assets_dict["year"][:,0],
                }
        ).astype(np.float32)
        encoded_vars.append(var)
        for var_asset in assets_dict:
            if f"{var}_" in var_asset and var_asset != var:
                cube[var_asset] = xr.DataArray(np.stack(list(assets_dict[var_asset].values())),
                    dims=["metric_name", "year"],
                    coords={
                        "metric_name": list(assets_dict[var_asset].keys()),
                        "year": assets_dict["year"][:,0],
                        }
                ).astype(np.float32)
                encoded_vars.append(var_asset)

    for var in ["yearuncertaintF", "yearuncertaintFs", "yearuncertaintFr"]:
        if var not in assets_dict:
            continue
        cube[var] = xr.DataArray(np.stack(list(assets_dict[var].values())),
            dims=["unc_metric_name", "year"],
            coords={
                "unc_metric_name": list(assets_dict[var].keys()),
                "year": assets_dict["year"][:,0],
                }
        ).astype(np.float32)
        encoded_vars.append(var)

    for var in ["samplemetrics", "samplemetricsF", "samplemetricsFs", "samplemetricsFr"]:
        if var not in assets_dict:
            continue
        cube[var] = xr.DataArray(np.stack([v.reshape(assets_dict["main"].shape[0], -1) for v in assets_dict[var].values()]),
            dims=["metric_name", "year", "cube"],
            coords={
                "metric_name": list(assets_dict[var].keys()),
                "year": assets_dict["year"][:,0],
                "cube": assets_dict["cube_name"][0],
                }
        ).astype(np.float32)
        encoded_vars.append(var)

    for var in encoded_vars:
        atrs_or = cube[var].attrs.copy()
        enc_or = cube[var].encoding.copy()

        target_dtype = np.uint16
        max_val = get_max_value(target_dtype)
        new_max_val = (max_val-1)/max_val
        cube[var] = cube[var].clip(max=new_max_val)
        cube[var].attrs = atrs_or
        cube[var].encoding = enc_or
        cube[var].encoding["dtype"] = target_dtype.__name__
        cube[var].encoding["scale_factor"] = 1.0 / max_val
        cube[var].encoding["_FillValue"] = max_val

    return cube



def calculate_uncertainties(pos_probas, reference, forest_mask, no_data_value=-1, n_bins=10):
    """
    Calculate uncertainty metrics for the given predictions and references.
    * pos_probas: (n_years, n_cubes, n_classes, y, x) tensor of predicted probabilities for the positive class (disturbance).
    * reference: (n_years, n_cubes, y, x) tensor of reference labels (0 for non-disturbance, 1 for disturbance).
    * forest_mask: (n_years, n_cubes, y, x) tensor indicating valid forest pixels (1 for valid, 0 for non-forest or no data).
    * no_data_value: The value representing no data in the reference tensor.
    """
    updated_mask = forest_mask * (reference != no_data_value) #valid forest pixels
    probas_masked_forest = pos_probas.masked_fill(updated_mask.unsqueeze(2)!= 1, torch.nan)
    
    unc_metrics = {}
    for c in range(pos_probas.shape[2]):
        c_probas = probas_masked_forest[:,:, c]
        c_reference = (reference==c)*1

        unc_metrics[f"{c}_meanprob"] = torch.nanmean(c_probas, dim=(1,2,3))
        unc_metrics[f"{c}_maxprob"] = torch.nanmean(torch.amax(c_probas, dim=(2,3)), dim=1) #spatial max -- mean over cubes
        unc_metrics[f"{c}_ece"] = ece_binary_batch(c_reference.flatten(1), c_probas.flatten(1), n_bins=n_bins)
        
    log_probas = torch.log(probas_masked_forest.clamp_min(1e-8))
    unc_metrics[f"entropy"] = - torch.nanmean( torch.sum(probas_masked_forest*log_probas, dim=2), dim=(1,2,3))
    
    return unc_metrics