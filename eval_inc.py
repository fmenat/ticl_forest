import pandas as pd
import xarray as xr
import numpy as np
import gc, warnings, copy, time, yaml, argparse, multiprocessing, os, sys
import torch
from torch.utils.data import DataLoader

from src.training.data_setup import get_train_val_test_cubes
from src.dataset.cubeloader import CubeLoader
from src.dataset.batchloader import YearGroupedBatchSampler
from src.dataset.transforms import get_numerical_transform, get_categorical_transform
from src.dataset.utils import sel_period_xarray
from src.model.simple import UNetConvLSTM, UNetSimple,EncoderDecoder
from src.evaluation.core import calculate_metrics, predict_numpy,reshape_predictions,create_xarray_assets, calculate_further_metrics, calculate_uncertainties
from src.evaluation.utils import get_metricscollection, store_nc_compressed, compressor, move_numpys
from src.incremental.modular import activate_step, add_step_module, update_modular_incremental_model


#filter all kinds of warnings for cleaner output
warnings.filterwarnings("ignore")


def evaluate_model(model, eval_cube, LABEL_NAME="disturbances", cubeloader_args={}, BATCH_SIZE=32, device=0, 
                   confusion_metrics=[], dataloader_args={}
                   ):
    ## CREATE DATALOADERS
    eval_data = CubeLoader(eval_cube, validation=True, **cubeloader_args)
    eval_yearlysampler = YearGroupedBatchSampler(eval_data, batch_size=BATCH_SIZE) #just to have predictions grouped by year
    eval_dataloader = DataLoader(eval_data, batch_sampler=eval_yearlysampler, **dataloader_args)
    
    ## FORWARD OVER MODEL
    output_data = predict_numpy(model, eval_dataloader, device="cpu" if device ==-1 else f"cuda:{device}", include_mask=False)
    reshape_predictions(output_data, len(eval_data.year_to_indices))
    print("Predictions shape after reshape:", {k: v.shape for k, v in output_data.items()})

    #DEFINE LABELS and NO DATA VALUES
    reference_ = eval_cube[LABEL_NAME].sel(cube=output_data["cube_name"][0]).isel(year=slice(-len(eval_data.year_to_indices), None)).transpose("year", "cube", "y", "x").load()
    DIS_NO_DATA_VALUE = eval_cube[LABEL_NAME].attrs["no_data_value"]
    N_LABELS = len(eval_cube[LABEL_NAME].attrs["flag_values"]) -1
    
    pred_class = output_data["main"].argmax(axis=2)
    pos_mask = np.ones(pred_class.shape, dtype=bool)
    pred_class_t, reference_t, pos_mask_t = move_numpys(pred_class, reference_.values, pos_mask) #, device=f"cuda:{device}")
    
    ## DEFINE and Calculate METRICS
    print("Calculating metrics...")
    metrics = get_metricscollection(DIS_NO_DATA_VALUE, N_LABELS, confusion_metrics=confusion_metrics, tolerance_flag=False) #, device=f"cuda:{device}")
    
    output_data["pixelmetricsF"] = calculate_metrics("overall", pred_class_t, reference_t, pos_mask_t, metrics, probas=False)
    output_data["yearmetricsF"] = calculate_metrics("per_year", pred_class_t, reference_t, pos_mask_t, metrics, probas=False)
    
    if "disturbance_agent" in eval_cube:
        agents_info = zip(eval_cube["disturbance_agent"].flag_values[:-1], eval_cube["disturbance_agent"].flag_meanings[:-1])
        agents_data = eval_cube["disturbance_agent"].isel(year=slice(-len(eval_data.year_to_indices), None)).transpose("year", "cube", "y", "x").load()
        for ag_val, ag_name in agents_info:
            print("Calculating metrics for agent:", ag_name)
            agent_bin  = move_numpys(agents_data.values == ag_val)[0]
            output_data["pixelmetricsF_" + ag_name] = calculate_metrics("overall", pred_class_t, reference_t, agent_bin, metrics, probas=False)
            output_data["yearmetricsF_" + ag_name] = calculate_metrics("per_year", pred_class_t, reference_t, agent_bin, metrics, probas=False)    

    #calculate uncertainty per year averaged spatially and for all cubes
    probas_ = move_numpys(output_data["main"])[0]
    output_data["yearuncertaintF"] = calculate_uncertainties(probas_, reference_t, pos_mask_t, no_data_value=DIS_NO_DATA_VALUE) #per year
    
    print("Calculating further metrics...")
    extra = calculate_further_metrics(pred_class_t, reference_t, pos_mask_t, probas=False, no_data_value=DIS_NO_DATA_VALUE)
    for m in extra:
        output_data["pixelmetricsF"][m] = extra[m]["overall"]
        if m == "area": #only area has per year metric
            output_data["yearmetricsF"][m] = extra[m]["per_year"]
    
    #create xarray dataset with predictions and metrics
    store_preds = create_xarray_assets(output_data, partition="test", probas=True)    
    return store_preds


def main_run(config_file, pre_load_data=None):
    DEVICE = config_file.get("device")[0]
    
    DATA_FOLDER = config_file["data_folder"]
    MODEL_PATH = config_file["model_folder"]
    STORAGE_PATH = config_file.get("storage_path", MODEL_PATH)
    DATASET_NAME = config_file["dataset_name"]
    INCR_MODEL_NAME = config_file["incr_model"]
    CONFIG_EXP_NAME = config_file["config_exp_name"]
    VERSION = config_file.get("version", 0)
    
    num_workers = config_file.get("num_workers")  if "num_workers" in config_file else min(4, multiprocessing.cpu_count() // 2)
    dataloader_args = {
        "pin_memory": True,
        "num_workers": num_workers,
        "persistent_workers": num_workers>0,
        "shuffle": False,
    }

    #DATASET CONFIGURATION
    dataset_config = config_file.get("dataset_config", {})
    VAL_TYPE = dataset_config.get("val_type", "temporal") #temporal or spatial, whether to use the last year of the training data as validation (temporal) or to randomly sample cubes from the training years for validation (spatial)
    train_val_split_args = dataset_config.get("train_val_split_args", {})
    LAG_YEAR = dataset_config.get("lag_year", -1)
    LABEL_NAME = dataset_config.get("label_name", "disturbances")    
    BATCH_SIZE = dataset_config.get("batch_size", 32)
    METRICS_EVAL = dataset_config.get("metrics_eval", ["f1", "pre", "rec", "iou", "ior", "f2", "ti", "yj", "count", "counthat"])    
    START_TIME = dataset_config.get("start_time", None)
    END_TIME = dataset_config.get("end_time", None)
    
    #### LOAD DATA
    if pre_load_data is None:
        data_load = xr.open_zarr(f"{DATA_FOLDER}/{DATASET_NAME}.zarr", mask_and_scale=True, consolidated=True)
        print("Data loaded from:", f"{DATA_FOLDER}/{DATASET_NAME}.zarr", "with chunking:", data_load["disturbances"].encoding["chunks"])
    else:
        print("Data pre-loaded in memory.")
        data_load = pre_load_data
    if START_TIME is not None or END_TIME is not None:
        data_load = sel_period_xarray(data_load, start=START_TIME, end=END_TIME)
        print(f"Data selected for years {data_load.year.values[0]} to {data_load.year.values[-1]}")
        if "time" in data_load:
            print(f"Data selected for time range {data_load.time.values[0]} to {data_load.time.values[-1 ]}")
    
    ### LOAD MODEL
    df = pd.read_csv(f"{MODEL_PATH}/experiments_summary_{INCR_MODEL_NAME}.csv") if INCR_MODEL_NAME != "None" else pd.read_csv(f"{MODEL_PATH}/experiments_summary.csv")
    model_info = df.set_index(["exp_name", "version", "step"]).loc[(CONFIG_EXP_NAME, VERSION, slice(None)), ["model_name", "best_model_path"]]    
    steps_available = model_info.index.get_level_values("step").unique().tolist()
    print(f"Steps/years available for evaluation: {steps_available}")
    for i, (step, group) in enumerate(model_info.groupby(level="step")):
        if step  == END_TIME and VAL_TYPE == "temporal":
            print(f"Reached end time {END_TIME} for temporal validation, stopping evaluation loop.")
            break
        model_path = group['best_model_path'].values[0].replace("_emissions","").split("model_")[0]
        model_path = model_path+[file for file in os.listdir(model_path) if file.endswith(".ckpt")][0]
        path_model_storage = model_path.split("/model_")[0]
        config_used = yaml.load(open(f"{path_model_storage}/config.yaml", 'r'), Loader=yaml.SafeLoader)
        
        if INCR_MODEL_NAME == "modular" :
            checkpoint = torch.load(model_path)
            if i ==0:
                load_model = eval(model_info['model_name'].values[0]).load_from_checkpoint(model_path, strict=False)
                update_modular_incremental_model(load_model, module_factory=config_used["incremental_strategy"]["method_args"]["step_module"])
            add_step_module(load_model, str(step))
            load_model.load_state_dict(checkpoint['state_dict']) #reload with step activated
            activate_step(load_model, str(step))
        else:
            load_model = eval(model_info['model_name'].values[0]).load_from_checkpoint(model_path)
        load_model.eval()
        print(f"Evaluating model for year/step {step}, Model loaded from: {path_model_storage}")
        
        ### LOAD DATA -- YEARLY BASED
        LAG_YEAR = config_used["dataset_config"]["lag_year"]
        trained_data, _, test_cube = get_train_val_test_cubes(data_load, step+1, data_load.year, LAG_YEAR, VAL_TYPE, **train_val_split_args)
        if len(test_cube["year"]) <= 1:
            print(f">>>>>>>> No minimum 2 time steps of test data available for step {step}. Skipping evaluation for this step.")
            continue
        print("training", trained_data["year"].values, ", with samples ", len(trained_data.cube), ", testing", test_cube["year"].values, "with samples ", len(test_cube.cube))
        
        dic_transform = {}
        for v in config_used["dataset_config"]["scaler_per_sensor"]:
            if config_used["dataset_config"]["scaler_per_sensor"][v] == "categorical":
                dic_transform[v] = get_categorical_transform(trained_data, v)
            else:
                dic_transform[v] = get_numerical_transform(trained_data, v, config_used["dataset_config"]["scaler_per_sensor"][v], mask_value=0)
        cubeloader_args = {"lag_years": LAG_YEAR, "transforms": dic_transform, "load_memory": config_file.get("load_memory", False), "years_validate": len(test_cube.year)}
        if "extra_bands" in config_used["dataset_config"]:
            cubeloader_args["extra_bands"] = config_used["dataset_config"]["extra_bands"]
        assert cubeloader_args["load_memory"] or dataloader_args["num_workers"] == 0, "Load_memory is False and num_workers is >0, which is not recommended for performance in dask loading"
        del trained_data

        evaluate_args = {
            "LABEL_NAME": LABEL_NAME, "BATCH_SIZE": BATCH_SIZE, "device": DEVICE, "cubeloader_args": cubeloader_args, "dataloader_args": dataloader_args, "confusion_metrics": METRICS_EVAL,
        }
        
        ### EVALUATE MODEL -- YEARLY BASED
        years_ = test_cube.year.values[1:].astype(str)
        df_summ = pd.DataFrame(columns=years_)

        print(f"Evaluating on the following years {test_cube.year.values} with lag of {LAG_YEAR} years")
        store_preds = evaluate_model(load_model, test_cube, **evaluate_args)
        
        path_model_storage = path_model_storage.replace(MODEL_PATH, STORAGE_PATH)
        store_nc_compressed(store_preds, f"{path_model_storage}/test_results.nc", compressor=compressor)
        for m in store_preds.metric_name.values:
            df_summ.loc[m, years_] = store_preds["yearmetricsF"].sel(metric_name=m).values
            df_summ.loc[m, "mean"] = store_preds["yearmetricsF"].sel(metric_name=m).mean().values
            df_summ.loc[m, "overall"] = store_preds["pixelmetricsF"].sel(metric_name=m).values
            
        print("Evaluation results:")
        print(df_summ.loc[sorted(df_summ.index)].round(3).to_markdown()) 

        print("Saving evaluation summary to:", f"{path_model_storage}/eval_summary.csv")
        df_summ.to_csv(f"{path_model_storage}/eval_summary.csv")
        
        data_load.close()
        test_cube.close()
        del test_cube, store_preds
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
    data_load.close()
        
    
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--settings_file",
        "-s",
        action="store",
        dest="settings_file",
        required=True,
        type=str,
        help="path of the settings file",
    )
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)

    pre_loaded_data = None
    if config_file.get("pre_load_data", False):
        DATA_FOLDER = config_file["data_folder"]
        DATASET_NAME = config_file["dataset_name"]
        pre_loaded_data = xr.open_zarr(f"{DATA_FOLDER}/{DATASET_NAME}.zarr", mask_and_scale=True, consolidated=True).load()
        print("Data pre-loaded from:", f"{DATA_FOLDER}/{DATASET_NAME}.zarr", "with chunking:", pre_loaded_data["disturbances"].encoding["chunks"])    

    for exp in config_file["exp_name"]:
        config_file_tmp = config_file.copy()
        config_file_tmp["incr_model"] = str(exp[0])
        config_file_tmp["config_exp_name"] = str(exp[1])
        config_file_tmp["version"] = int(exp[2])
        print(f"Evaluating model with config {exp} ")
        main_run(config_file_tmp, pre_loaded_data)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()