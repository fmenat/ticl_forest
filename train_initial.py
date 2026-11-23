import pandas as pd
import xarray as xr
import numpy as np
import torch, gc, yaml, argparse, multiprocessing, copy, sys, os
from codecarbon import EmissionsTracker
from contextlib import nullcontext

from src.dataset.prepare import restore_dataset_attrs
from src.dataset.transforms import get_numerical_transform, get_categorical_transform
from src.dataset.utils import sel_period_xarray, concat_datasets
from src.training.pl_train import train_model_year
from src.training.data_setup import get_train_val_test_cubes, get_dataloaders, compute_class_weights
from src.training.names import COLUMNS_DF
from src.model.simple import UNetConvLSTM, UNetSimple, EncoderDecoder



def main_run(config_file, early_stopping=True, store_checkpoints=True):
    DATA_FOLDER = config_file["data_folder"]
    STORAGE_PATH = config_file["storage_path"]
    DATASET_NAME = config_file.get("dataset_name", None)
    DEVICE = config_file.get("device")

    EXP_DATA = config_file.get("exp_data", {})
    EXP_NAME = EXP_DATA.get("exp_name")
    VERSION = EXP_DATA.get("version", 0)
    MODEL_PATH = EXP_DATA.get("model_folder")
    if len(EXP_DATA)>0:
        full_exp_name = f"{EXP_NAME}_v{VERSION}"
    
    model_args = config_file.get("model", {})
    MODEL_TYPE = model_args.get("type") 
    encoder_args = model_args.get("encoder_args", {})
    decoder_args = model_args.get("decoder_args", {})
    if len(model_args)>0:
        full_exp_name = f"{MODEL_TYPE}"

    num_workers = config_file.get("num_workers")  if "num_workers" in config_file else min(4, multiprocessing.cpu_count() // 2)
    dataloader_args = {
        "pin_memory": True ,
        "num_workers": num_workers,
        "persistent_workers": num_workers>0, 
        "shuffle": False,
    }

    POST_NAME = config_file.get("post_name", "")

    #DATASET CONFIGURATION
    dataset_config = config_file.get("dataset_config", {})
    LAG_YEAR = dataset_config.get("lag_year", 3)
    scaler_per_sensor = dataset_config.get("scaler_per_sensor")
    extra_bands = dataset_config.get("extra_bands", {})
    START_FROM_STEP = dataset_config.get("start_from_step", None)
    VAL_TYPE = dataset_config.get("val_type", "temporal") #temporal or spatial, whether to use the last year of the training data as validation (temporal) or to randomly sample cubes from the training years for validation (spatial)
    train_val_split_args = dataset_config.get("train_val_split_args", {})
    LABEL_NAME = dataset_config.get("label_name", "disturbances")
    START_TIME = dataset_config.get("start_time", None)
    END_TIME = dataset_config.get("end_time", None)

    #TRAINING-OPTIMIZATION CONFIGURATION
    training_args = config_file.get("training")
    BATCH_SIZE = training_args.get("batch_size", 32)
    EPOCHS = training_args.get("epochs", 5)
    optim_args = training_args.get("optimization", {})
    optim_args["BS"] = BATCH_SIZE
    stopping_args = training_args.get("stopping", {})
    metric_stopval = stopping_args.get("metric_stopval", "ma_f1")
    delta_stopval = stopping_args.get("min_delta", 0.01)
    pat_stopval = stopping_args.get("patience", 3)
    mode_stopval = stopping_args.get("mode_stopval", "max")
    earlystopping_args = {"metric_stopval": metric_stopval, "mode_stopval": mode_stopval, "delta_stopval": delta_stopval, "pat_stopval": pat_stopval}
    NORM_PREV_FLAG = training_args.get("norm_prev_flag", False)
    augmentation_params = training_args.get("augmentation_params",{})
    UPDATE_LOSS_WEIGHTS = training_args.get("update_loss_weights", True)

    loss_args = config_file.get("loss_args", {})

    #### LOAD DATA
    if type(DATASET_NAME) == list:
        data_load = concat_datasets(DATASET_NAME, DATA_FOLDER, concat_dim="cube")
    else:
        print("Loading data from:", DATA_FOLDER,DATASET_NAME, "...")
        data_load = xr.open_zarr(f"{DATA_FOLDER}/{DATASET_NAME}.zarr", mask_and_scale=True, consolidated=True)
    data_load = restore_dataset_attrs(data_load)
    print("Chunking configuration", data_load["disturbances"].encoding["chunks"])
    if START_TIME is not None or END_TIME is not None:
        data_load = sel_period_xarray(data_load, start=START_TIME, end=END_TIME)
        print(f"Data selected for years {data_load.year.values[0]} to {data_load.year.values[-1]}")
        if "time" in data_load:
            print(f"Data selected for time range {data_load.time.values[0]} to {data_load.time.values[-1 ]}")
        
    STEP_INCREMENTAL = data_load.year.values
    if START_FROM_STEP is None:
        START_FROM_STEP = STEP_INCREMENTAL[0] + (LAG_YEAR-1)
    else:
        assert START_FROM_STEP > STEP_INCREMENTAL[0], f"start_from_STEP {START_FROM_STEP} not in dataset years {STEP_INCREMENTAL}"
    DIS_NO_DATA_VALUE = data_load[LABEL_NAME].attrs["no_data_value"]
    N_LABELS = (len(data_load[LABEL_NAME].attrs["flag_values"])  -1) if "flag_values" in data_load[LABEL_NAME].attrs else 2


    #### LOAD MODEL
    if len(EXP_DATA) > 0: #load data from an old experiment
        df = pd.read_csv(f"{MODEL_PATH}/experiments_summary.csv")
        model_info = df.set_index(["exp_name", "version"]).loc[[(EXP_NAME, VERSION)], ["model_name", "best_model_path"]]
        new_model_path = model_info['best_model_path'].values[0].replace("/home/mena/results", "/misc/glm1/person/mena/results")
        path_model_storage = new_model_path.split("/model_")[0]
        load_model = eval(model_info['model_name'].values[0]).load_from_checkpoint(new_model_path)    
        print(f"Model loaded from: {path_model_storage}")
        
        load_model.set_loss_args(loss_args["main_args"], loss_args.get("macro_args", {}))
        load_model.hparams.update(**optim_args)

    elif len(model_args)>0: #create model from configuration file without loading old experiment data
        if MODEL_TYPE == "unet_convlstm":
            for sensor in encoder_args:
                if "n_times" not in encoder_args[sensor]:
                    encoder_args[sensor]["n_times"] = LAG_YEAR
            decoder_args["n_classes"] = N_LABELS
            decoder_args["label_name"] = LABEL_NAME
            load_model = UNetConvLSTM(encoder_args, decoder_args, no_data_value=DIS_NO_DATA_VALUE, optim_args=optim_args, loss_args=loss_args["main_args"], macro_loss_args=loss_args.get("macro_args", {}))

        elif MODEL_TYPE == "unet_simple":
            decoder_args["n_classes"] = N_LABELS
            decoder_args["label_name"] = LABEL_NAME
            load_model = UNetSimple(encoder_args, decoder_args, no_data_value=DIS_NO_DATA_VALUE, optim_args=optim_args, loss_args=loss_args["main_args"], macro_loss_args=loss_args.get("macro_args", {}))

        elif MODEL_TYPE == "encdec":
            decoder_args["n_classes"] = N_LABELS
            decoder_args["label_name"] = LABEL_NAME
            load_model = EncoderDecoder(encoder_args, decoder_args, no_data_value=DIS_NO_DATA_VALUE, optim_args=optim_args, loss_args=loss_args["main_args"], macro_loss_args=loss_args.get("macro_args", {}))
        
        
    ### DATA PREPRATION
    if NORM_PREV_FLAG and len(EXP_DATA)>0: 
        ### LOAD CONFIG FILES AND APPLY THEM
        config_used = yaml.load(open(f"{path_model_storage}/config.yaml", 'r'), Loader=yaml.SafeLoader)
        new_datapath = config_used['data_folder'].replace("/misc/glm3/fonda/", "/misc/glm1/person/mena/fonda/")
        cube_stats = xr.open_zarr(f"{new_datapath}/{config_used['dataset_name']}.zarr", mask_and_scale=True, consolidated=True)
        scaler_per_sensor = config_used["dataset_config"]["scaler_per_sensor"]
    else:
        cube_stats = data_load

    dic_transform = {}
    for v in scaler_per_sensor:
        if scaler_per_sensor[v] == "categorical":
            dic_transform[v] = get_categorical_transform(cube_stats, v)
        else:
            dic_transform[v] = get_numerical_transform(cube_stats, v, scaler_per_sensor[v], mask_value=0)
    cubeloader_args = {
        "lag_years": LAG_YEAR, 
        "transforms": dic_transform, 
        "extra_bands": extra_bands,
        "load_memory": config_file.get("load_memory", False), 
        }
    
    ### MAIN PROCESS
    try:
        df_initial = pd.read_csv(f"{STORAGE_PATH}/experiments_summary.csv")
    except FileNotFoundError:
        df_initial = pd.DataFrame(columns=COLUMNS_DF)


    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training initial step on step {START_FROM_STEP}... ") 

    ### GET TRAINING AND VALIDATION CUBES AND DATALOADERS
    train_cube, val_cube, _ = get_train_val_test_cubes(data_load, START_FROM_STEP+1, STEP_INCREMENTAL, LAG_YEAR, VAL_TYPE, first_flag=True, **train_val_split_args)
    print("training steps/year", train_cube["year"].values, "validation", val_cube["year"].values)
    print("training samples", len(train_cube["cube"]), "validation samples", len(val_cube["cube"]))
    train_dataloader, val_dataloader = get_dataloaders(train_cube, val_cube, augmentation_params, cubeloader_args, dataloader_args, BATCH_SIZE, sorted_years=False)
    if UPDATE_LOSS_WEIGHTS:
        loss_args["main_args"]["weight"] = compute_class_weights(train_cube).tolist()
        load_model.set_loss_args(loss_args["main_args"], loss_args.get("macro_args", {}))
    print("Dataloaders created.")
    
    #### FINE_TUNE MODEL ON YEAR DATA        
    with EmissionsTracker() as tracker:
        trainer_fine, time_p_epoch = train_model_year(load_model, train_dataloader, val_dataloader, f"{full_exp_name}/{START_FROM_STEP}/{POST_NAME}", STORAGE_PATH, DEVICE=DEVICE, EPOCHS=EPOCHS, 
            early_stopping=early_stopping, store_checkpoints=store_checkpoints, **earlystopping_args)
    
    df_initial.loc[f"data_{START_FROM_STEP}"] = [full_exp_name, trainer_fine.logger.version, type(trainer_fine.model).__name__, START_FROM_STEP, 
                                trainer_fine.early_stopping_callback.best_score.item() if early_stopping else None, trainer_fine.checkpoint_callback.best_model_path if store_checkpoints else None, time_p_epoch, trainer_fine.current_epoch,
                                sum(p.numel() for p in trainer_fine.model.parameters()), sum(p.numel() for p in trainer_fine.model.parameters() if p.requires_grad), tracker.final_emissions]
    df_initial.to_csv(f"{STORAGE_PATH}/experiments_summary.csv", index=False)
    print("Saving config file in:", f"{trainer_fine.logger.log_dir}/config.yaml")
    yaml.dump(config_file, open(f"{trainer_fine.logger.log_dir}/config.yaml", 'w'))
        

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
    arg_parser.add_argument(
        "--version_load",
        "-v",
        required=False,
        type=int,
        default=0,
        help="version of the model to load from the model folder",
    )
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)

    main_run(config_file)
