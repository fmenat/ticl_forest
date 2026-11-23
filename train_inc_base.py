import pandas as pd
import xarray as xr
import numpy as np
import torch, gc, yaml, argparse, multiprocessing, copy, sys, os
from codecarbon import EmissionsTracker
from pathlib import Path
import pytorch_lightning as pl
from contextlib import nullcontext

from src.dataset.prepare import restore_dataset_attrs
from src.dataset.transforms import get_numerical_transform, get_categorical_transform
from src.dataset.utils import sel_period_xarray, concat_datasets
from src.training.pl_train import train_model_year, get_val_f1_from_tb
from src.training.utils import reset_weights
from src.training.data_setup import get_train_val_test_cubes, get_dataloaders, compute_class_weights
from src.training.names import COLUMNS_DF
from src.model.simple import UNetConvLSTM, UNetSimple, EncoderDecoder


def main_run(config_file, save_everything=True, early_stopping=True, version_load=0):
    DATA_FOLDER = config_file["data_folder"]
    STORAGE_PATH = config_file["storage_path"]
    DATASET_NAME = config_file.get("dataset_name", None)
    DEVICE = config_file.get("device")

    EXP_DATA = config_file.get("exp_data", {})
    EXP_NAME = EXP_DATA.get("exp_name")
    VERSION = EXP_DATA.get("version", version_load)
    MODEL_PATH = EXP_DATA.get("model_folder", STORAGE_PATH)
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
        "pin_memory": num_workers>0,
        "num_workers": num_workers,
        "persistent_workers": num_workers>0, 
        "shuffle": False,
    }

    INCREMENTAL_STRATEGY_ARGS = config_file.get("incremental_strategy")
    MODELS_TRAIN = INCREMENTAL_STRATEGY_ARGS.get("approaches")  #which models to train: fine (incremental), fresh (re-initialized weights), concat (re-initialized weights but trained on all data until the current year, not just the incremental year)
    FIXED_INCREMENTAL_START = INCREMENTAL_STRATEGY_ARGS.get("fixed_incremental_start", True) #whether to fix the start year for the incremental model, which can be useful when training on a subset of the data, but should be set to False when training on all data to allow the incremental model to start from the first year of the dataset.
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
    RESET_WEIGHTS_PRETRAINED = training_args.get("reset_weights_pretrained", False)
    RESET_WEIGHTS_INITIAL_MODEL = training_args.get("reset_weights_initial_model", True)
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
        
        load_model.set_loss_args(loss_args["main_args"], loss_args.get("macro_args", {}))
        load_model.hparams.update(**optim_args)
        if RESET_WEIGHTS_PRETRAINED:
            reset_weights(load_model)
            full_exp_name = full_exp_name.split("_v")[0]

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
        
    initial_model = copy.deepcopy(load_model) #copy for training fresh model later with reset weights but same initial weights as the incremental model
    if RESET_WEIGHTS_INITIAL_MODEL:
        reset_weights(initial_model)
        full_exp_name_initial = full_exp_name.split("_v")[0]
    else:
        full_exp_name_initial = full_exp_name
    
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
        df_fresh = pd.read_csv(f"{STORAGE_PATH}/experiments_summary_fresh{POST_NAME}.csv")
    except FileNotFoundError:
        df_fresh = pd.DataFrame(columns=COLUMNS_DF)

    try:
        df_concat = pd.read_csv(f"{STORAGE_PATH}/experiments_summary_concat{POST_NAME}.csv")
    except FileNotFoundError:
        df_concat = pd.DataFrame(columns=COLUMNS_DF)

    step_vals = []
    for i, step in enumerate(range(START_FROM_STEP, STEP_INCREMENTAL[-1]+ (0 if VAL_TYPE=="temporal" else 1))): 
        
        if i == 0 and len(EXP_DATA)>0 and FIXED_INCREMENTAL_START: #first step model only store
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> First step model loaded, then only store, no training... ")
    
            if "concat" in MODELS_TRAIN:
                trainer = train_model_year(load_model, None, None, f"{full_exp_name_initial}/{step}", STORAGE_PATH, DEVICE=DEVICE, EPOCHS=0, prefix=f"concat{POST_NAME}", only_return_trainer=True,**earlystopping_args)            
                temporal_storage = Path(trainer.logger.log_dir)
                temporal_storage.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": load_model.state_dict(),"hyper_parameters": load_model.hparams, "pytorch-lightning_version": pl.__version__,}, f"{temporal_storage}/model_initial.ckpt")
                yaml.dump(config_file, open(f"{temporal_storage}/config.yaml", 'w'))
                df_concat.loc[f"data_{step}", COLUMNS_DF] = [full_exp_name_initial, str(trainer.logger.version), type(load_model).__name__, str(step), np.nan, f"{temporal_storage}/model_initial.ckpt", np.nan, 0,
                                            sum(p.numel() for p in load_model.parameters()), sum(p.numel() for p in load_model.parameters() if p.requires_grad), np.nan]
                df_concat.to_csv(f"{STORAGE_PATH}/experiments_summary_concat{POST_NAME}.csv", index=False)
            
            del trainer
            continue
        

        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Learning base models on year {step}... for {EPOCHS} epochs ") 
        step_limit = step+1

        if "fresh" in MODELS_TRAIN and i != 0:
            ### GET TRAINING AND VALIDATION CUBES AND DATALOADERS
            train_cube, val_cube, _ = get_train_val_test_cubes(data_load, step_limit, STEP_INCREMENTAL, LAG_YEAR, VAL_TYPE, first_flag=i==0, **train_val_split_args)
            print("training steps/year", train_cube["year"].values, "validation", val_cube["year"].values)
            print("training samples", len(train_cube["cube"]), "validation samples", len(val_cube["cube"]))
            train_dataloader, val_dataloader = get_dataloaders(train_cube, val_cube, augmentation_params, cubeloader_args, dataloader_args, BATCH_SIZE, sorted_years=False)
            if UPDATE_LOSS_WEIGHTS:
                loss_args["main_args"]["weight"] = compute_class_weights(train_cube).tolist()
            print("Dataloaders created.")

            ### FOR COMPARISON TRAIN A FRESH MODEL ON THE SAME YEAR DATA
            print(">>>>>>>>>>>>>>>>>>>> Training model for step", step, "with weights re-initialized (training from scratch)...")        
            #create a copy with reset weights
            fresh_model = copy.deepcopy(initial_model) #clean without loading new optimizer values
            if UPDATE_LOSS_WEIGHTS:
                fresh_model.set_loss_args(loss_args["main_args"], loss_args.get("macro_args", {}))

            with EmissionsTracker() if save_everything else nullcontext() as tracker:
                trainer_fresh, time_p_epoch = train_model_year(fresh_model, train_dataloader, val_dataloader, f"{full_exp_name_initial}/{step}", STORAGE_PATH, EPOCHS=EPOCHS, prefix=f"fresh{POST_NAME}", 
                                            store_checkpoints=save_everything, early_stopping=early_stopping, DEVICE=DEVICE,  **earlystopping_args)
            
            if save_everything:
                ### SAVE EXPERIMENT SUMMARY
                df_fresh.loc[f"data_{step}", COLUMNS_DF] = [full_exp_name_initial, str(trainer_fresh.logger.version), type(trainer_fresh.model).__name__, str(step), str(trainer_fresh.early_stopping_callback.best_score.item()) if early_stopping else np.nan, trainer_fresh.checkpoint_callback.best_model_path, time_p_epoch, str(trainer_fresh.current_epoch),
                                            sum(p.numel() for p in trainer_fresh.model.parameters()), sum(p.numel() for p in trainer_fresh.model.parameters() if p.requires_grad), tracker.final_emissions]
                df_fresh.to_csv(f"{STORAGE_PATH}/experiments_summary_fresh{POST_NAME}.csv", index=False)
                print("Saving config file in:", f"{trainer_fresh.logger.log_dir}/config.yaml")
                yaml.dump(config_file, open(f"{trainer_fresh.logger.log_dir}/config.yaml", 'w'))
            
            train_cube.close()
            val_cube.close()
            del train_cube, val_cube, train_dataloader, val_dataloader, fresh_model, trainer_fresh
            torch.cuda.empty_cache()
            gc.collect()


        if "concat" in MODELS_TRAIN:
            ### FOR COMPARISON TRAIN A STACKED MODEL ON ALL THE COLLECTIVE YEAR DATA
            print(">>>>>>>>>>>>>>>>>>>> Training model with data until step", step, "with weights re-initialized (training from scratch)...")
            train_cube, val_cube, _ = get_train_val_test_cubes(data_load, step_limit, STEP_INCREMENTAL, LAG_YEAR, VAL_TYPE, first_flag=True, **train_val_split_args)
            print("training", train_cube["year"].values, "validation", val_cube["year"].values)
            print("training samples", len(train_cube["cube"]), "validation samples", len(val_cube["cube"]))
            train_dataloader, val_dataloader = get_dataloaders(train_cube, val_cube, augmentation_params, cubeloader_args, dataloader_args, BATCH_SIZE, sorted_years=False)
            print("Dataloaders created.")
            if UPDATE_LOSS_WEIGHTS:
                loss_args["main_args"]["weight"] = compute_class_weights(train_cube).tolist()

            #create a copy with reset weights
            concat_model = copy.deepcopy(initial_model) #clean without loading new optimizer values
            if UPDATE_LOSS_WEIGHTS:
                concat_model.set_loss_args(loss_args["main_args"], loss_args.get("macro_args", {}))
            
            with EmissionsTracker() if save_everything else nullcontext() as tracker:
                trainer_concat, time_p_epoch = train_model_year(concat_model, train_dataloader, val_dataloader, f"{full_exp_name_initial}/{step}", STORAGE_PATH, EPOCHS=EPOCHS, prefix=f"concat{POST_NAME}",
                                                store_checkpoints=save_everything, early_stopping=early_stopping, DEVICE=DEVICE,  **earlystopping_args)

            if save_everything:
                ### SAVE EXPERIMENT SUMMARY
                df_concat.loc[f"data_{step}", COLUMNS_DF] = [full_exp_name_initial, str(trainer_concat.logger.version), type(trainer_concat.model).__name__, str(step), str(trainer_concat.early_stopping_callback.best_score.item()) if early_stopping else np.nan, trainer_concat.checkpoint_callback.best_model_path, time_p_epoch, str(trainer_concat.current_epoch),
                                            sum(p.numel() for p in trainer_concat.model.parameters()), sum(p.numel() for p in trainer_concat.model.parameters() if p.requires_grad), tracker.final_emissions]
                df_concat.to_csv(f"{STORAGE_PATH}/experiments_summary_concat{POST_NAME}.csv", index=False)
                print("Saving config file in:", f"{trainer_concat.logger.log_dir}/config.yaml")
                yaml.dump(config_file, open(f"{trainer_concat.logger.log_dir}/config.yaml", 'w'))

            if early_stopping:
                step_best = trainer_concat.early_stopping_callback.best_score.item()
            else: #last epoch score (no checpoint)
                step_best = get_val_f1_from_tb(trainer_concat, which="best", tag=f"{metric_stopval}/val")
            step_vals.append(step_best)

            del train_dataloader, val_dataloader, concat_model, trainer_concat
            gc.collect()
            train_cube.close()
            val_cube.close()
            del train_cube, val_cube
            torch.cuda.empty_cache()
            gc.collect()
        
    return np.mean(step_vals) if len(step_vals)>0 else None


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
    version_load = args.version_load
    
    main_run(config_file, version_load=version_load)
