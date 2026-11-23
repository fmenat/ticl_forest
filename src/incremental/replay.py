import numpy as np
import xarray as xr

def replay_update_training_cube(train_cube, data_load, incremental_steps, replay_type, replay_size, step="year", label_name="disturbances"):
    missing_years = set(incremental_steps) - set(train_cube[step].values)
    if len(missing_years) == 0:
        print("No missing years to replay.")
        return train_cube

    #select replay data/buffer
    if replay_type == "random":
        if type(replay_size) == float:
            replay_size = int(len(train_cube.cube) * replay_size)
        selected_cubes = np.random.choice(train_cube.cube.values, size=replay_size, replace=False)
    else:
        selected_cubes = train_cube.cube.values
    previous_step_data = data_load.sel(**{step: list(missing_years), "cube": selected_cubes})

    #concat new data/buffer to new training cube
    train_cube = xr.concat([previous_step_data, train_cube], dim="year", join="outer")

    valid_time_mask = ~train_cube[label_name].isnull().all(dim=("y", "x"))
    train_cube = train_cube.assign_coords(valid_time=(("cube", "year"), valid_time_mask.values))
    
    print(f"Keeping {replay_size} samples from previous training data for replay.")
    return train_cube