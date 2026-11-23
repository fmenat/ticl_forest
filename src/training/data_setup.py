
import numpy as np
import sys
from torch.utils.data import DataLoader
from sklearn.utils import class_weight


from src.dataset.prepare import train_test_splitting
from src.dataset.cubeloader import CubeLoader
from src.dataset.batchloader import YearGroupedBatchSampler
from src.dataset.transforms import RandGroupTransform


def get_train_val_test_cubes(data_load, year_limit, YEARS_INCREMENTAL, LAG_YEAR, VAL_TYPE, sample_train=0, sample_val=0, sample_test=0, first_flag=True, random_state=42):
    np.random.seed(random_state)
    if VAL_TYPE == "temporal":
        if first_flag:
            train_cube, test_cube = train_test_splitting(data_load, YEARS_INCREMENTAL[0], 9999, LIM_TEST=year_limit+1, YEARS_LEN=LAG_YEAR, 
                                        return_test=False, return_val=True, return_train=True)
        else:
            train_cube, test_cube = train_test_splitting(data_load, 0, year_limit, LIM_TEST=year_limit, YEARS_LEN=LAG_YEAR, 
                                        return_test=True, return_val=True, return_train=False)

        if sample_train > 0:
            if type(sample_train) == float:
                sample_train = int(len(train_cube.cube)*sample_train)
            cubes_train = np.random.choice(train_cube.cube.values, size=sample_train, replace=False)
            train_cube = train_cube.sel(cube=cubes_train)
        
        if sample_val > 0: #in this case the val is created from test since it is in the future year
            if sample_test > 0:
                remaining_cubes_val_test = np.setdiff1d(test_cube.cube.values, cubes_test)  
            else:
                remaining_cubes_val_test = test_cube.cube.values
            if type(sample_val) == float:
                sample_val = int(len(remaining_cubes_val_test)*sample_val)
            cubes_val = np.random.choice(remaining_cubes_val_test, size=sample_val, replace=False)
            val_cube = test_cube.sel(cube=cubes_val)
            test_cube = test_cube.drop_sel(cube=cubes_val)
        
        elif sample_test > 0:
            if type(sample_test) == float:
                sample_test = int(len(test_cube.cube)*sample_test)
            cubes_test = np.random.choice(test_cube.cube.values, size=sample_test, replace=False)
            #cubes_val = np.setdiff1d(test_cube.cube.values, cubes_test) 
            val_cube = test_cube.drop_sel(cube=cubes_test)
            test_cube = test_cube.sel(cube=cubes_test)
    

    elif VAL_TYPE == "spatial":
        if first_flag:
            cube_data = train_test_splitting(data_load, YEARS_INCREMENTAL[0], 9999, LIM_TEST=year_limit+1, YEARS_LEN=LAG_YEAR, 
                                        return_test=False, return_val=False, return_train=True)[0]
        else:
            cube_data = train_test_splitting(data_load, 0, year_limit, LIM_TEST=year_limit, YEARS_LEN=LAG_YEAR, 
                                        return_test=False, return_val=True, return_train=False)[0]
        if sample_test > 0: #test set is seprated
            if type(sample_test) == float:
                sample_test = int(len(cube_data.cube)*sample_test)
            cubes_test = np.random.choice(cube_data.cube.values, size=sample_test, replace=False)
            
            remaining_cubes_train_val = np.setdiff1d(cube_data.cube.values, cubes_test)
            if sample_val > 0:
                if type(sample_val) == float:
                    sample_val = int(len(remaining_cubes_train_val)*sample_val)
                cubes_val = np.random.choice(remaining_cubes_train_val, size=sample_val, replace=False)
                if sample_train > 0:
                    remaining_cubes_train = np.setdiff1d(remaining_cubes_train_val, cubes_val)
                    if type(sample_train) == float:
                        sample_train = int(len(remaining_cubes_train)*sample_train)
                    cubes_train = np.random.choice(remaining_cubes_train, size=sample_train, replace=False)
                else:   
                    cubes_train = np.setdiff1d(remaining_cubes_train_val, cubes_val)
            elif sample_train > 0:
                if type(sample_train) == float:
                    sample_train = int(len(remaining_cubes_train_val)*sample_train)
                cubes_train = np.random.choice(remaining_cubes_train_val, size=sample_train, replace=False)
                cubes_val = np.setdiff1d(remaining_cubes_train_val, cubes_train)
            else:
                cubes_train = remaining_cubes_train_val

        elif sample_train > 0: #training data is seprated from the rest
            if type(sample_train) == float:
                sample_train = int(len(cube_data.cube)*sample_train)
            cubes_train = np.random.choice(cube_data.cube.values, size=sample_train, replace=False)
            
            remaining_cubes_val_test = np.setdiff1d(cube_data.cube.values, cubes_train)
            if sample_test > 0:
                if type(sample_test) == float:
                    sample_test = int(len(remaining_cubes_val_test)*sample_test)
                cubes_test = np.random.choice(remaining_cubes_val_test, size=sample_test, replace=False)
                cubes_val = np.setdiff1d(remaining_cubes_val_test, cubes_test)
            elif sample_val > 0:
                if type(sample_val) == float:
                    sample_val = int(len(remaining_cubes_val_test)*sample_val)
                cubes_val = np.random.choice(remaining_cubes_val_test, size=sample_val, replace=False)
                cubes_test = np.setdiff1d(remaining_cubes_val_test, cubes_val)
            else:                
                cubes_test = remaining_cubes_val_test

        train_cube = cube_data.sel(cube=cubes_train) 
        val_cube = cube_data.sel(cube=cubes_val)
        test_cube = cube_data.sel(cube=cubes_test)
        
    return train_cube, val_cube, test_cube


def get_dataloaders(train_cube, val_cube, augmentation_params, cubeloader_args, dataloader_args, BATCH_SIZE, sorted_years=True):
    rand_aug = None if len(augmentation_params) == 0 else RandGroupTransform(**augmentation_params)
    training_data = CubeLoader(train_cube, validation=False, aug_transform=rand_aug, **cubeloader_args)
    if val_cube is not None:
        validation_data = CubeLoader(val_cube, validation=True, years_validate=1,**cubeloader_args)

    train_yearlysampler = YearGroupedBatchSampler(training_data, batch_size=BATCH_SIZE, sorted_years=sorted_years)
    train_dataloader = DataLoader(training_data, batch_sampler=train_yearlysampler, **dataloader_args)
    if val_cube is not None:
        val_yearlysampler = YearGroupedBatchSampler(validation_data, batch_size=BATCH_SIZE)
        val_dataloader = DataLoader(validation_data, batch_sampler=val_yearlysampler, **dataloader_args)
    else:
        val_dataloader = None
    return train_dataloader, val_dataloader

    
def compute_class_weights(train_cube, label_name="disturbances"): #, forest_mask_name="forest_mask_static"):
    classes = np.asarray(train_cube[label_name].attrs["flag_values"][:-1])
    labels_ = train_cube[label_name].values #[train_cube[forest_mask_name].values[:,None,:,:].repeat(len(train_cube.year), axis=1) == 1]
    labels_ = labels_[(labels_ != train_cube[label_name].attrs["no_data_value"]) & ~np.isnan(labels_)]
    return class_weight.compute_class_weight(class_weight='balanced',classes=classes, y=labels_)