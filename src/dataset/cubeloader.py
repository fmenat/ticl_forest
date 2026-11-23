import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
    
from .utils import sel_period_xarray
    
class CubeLoader(Dataset):
    def __init__(self, cube, lag_years, transforms, 
                 validation=False, years_validate = 1, aug_transform=None, extra_bands={}, 
                 load_memory=False, **kwargs):
        self.label_list = [label for label in ["disturbances", "disturbance_agent", "forest_mask_static"] if label in cube.data_vars]
        self.extra_labels = [label for label in ["sincelast_disturbances"] if label in cube.data_vars]
        self.transforms = transforms
        self.used_sensors = list(transforms.keys())
        self.extra_bands = extra_bands
        self.aug_transform = aug_transform
        self.validation = validation
        self.lag_years = lag_years
        
        #load dataset only with the necessary sensors and labels to save memory if needed
        total_vars = set(self.used_sensors + list(self.extra_bands.keys()) + self.label_list + self.extra_labels)
        total_vars |= set([v+"_norm" for v in total_vars if v+"_norm" in cube.data_vars]) 
        self.dataset = cube[total_vars]
        if load_memory:
            print("Loading entire dataset into memory...")
            self.dataset = self.dataset.compute()
        self.min_year = int(self.dataset.year.values[0])
        self.years_train = [int(y) for y in self.dataset.year.values if y > self.min_year] #there is at least one year before for lagged data
        if self.validation:
            self.years_validate = [y for y in self.years_train[-years_validate:]]
        
        self.year_to_indices = {year: [] for year in (self.years_validate if self.validation else self.years_train)} 
        self.samples = []
        for i_sample in range(len(self.dataset.cube)):
            if "valid_time" in self.dataset.coords:
                i_sample_validtime = self.dataset.coords["valid_time"].isel(cube=i_sample)
            for year in self.year_to_indices.keys(): 
                if ("valid_time" in self.dataset.coords) and (not all(i_sample_validtime.sel(year=slice(year-self.lag_years+1, year)).values)):
                    continue
                self.year_to_indices[year].append(len(self.samples))
                self.samples.append((i_sample, int(year)))
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        i_idx, year = self.samples[item]
        back_year = max(year-self.lag_years+1, self.min_year)
        idx_data = self.dataset.isel(cube=i_idx)
        idx_data.compute()

        filtered_data = sel_period_xarray(idx_data, start=back_year, end=year)
        
        sensors_sample = {}  #covariates (input data)
        for sensor in self.used_sensors:
            available_sensor = True
            for var_time in ["time", "time_10", "year"]:
                if f"start_{var_time}" not in filtered_data[sensor].attrs or f"end_{var_time}" not in filtered_data[sensor].attrs:
                    continue
                start_time_sensor = pd.Timestamp(filtered_data[sensor].attrs[f"start_{var_time}"])
                end_time_sensor = pd.Timestamp(filtered_data[sensor].attrs[f"end_{var_time}"])
                if pd.Timestamp(str(year-1)) < start_time_sensor: #skipping sensor
                    available_sensor = False 
                    continue
                if pd.Timestamp(str(year)) > end_time_sensor: #skipping sensor
                    available_sensor = False
                    continue
            if available_sensor:
                tensor_sample = torch.tensor(filtered_data[sensor].values)
                sensors_sample[sensor] = torch.nan_to_num(self.transforms[sensor](tensor_sample).to(torch.float32), 0.0)
                


        labels_data = sel_period_xarray(filtered_data, start=year, end=year) 
        
        labels_sample = {} #reference data (labels)
        for key in self.label_list: #for current year prediction
            labels_sample[key] = torch.tensor(labels_data[key].values.squeeze(), dtype=torch.long)

        #JUST FOR LOSS CALCULATION
        if "sincelast_disturbances" in filtered_data.data_vars:
            labels_sample["sincelast_disturbances"] = torch.tensor(labels_data["sincelast_disturbances"].values.squeeze(), dtype=torch.long)
        future_year= int(min(year+1, self.dataset.year.values[-1]))
        labels_sample["future_disturbance"] = torch.tensor(sel_period_xarray(idx_data, start=future_year, end=future_year)["disturbances"].values.squeeze()==1, dtype=torch.long)


        if len(self.extra_bands) > 0:
            for sensor in self.extra_bands:
                if sensor not in sensors_sample:
                    continue
                attach = [sensors_sample[sensor]]
                for band in self.extra_bands[sensor]:
                    if band in sensors_sample:
                        attach.append(sensors_sample.pop(band))
                    elif band in labels_sample:
                        if band == "sincelast_disturbances":
                            data = labels_sample[band]/idx_data["sincelast_disturbances_norm"].sel(norm="max").values
                        else:
                            data = labels_sample[band]
                        attach.append(data.repeat(attach[0].shape[0], 1, 1).unsqueeze(-1).to(attach[0].dtype))
                sensors_sample[sensor] = torch.concat(attach, dim=-1)
    
        
        if self.aug_transform is not None and not self.validation:
            input_data_to_augment = {}
            for sensor in ["S2", "alphaearth", "landcover"]:
                if sensor in sensors_sample:
                    input_data_to_augment[sensor] = (sensors_sample[sensor], (1, 2)) #augment only spatial dimensions
            for s in sensors_sample:
                if "_forest" in s or "dem" == s:
                    input_data_to_augment[s] = (sensors_sample[s], (0, 1)) #augment only spatial 

            label_data_to_augment = {}
            for label_key in labels_sample:
                label_data_to_augment[label_key] = (labels_sample[label_key], (0, 1))

            return_dic = self.aug_transform(input_data_to_augment, label_data_to_augment)
            for k in return_dic:
                if k in sensors_sample:
                    sensors_sample[k] = return_dic[k]
                elif k in labels_sample:
                    labels_sample[k] = return_dic[k]

        return {"input": sensors_sample, "label": labels_sample, "cube_name": str(filtered_data["cube"].values), "idx": i_idx, "year": year}
