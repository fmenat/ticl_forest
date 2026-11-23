import pandas as pd
import numpy as np
import xarray as xr

def apply_indexing(array, dims, i, j, h, w):
    indexs = []
    for t in range(len(array.shape)):
        if t == dims[0]:
            indexs.append(slice(i, i+h))
        elif t == dims[1]:
            indexs.append(slice(j, j+w))
        else:
            indexs.append(slice(None))
    return array[tuple(indexs)]


def sel_period_xarray(ds, start=None, end=None, time_vars=["year", "time", "time_10"]):
    start_year, end_year = None, None
    if start is not None:
        if isinstance(start, int):
            start = f"{start}-01-01"
        start = pd.Timestamp(str(start))
        start_year = start.year
    if end is not None:
        if isinstance(end, int):
            end = f"{end}-12-31"
        end = pd.Timestamp(str(end))
        end_year = end.year

    for var in time_vars:
        if var in ds:
            if np.issubdtype(ds[var].dtype, np.integer):
                ds = ds.sel({var: slice(start_year, end_year)})
            else:
                ds = ds.sel({var: slice(start, end)})
    return ds

def sel_year_xarray(ds, year_data, time_vars=["year","time", "time_10"]):
    return sel_period_xarray(ds, start=year_data, end=year_data, time_vars=time_vars)


def concat_datasets(dataset_list, DATA_FOLDER, concat_dim="cube"):
    data_load = []
    for name in dataset_list:
        print("Loading data from:", DATA_FOLDER, name, "...")
        data = xr.open_zarr(f"{DATA_FOLDER}/{name}.zarr", mask_and_scale=True, consolidated=True)
        data_load.append(data)
    
    global_vars = [v for v in data_load[0].data_vars if concat_dim not in data_load[0][v].dims]
    aggregated = {
        v: xr.concat(
            [ds[v] for ds in data_load],
            dim="dataset"
        ).mean("dataset")
        for v in global_vars
    }

    data_load = [ds.drop_vars(global_vars) for ds in data_load] 
    cube_index = xr.concat([  xr.DataArray([dataset_list[i].split("_")[-1] + "_" + c for c in ds[concat_dim].values], dims=concat_dim) for i, ds in enumerate(data_load)],  dim=concat_dim)
    concat_dataset = xr.concat(data_load, dim=concat_dim, data_vars="minimal", coords="minimal", compat="equals", join="exact").assign_coords(**{concat_dim: cube_index})
    
    for v, data in aggregated.items():
        concat_dataset[v] = data

    return concat_dataset