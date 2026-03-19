# era5_dataset.py
import xarray as xr
import torch
from torch.utils.data import Dataset
import xbatcher


import time
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset

from graphcast import graphcast, data_utils
import dataclasses

import jax
import jax.numpy as jnp



import os
import glob
import random
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
import dataclasses
from graphcast import data_utils



def data_merge_make_circular_one_hot_cpu(
    batch_size: int,
    height: int,
    width: int,
    centers: list,  # List of (i,j) tuples for multiple typhoon centers
    radius: int
) -> torch.Tensor:
    """Generate one-hot tensor with multiple circular regions."""
    yy = np.arange(height)[:, None] # 181
    xx = np.arange(width)[None, :]  # 360
    
    # Initialize empty mask
    mask2d = np.zeros((height, width), dtype=np.int64)
    
    # Add circular region for each center
    for center_i, center_j in centers:
        curr_mask = ((yy - center_i)**2 + (xx - center_j)**2 <= radius**2)
        mask2d |= curr_mask  # Combine masks using OR operation
    
    # Broadcast to batch dimension
    labels = np.broadcast_to(mask2d, (batch_size, height, width))
    labels_t = torch.from_numpy(labels)
    one_hot = torch.nn.functional.one_hot(labels_t, num_classes=2)
    
    return one_hot

class DateMergedERA5TyphoonDataset(Dataset):
    def __init__(
        self,
        full_ds: xr.Dataset,
        task_config,
        tracks_folder: str,
        start_year: int,
        end_year: int,
        label_radius: int = 5,
    ):
        """
        full_ds: ERA5 xarray.Dataset with 12h intervals
        task_config: graphcast dataclass parameters
        tracks_folder: Directory containing tracks_YYYY.txt files
        start_year, end_year: Year range for track files
        label_radius: Radius (in grid points) for one-hot mask circles
        """
        label_radius = 2

        self.ds = full_ds
        self.task_config = task_config
        self.label_radius = label_radius

        # 1) Read and concatenate all track files
        dfs = []
        for yr in range(start_year, end_year + 1):
            path = os.path.join(tracks_folder, f"tracks_{yr}.txt")
            if os.path.exists(path):
                df = pd.read_csv(path, sep=",", comment="#")
                df.columns = df.columns.str.strip()
                dfs.append(df)
        self.tracks = pd.concat(dfs, ignore_index=True)

        # 2) Filter: keep only timestamps that exist in full_ds.time
        ds_times = pd.to_datetime(self.ds["time"].values)
        valid_ts = set(ds_times.strftime("%Y-%m-%dT%H:%M:%S"))
        
        def row_in_ds(r):
            ts = pd.Timestamp(
                year=int(r["year"]),
                month=int(r["month"]),
                day=int(r["day"]),
                hour=int(r["hour"]),
            ).strftime("%Y-%m-%dT%H:%M:%S")
            return ts in valid_ts

        mask = self.tracks.apply(row_in_ds, axis=1)
        self.tracks = self.tracks[mask].reset_index(drop=True)

        # 3) Group tracks by timestamp for faster lookup | using i j
        # self.grouped_tracks = self.tracks.groupby(
        #     ["year", "month", "day", "hour"]
        # ).apply(lambda x: x[["track_id", "i", "j"]].values.tolist()).to_dict()

        # 3) Group tracks by timestamp for faster lookup | using lon lat
        self.grouped_tracks = self.tracks.groupby(
            ["year", "month", "day", "hour"]
        ).apply(lambda x: x[["track_id", "lon", "lat"]].values.tolist()).to_dict()

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        row = self.tracks.iloc[idx]
        ts = pd.Timestamp(
            year=int(row["year"]),
            month=int(row["month"]),
            day=int(row["day"]),
            hour=int(row["hour"]),
        )

        # 处理日期边界，不能是{start_year}-01-01
        if ts - np.timedelta64(24, "h") < self.ds["time"].min():
            idx = idx + 3
            row = self.tracks.iloc[idx]
            ts = pd.Timestamp(
                year=int(row["year"]),
                month=int(row["month"]),
                day=int(row["day"]),
                hour=int(row["hour"]),
            )

        # print fetch time in readable format
        # print(f"Fetching data for timestamp: {ts.strftime('%Y-%m-%d %H:%M:%S')} (index {idx})")

        # Get data window
        window = self.ds.sel(time=[
            ts - np.timedelta64(24, "h"),
            ts - np.timedelta64(12, "h"),
            ts
        ]).load()
        patch_ds = self.transform_sample(window)


        # Extract features
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            patch_ds,
            target_lead_times=slice("12h", "12h"),
            **dataclasses.asdict(self.task_config)
        )

        batch = inputs.sizes["batch"]

        # Find all typhoons at this timestamp
        key = (int(row["year"]), int(row["month"]), int(row["day"]), int(row["hour"]))
        typhoon_positions = self.grouped_tracks.get(key, [])
        # print(f"Found {len(typhoon_positions)} typhoons at {ts}.")
        # Convert all positions to downsampled grid coordinates
        ds_factor = 4
        centers = []
        # i, j, lon, lat
        # for _, orig_i, orig_j in typhoon_positions:
        #     i_small = int(orig_i) // ds_factor
        #     j_small = int(orig_j) // ds_factor
        #     center_i = i_small  # Downsampled i
        #     center_j = (720 // ds_factor - 1) - j_small  # Inverted j for downsampled grid
        #     centers.append((center_i, center_j))

        for _, orig_lon, orig_lat in typhoon_positions:
            i_small = int(orig_lon)
            j_small = int(orig_lat + 90)

            # detect if i_small < 0 or i_small >= 360 or j_small < 0 or j_small >= 181:
            i_small = 0 if i_small < 0 else i_small
            i_small = 360 if i_small >= 360 else i_small
            j_small = 0 if j_small < 0 else j_small
            j_small = 181 if j_small >= 181 else j_small

            center_row_id = j_small 
            center_col_id = i_small
            centers.append((center_row_id, center_col_id))

        # Generate combined one-hot mask for all typhoons
        one_hot = data_merge_make_circular_one_hot_cpu(
            batch_size=batch,
            height=181,
            width=360,
            centers=centers,
            radius=self.label_radius,
        )


        return inputs, targets, forcings, one_hot, ts

    def transform_sample(self, sample_ds: xr.Dataset) -> xr.Dataset:
        # 1. Rename dimensions
        sample_ds = sample_ds.rename({"longitude": "lon", "latitude": "lat", "time": "datetime"})
        
        # 2. Downsample to 1° & flip latitude
        sample_ds = (
            sample_ds
            .isel(lat=slice(0, None, 4), lon=slice(0, None, 4))
            .isel(lat=slice(None, None, -1))
        )
        
        # 3. Create relative time coordinates
        delta = np.timedelta64(12, "h")
        n = len(sample_ds["datetime"])
        rel_times = np.arange(0, n * delta, delta, dtype="timedelta64[ns]")
        sample_ds = sample_ds.assign_coords(time=("datetime", rel_times))
        
        # 4. Preserve absolute datetime
        abs_times = sample_ds["datetime"].values[np.newaxis, :]
        sample_ds = sample_ds.swap_dims({"datetime": "time"}).drop_vars("datetime", errors="ignore")
        sample_ds = sample_ds.assign_coords(datetime=(("batch", "time"), abs_times))
        
        # 5. Add batch dimension
        for var in sample_ds.data_vars:
            if "batch" not in sample_ds[var].dims:
                sample_ds[var] = sample_ds[var].expand_dims(batch=1)
        return sample_ds




def make_circular_one_hot_cpu(
    batch_size: int,
    height: int,
    width: int,
    center_i: int,
    center_j: int,
    radius: int
) -> torch.Tensor:
    yy = np.arange(height)[:, None]
    xx = np.arange(width)[None, :]
    mask2d = ((yy - center_i)**2 + (xx - center_j)**2 <= radius**2).astype(np.int64)
    labels = np.broadcast_to(mask2d, (batch_size, height, width))
    labels_t = torch.from_numpy(labels)  # LongTensor
    one_hot = torch.nn.functional.one_hot(labels_t, num_classes=2)  # (B, H, W, 2)
    return one_hot

class ERA5TyphoonDataset(Dataset):
    def __init__(
        self,
        full_ds: xr.Dataset,
        task_config,
        tracks_folder: str,
        start_year: int,
        end_year: int,
        label_radius: int = 20,
    ):
        """
        full_ds: 已切成 12h 间隔的完整 ERA5 xarray.Dataset
        task_config: graphcast 的 dataclass 参数
        tracks_folder: 放 tracks_YYYY.txt 的目录
        start_year, end_year: 轨迹文件年份范围
        label_radius: 生成 one-hot 掩码的半径（格点数）
        """
        label_radius = 5
        self.ds = full_ds
        self.task_config = task_config
        self.label_radius = label_radius

        # 1) 读取并拼接所有轨迹文件
        dfs = []
        for yr in range(start_year, end_year + 1):
            path = os.path.join(tracks_folder, f"tracks_{yr}.txt")
            if os.path.exists(path):
                df = pd.read_csv(path, sep=",", comment="#")
                df.columns = df.columns.str.strip()
                dfs.append(df)
        self.tracks = pd.concat(dfs, ignore_index=True)

        # 2) 过滤：只保留那些时间点 TS 恰好出现在 full_ds.time 中的行
        ds_times = pd.to_datetime(self.ds["time"].values)
        # 构造一组“字符串形式的时间戳”以便快速匹配
        valid_ts = set(ds_times.strftime("%Y-%m-%dT%H:%M:%S"))
        def row_in_ds(r):
            ts = pd.Timestamp(
                year = int(r["year"]),
                month= int(r["month"]),
                day  = int(r["day"]),
                hour = int(r["hour"]),
            ).strftime("%Y-%m-%dT%H:%M:%S")
            return ts in valid_ts

        mask = self.tracks.apply(row_in_ds, axis=1)
        self.tracks = self.tracks[mask].reset_index(drop=True)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        row = self.tracks.iloc[idx]
        ts = pd.Timestamp(
            year = int(row["year"]),
            month= int(row["month"]),
            day  = int(row["day"]),
            hour = int(row["hour"]),
        )

        window = self.ds.sel(time=[
            ts - np.timedelta64(24, "h"),
            ts - np.timedelta64(12, "h"),
            ts
        ]).load()
        patch_ds = self.transform_sample(window)

        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            patch_ds,
            target_lead_times = slice("12h", "12h"),
            **dataclasses.asdict(self.task_config)
        )

        batch = inputs.sizes["batch"]


        # H, W = inputs.sizes["lat"], inputs.sizes["lon"]
        # orig_i, orig_j = int(row["i"]), int(row["j"])
        # i_small, j_small = orig_i // 4, orig_j // 4
        # center_i, center_j = H - 1 - j_small, i_small

        # one_hot = make_circular_one_hot_cpu(
        #     batch_size = batch,
        #     height     = H,
        #     width      = W,
        #     center_i   = center_i,
        #     center_j   = center_j,
        #     radius     = self.label_radius,
        # )

        # 原始全分辨率网格大小
        lon_full = 1440
        lat_full =  720
        ds_factor = 4  # 下采样因子

        orig_i, orig_j = int(row["i"]), int(row["j"])
        # 下采样到小网格
        i_small = orig_i // ds_factor    # ∈ [0, 359]
        j_small = orig_j // ds_factor    # ∈ [0, 179]

        # 翻转纬度索引（北起算）并组成 center
        center_i = (lat_full // ds_factor - 1) - j_small  # = 180-1 - j_small
        center_j = i_small

        one_hot = make_circular_one_hot_cpu(
            batch_size = batch,
            height     = 181,  # 180
            width      = 360,  # 360
            center_i   = center_i,
            center_j   = center_j,
            radius     = self.label_radius,
        )

        return inputs, targets, forcings, one_hot



    def transform_sample(self, sample_ds: xr.Dataset) -> xr.Dataset:
        # 与 ERA5Dataset_train_GenCast 一致的预处理
        # 1. 重命名
        sample_ds = sample_ds.rename({"longitude": "lon", "latitude": "lat", "time": "datetime"})
        # 2. 1° 重采样 & 纬度翻转
        sample_ds = (
            sample_ds
            .isel(lat=slice(0, None, 4), lon=slice(0, None, 4))
            .isel(lat=slice(None, None, -1))
        )
        # 3. 相对时间坐标
        delta = np.timedelta64(12, "h")
        n = len(sample_ds["datetime"])
        rel_times = np.arange(0, n * delta, delta, dtype="timedelta64[ns]")
        sample_ds = sample_ds.assign_coords(time=("datetime", rel_times))
        # 4. 保留绝对 datetime
        abs_times = sample_ds["datetime"].values[np.newaxis, :]
        sample_ds = sample_ds.swap_dims({"datetime": "time"}).drop_vars("datetime", errors="ignore")
        sample_ds = sample_ds.assign_coords(datetime=(("batch", "time"), abs_times))
        # 5. 扩展 batch 维度
        for var in sample_ds.data_vars:
            if "batch" not in sample_ds[var].dims:
                sample_ds[var] = sample_ds[var].expand_dims(batch=1)
        return sample_ds










class ERA5Dataset_train_GenCast(Dataset):
    def __init__(
        self,
        batch_generator: xbatcher.BatchGenerator,
        input_vars: list[str],
        target_var: str,
        task_config: None,
        time_interval: str = "12h",
        cache_size: int = 10,
    ):
        self.bgen = batch_generator
        self.input_vars = input_vars
        self.target_var = target_var
        self.time_interval = time_interval
        self.cache_size = cache_size
        self.cache = {}
        self.length = len(self.bgen)
        self.task_config = task_config

    def __len__(self):
        return self.length

    def transform_sample(self, sample_ds: xr.Dataset) -> xr.Dataset:
        """Transform a single sample to GenCast format."""
        start = time.time()
        try:
            # 1. Rename dims
            sample_ds = sample_ds.rename({"longitude": "lon", "latitude": "lat", "time": "datetime"})
            # 2. Subsample to 1° resolution & invert latitude
            sample_ds = (
                sample_ds
                .isel(lat=slice(0, None, 4), lon=slice(0, None, 4))
                .isel(lat=slice(None, None, -1))
            )
            # 3. Relative time coordinate
            delta = np.timedelta64(12, "h") if self.time_interval == "12h" else np.timedelta64(6, "h")
            n = len(sample_ds["datetime"])
            rel_times = np.arange(0, n * delta, delta, dtype="timedelta64[ns]")
            sample_ds = sample_ds.assign_coords(time=("datetime", rel_times))
            # 4. Preserve absolute datetime
            abs_times = sample_ds["datetime"].values[np.newaxis, :]
            sample_ds = sample_ds.swap_dims({"datetime": "time"}).drop_vars("datetime", errors="ignore")
            sample_ds = sample_ds.assign_coords(datetime=(("batch", "time"), abs_times))


            # ───────── DEBUG: force all values to each variable’s mean ─────────
            # for var in sample_ds.data_vars:
            #     da       = sample_ds[var]
            #     # Compute the mean over all dims (ignores coords)
            #     var_mean = float(da.mean().values)
            #     # Create a DataArray full of that mean, preserving shape, coords, dtype, attrs
            #     filled   = xr.full_like(da, var_mean)
            #     filled.attrs = da.attrs
            #     sample_ds[var] = filled
            # ────────────────────────────────────────────────────────────────


            # # radius
            # r = 40

            # # 要修改的变量名
            # var = 'total_precipitation_12hr'
            # da = sample_ds[var]

            # # 计算这个变量在整个样本上的最大值
            # var_max = float(da.max().values)

            # # 准备一个全 0 的 ndarray，shape 与 da 一致
            # mask = np.zeros(da.shape, dtype=da.dtype)

            # # 找到 lat/lon 维度对应的索引
            # lat_idx = da.dims.index("lat")
            # lon_idx = da.dims.index("lon")

            # H = da.sizes["lat"]
            # W = da.sizes["lon"]
            # yc, xc = H // 2, W // 2

            # # 构造切片，其他维度全取，lat/lon 只取中心方块
            # slicer = [slice(None)] * da.ndim
            # slicer[lat_idx] = slice(yc - r, yc + r + 1)
            # slicer[lon_idx] = slice(xc - r, xc + r + 1)

            # # 在中心方块内赋值为 var_max
            # mask[tuple(slicer)] = var_max

            # # 用新的 DataArray 覆盖原来的 sample_ds[var]
            # sample_ds[var] = xr.DataArray(
            #     mask,
            #     dims=da.dims,
            #     coords=da.coords,
            #     attrs=da.attrs,
            # )
            # ───────────────────────────────────────────────────────────


            # 5. Add batch dimension
            for var in sample_ds.data_vars:
                if "batch" not in sample_ds[var].dims:
                    sample_ds[var] = sample_ds[var].expand_dims(batch=1)
            return sample_ds
        except Exception as e:
            print(f"Error in transform_sample: {e}")
            raise

    def __getitem__(self, idx):
        # 1) Load raw patch (dims: time, level, latitude, longitude)
        patch_ds: xr.Dataset = self.bgen[idx].load()

        # 2) Apply GenCast transforms
        patch_ds = self.transform_sample(patch_ds)

        lead = slice("12h", "12h")
        res = data_utils.extract_inputs_targets_forcings(patch_ds, target_lead_times=lead, **dataclasses.asdict(self.task_config))

        # lead = slice("12h", "360h")
        # res = data_utils.extract_inputs_targets_forcings(patch_ds, target_lead_times=lead, **dataclasses.asdict(self.task_config))
        return res



class ERA5Dataset_eval_GenCast(Dataset):
    def __init__(
        self,
        batch_generator: xbatcher.BatchGenerator,
        input_vars: list[str],
        target_var: str,
        task_config: None,
        time_interval: str = "12h",
        cache_size: int = 10,
    ):
        self.bgen = batch_generator
        self.input_vars = input_vars
        self.target_var = target_var
        self.time_interval = time_interval
        self.cache_size = cache_size
        self.cache = {}
        self.length = len(self.bgen)
        self.task_config = task_config

    def __len__(self):
        return self.length

    def transform_sample(self, sample_ds: xr.Dataset) -> xr.Dataset:
        """Transform a single sample to GenCast format."""
        start = time.time()
        try:
            # 1. Rename dims
            sample_ds = sample_ds.rename({"longitude": "lon", "latitude": "lat", "time": "datetime"})
            # 2. Subsample to 1° resolution & invert latitude
            sample_ds = (
                sample_ds
                .isel(lat=slice(0, None, 4), lon=slice(0, None, 4))
                .isel(lat=slice(None, None, -1))
            )
            # 3. Relative time coordinate
            delta = np.timedelta64(12, "h") if self.time_interval == "12h" else np.timedelta64(6, "h")
            n = len(sample_ds["datetime"])
            rel_times = np.arange(0, n * delta, delta, dtype="timedelta64[ns]")
            sample_ds = sample_ds.assign_coords(time=("datetime", rel_times))
            # 4. Preserve absolute datetime
            abs_times = sample_ds["datetime"].values[np.newaxis, :]
            sample_ds = sample_ds.swap_dims({"datetime": "time"}).drop_vars("datetime", errors="ignore")
            sample_ds = sample_ds.assign_coords(datetime=(("batch", "time"), abs_times))
            # 5. Add batch dimension
            for var in sample_ds.data_vars:
                if "batch" not in sample_ds[var].dims:
                    sample_ds[var] = sample_ds[var].expand_dims(batch=1)
            return sample_ds
        except Exception as e:
            print(f"Error in transform_sample: {e}")
            raise

    def __getitem__(self, idx):
        # 1) Load raw patch (dims: time, level, latitude, longitude)
        patch_ds: xr.Dataset = self.bgen[idx].load()

        # 2) Apply GenCast transforms
        patch_ds = self.transform_sample(patch_ds)

        # lead = slice("12h", "12h")
        # res = data_utils.extract_inputs_targets_forcings(patch_ds, target_lead_times=lead, **dataclasses.asdict(self.task_config))

        lead = slice("12h", "120h")
        res = data_utils.extract_inputs_targets_forcings(patch_ds, target_lead_times=lead, **dataclasses.asdict(self.task_config))
        return res



class ERA5Dataset(Dataset):
    def __init__(
        self,
        batch_generator: xbatcher.BatchGenerator,
        input_vars: list[str],
        target_var: str
    ):
        self.bgen = batch_generator
        self.input_vars = input_vars
        self.target_var = target_var
        self.length = len(self.bgen)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        patch_ds: xr.Dataset = self.bgen[idx].load()
        # patch_ds has dims: ("time", "level", "latitude", "longitude")
        levels = patch_ds.coords["level"].values  # array of length 13

        # 1) Build input stack with exactly those 13 levels for every variable
        input_arrays = []
        for var in self.input_vars:
            da = patch_ds[var]
            if "level" not in da.dims:
                # da is (time, lat, lon). Broadcast to (time, 13, lat, lon)
                # by repeating the same 2D slice along the level axis:
                da = xr.concat([da] * len(levels), dim="level")
                da = da.assign_coords(level=levels)
            # Now da dims = ("time", "level", "latitude", "longitude") with exactly 13 levels
            input_arrays.append(da)

        # Concatenate all inputs along a new "var_index" dimension
        input_stacked = xr.concat(input_arrays, dim="var_index")
        # dims = ("var_index","time","level","latitude","longitude")

        # Reorder so that NumPy array becomes (time, level, latitude, longitude, var_index)
        inp_np = input_stacked.transpose(
            "time", "level", "latitude", "longitude", "var_index"
        ).values
        # inp_np.shape = (time, 13, lat, lon, n_input_vars)

        # Reorder to (time, n_input_vars, level, lat, lon) for PyTorch
        x_np = inp_np.transpose(0, 4, 1, 2, 3)

        # 2) Build the target stack (always has level)
        da_t = patch_ds[self.target_var]
        if "level" not in da_t.dims:
            # If target is also surface‐only (unlikely if you chose a level variable),
            # broadcast similarly:
            da_t = xr.concat([da_t] * len(levels), dim="level")
            da_t = da_t.assign_coords(level=levels)
        tar_np = da_t.values  # shape = (time, 13, lat, lon)

        # Add channel dimension → (time, 1, 13, lat, lon)
        y_np = tar_np.reshape(
            tar_np.shape[0], 1, tar_np.shape[1], tar_np.shape[2], tar_np.shape[3]
        )

        x_tensor = torch.from_numpy(x_np).float()
        y_tensor = torch.from_numpy(y_np).float()
        return x_tensor, y_tensor




class ERA5BatchGenDataset_Mini(Dataset):
    def __init__(self, batch_generator: xbatcher.BatchGenerator):
        self.bgen = batch_generator
        self.length = len(self.bgen)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        patch_ds: xr.Dataset = self.bgen[idx].load()
        # -------------------------------
        # DEBUG (you already did this):
        # print("patch_ds dims:", patch_ds.dims)            # -> FrozenMapping({'time': 4, 'latitude': 721, 'longitude': 1440})
        # print("patch_ds data_vars:", list(patch_ds.data_vars))  # -> ['2m_temperature', '10m_v_component_of_wind']
        # -------------------------------

        # Stack the two data variables along a brand‐new dimension "var_index"
        stacked: xr.DataArray = patch_ds.to_stacked_array(
            new_dim="var_index",
            sample_dims=("time", "latitude", "longitude")
        )
        # Now stacked.dims == ("var_index", "time", "latitude", "longitude")

        # Reorder to (time, latitude, longitude, var_index) so we can .values() into a NumPy array
        all_np = stacked.transpose("time", "latitude", "longitude", "var_index").values
        # all_np.shape == (4, 721, 1440, 2)  since we had 2 data_vars

        # Split the last axis into inputs vs. targets
        x_np = all_np[..., 0]  # (4, 721, 1440) for "2m_temperature"
        y_np = all_np[..., 1]  # (4, 721, 1440) for "10m_v_component_of_wind"

        # Convert to torch.Tensor and add a channel dimension at index=1:
        # This yields (time, channel, lat, lon) = (4, 1, 721, 1440)
        x_tensor = torch.from_numpy(x_np).float().unsqueeze(1)
        y_tensor = torch.from_numpy(y_np).float().unsqueeze(1)

        return x_tensor, y_tensor


