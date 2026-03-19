'''
        # zarr_path: str = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr',
        zarr_path: str = '/fs/nexus-projects/DeepOptic_Ev/16_WeatherControl/res_temp/debug_small_data.zarr',
        zarr_path= 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
'''

import xarray as xr
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
import dataclasses
from graphcast import graphcast, data_utils




class ERA5DataLoader:
    def __init__(
        self,
        # zarr_path: str = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr',
        zarr_path: str = '/fs/nexus-projects/DeepOptic_Ev/16_WeatherControl/res_temp/debug_small_data.zarr',
        time_interval: str = '12h',
        batch_size: int = 1,
        shuffle: bool = True,
        target_variables: Optional[Tuple[str, ...]] = None,
        cache_size: int = 10,
        task_config: Optional[graphcast.TaskConfig] = None,
        target_dates: Optional[List[str]] = None
    ):
        print(f"\nLoader 2 - Initializing ERA5DataLoader with target_dates: {target_dates}")
        # Open dataset lazily with dask
        self.ds = xr.open_zarr(zarr_path, chunks={'time': 10}, decode_timedelta=True)
        self.time_interval = time_interval
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache_size = cache_size
        self.task_config = task_config or graphcast.TASK_13
        self.target_variables = target_variables or (
            "2m_temperature", "mean_sea_level_pressure", 
            "10m_v_component_of_wind", "10m_u_component_of_wind",
            "temperature", "geopotential", "u_component_of_wind",
            "v_component_of_wind", "vertical_velocity", "specific_humidity"
        )
        self.cache = {}

        # Filter timestamps if dates provided
        if target_dates:
            self.target_dates = target_dates
            # self.valid_indices = self._filter_timestamps_by_dates(target_dates)
            self.valid_indices = self._filter_timestamps_by_dates_for_train(target_dates)
            self.timestamps = self.ds.time.values[self.valid_indices]

            # Take only those time positions
            ds_small = self.ds.isel(time=self.valid_indices)
            # 6) Persist that small slice into RAM
            self.ds = ds_small
            # And update timestamps to match the new, smaller ds
            self.timestamps = self.ds.time.values
            self.num_samples = len(self.timestamps)
            print(f"Dataset reduced to {self.num_samples} time‐steps and persisted in memory")
        else:
            self.target_dates = None
            self.valid_indices = np.arange(len(self.ds.time))
            self.timestamps = self.ds.time.values
            self.num_samples = len(self.timestamps)

        print(f"Found {self.num_samples} valid samples for the specified dates")
        print(f"Valid timestamps: {self.timestamps[:5]} ... {self.timestamps[-5:]}")

        # Prepare indices for iteration
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0


    def _filter_timestamps_by_dates_for_train(self, target_dates: List[str]) -> np.ndarray:
        """
        For each target date (YYYY-MM-DD), include all time‐steps
        from 2 days before up to 16 days after.
        Returns sorted array of integer indices into self.ds.time.
        """
        # 1) Parse the base dates at day precision
        base_dates = np.array(target_dates, dtype="datetime64[D]")

        # 2) Build (start, end) windows around each date
        windows = [
            (d - np.timedelta64(2, "D"), d + np.timedelta64(16, "D"))
            for d in base_dates
        ]

        # 3) Pull out the full time axis (datetime64[ns])
        times = self.ds.time.values  # e.g. hourly or 6-hourly stamps

        # 4) Aggregate a mask for any window
        mask = np.zeros(times.shape, dtype=bool)
        for start, end in windows:
            mask |= (times >= start) & (times <= end)

        # 5) Extract and sort the matching indices
        valid_indices = np.where(mask)[0]
        if valid_indices.size == 0:
            raise ValueError(f"No timestamps found in ±2D to +16D around {target_dates!r}")
        valid_indices = np.sort(valid_indices)

        print(f"Found {valid_indices.size} indices covering –2D to +16D around each target date")

        return valid_indices



    def _filter_timestamps_by_dates(self, target_dates: List[str]) -> np.ndarray:
        valid_indices = []
        target_dates_np = np.array(target_dates, dtype='datetime64[D]')
        dates = self.ds.time.values.astype('datetime64[D]')
        for i, (date, timestamp) in enumerate(zip(dates, self.ds.time.values)):
            if date in target_dates_np and str(timestamp).endswith('T00:00:00.000000000'):
                valid_indices.append(i)
        if not valid_indices:
            raise ValueError(f"No valid timestamps found for the provided target dates: {target_dates}")
        valid_indices = np.array(sorted(valid_indices))
        print(f"Found {len(valid_indices)} valid timestamps for dates: {target_dates}")
        return valid_indices

    def transform_sample(self, sample_ds: xr.Dataset) -> xr.Dataset:
        """Transform a single sample to GenCast format."""
        start = time.time()
        try:
            # 1. Rename spatial dims
            sample_ds = sample_ds.rename({"longitude": "lon", "latitude": "lat", "time": "datetime"})
            # 2. Subsample to 1-degree resolution & invert latitude
            sample_ds = sample_ds.isel(lat=slice(0, None, 4), lon=slice(0, None, 4)).isel(lat=slice(None, None, -1))
            # 3. Relative time coordinate
            delta = np.timedelta64(12, 'h') if self.time_interval == '12h' else np.timedelta64(6, 'h')
            n = len(sample_ds['datetime'])
            rel_times = np.arange(0, n * delta, delta, dtype='timedelta64[ns]')
            sample_ds = sample_ds.assign_coords(time=("datetime", rel_times))
            # 4. Preserve absolute datetime as batch coord
            abs_times = np.expand_dims(sample_ds['datetime'].values, axis=0)
            sample_ds = sample_ds.swap_dims({"datetime": "time"}).drop_vars("datetime", errors="ignore")
            sample_ds = sample_ds.assign_coords(datetime=(("batch", "time"), abs_times))
            # 5. Add batch dimension
            for var in sample_ds.data_vars:
                if "batch" not in sample_ds[var].dims:
                    sample_ds[var] = sample_ds[var].expand_dims(batch=1)
            elapsed = time.time() - start
            # print(f"transform_sample took {elapsed:.3f}s")
            return sample_ds
        except Exception as e:
            print(f"Error in transform_sample: {e}")
            raise


    def get_sample(self, idx: int, model_mode: str = 'eval') -> xr.Dataset:
        import time
        t0 = time.time()

        # 1) Compute integer start/end positions
        # start_i = self.valid_indices[idx]
        start_i = idx  # Use idx directly for simplicity
        # if self.time_interval == '12h':
        #     # train needs 3 frames (0, +12h, +24h); eval needs 33 (0…+384h)
        #     num_steps = (2 if model_mode=='train' else 32) + 1
        # else:
        #     # for 6h interval, always 33 steps (0…+192h)
        #     num_steps = 32 + 1
        # end_i = start_i + num_steps
        end_i = start_i + 3  # Always take 33 steps for simplicity

        # 2) Fast integer‐based slice (no date lookup)
        sample = self.ds.isel(time=slice(start_i, end_i))


    # def get_sample(self, idx: int, model_mode: str = 'eval') -> xr.Dataset:

    #     timestamp = self.timestamps[idx]

    #     # Check cache
    #     if idx in self.cache:
    #         print(f"[get_sample {timestamp}] Using cached sample (total: {time.time() - t0:.3f}s)")
    #         return self.cache[idx]

    #     # Determine end time
    #     if self.time_interval == '12h':
    #         end = timestamp + np.timedelta64((2 if model_mode == 'train' else 32) * 12, 'h')
    #     else:
    #         end = timestamp + np.timedelta64(32 * 6, 'h')

    #     # 1) Slice dataset
    #     sample = self.ds.sel(time=slice(timestamp, end))



        # 2) Transform sample
        sample = self.transform_sample(sample)

        # 3) Cache management
        if len(self.cache) >= self.cache_size:
            evicted = next(iter(self.cache))
            self.cache.pop(evicted)
        self.cache[idx] = sample

        return sample
        

    # def get_sample(self, idx: int, model_mode: str = 'eval') -> xr.Dataset:
    #     import time
    #     t0 = time.time()

    #     # Log and time-fetch
    #     timestamp = self.timestamps[idx]
    #     # print(f"Fetching sample for timestamp: {timestamp}")
    #     t0 = time.time()
    #     if idx in self.cache:
    #         print(f"Using cached sample for {timestamp}")
    #         return self.cache[idx]

    #     # Determine end time
    #     if self.time_interval == '12h':
    #         end = timestamp + np.timedelta64((2 if model_mode=='train' else 32) * 12, 'h')
    #     else:
    #         end = timestamp + np.timedelta64(32 * 6, 'h')

    #     # Slice dataset
    #     sel_start = time.time()
    #     sample = self.ds.sel(time=slice(timestamp, end))
    #     # print(f"sel() took {time.time() - sel_start:.3f}s")

    #     # Transform
    #     sample = self.transform_sample(sample)

    #     # Cache management
    #     if len(self.cache) >= self.cache_size:
    #         self.cache.pop(next(iter(self.cache)))
    #     self.cache[idx] = sample
    #     total_elapsed = time.time() - t0
    #     # print(f"get_sample total for {timestamp}: {total_elapsed:.3f}s")
    #     return sample

    def __len__(self) -> int:
        return self.num_samples

class ERA5TrainingLoader(ERA5DataLoader):
    def __init__(self, target_dates: Optional[List[str]] = None, *args, **kwargs):
        print("\nInitializing ERA5TrainingLoader")
        super().__init__(target_dates=target_dates, *args, **kwargs)

    def get_random_batch(self) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        import time
        t_start = time.time()
        batch_inds = np.random.choice(self.num_samples, size=self.batch_size, replace=False)
        # samples = [self.get_sample(i, 'train').compute() for i in batch_inds]
        lazy_samples = [self.get_sample(i, 'train') for i in batch_inds]
        lazy_batch = xr.concat(lazy_samples, dim="batch")

        # batch = xr.concat(samples, dim='batch')
        batch = lazy_batch.compute()
        # ----- INSERT PROFILING WRAPPER HERE -----

        # Normal processing
        lead = slice("12h", "12h")
        res = data_utils.extract_inputs_targets_forcings(batch, target_lead_times=lead, **dataclasses.asdict(self.task_config))
        return res

        # Dataloader Testing
        # res = None, None, None
        # return res

    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        if self.current_idx >= self.num_samples:
            if self.shuffle:
                np.random.shuffle(self.indices)
            self.current_idx = 0
            raise StopIteration
        inds = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        samples = [self.get_sample(i).compute() for i in inds]
        batch = xr.concat(samples, dim='batch')
        lead = slice("12h", "12h")
        return data_utils.extract_inputs_targets_forcings(batch, target_lead_times=lead, **dataclasses.asdict(self.task_config))

class ERA5EvalLoader(ERA5DataLoader):
    def __init__(self, eval_dates: List[str], *args, **kwargs):
        print("\nInitializing ERA5EvalLoader")
        super().__init__(target_dates=eval_dates, shuffle=False, *args, **kwargs)

    def get_eval_samples(self) -> Tuple[List[xr.Dataset], List[xr.Dataset], List[xr.Dataset]]:
        print("\nGetting evaluation samples")
        inputs_list, targets_list, forcings_list = [], [], []
        for i in range(self.num_samples):
            sample = self.get_sample(i).compute()
            nts = sample.sizes['time']
            lead = slice("12h", f"{(nts-2)*12}h")
            inp, tgt, frc = data_utils.extract_inputs_targets_forcings(sample, target_lead_times=lead, **dataclasses.asdict(self.task_config))
            inputs_list.append(inp)
            targets_list.append(tgt)
            forcings_list.append(frc)
        return inputs_list, targets_list, forcings_list
