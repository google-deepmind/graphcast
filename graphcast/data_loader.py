import xarray as xr
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import dask
from graphcast import data_utils
import dataclasses
from graphcast import graphcast
'''
        # zarr_path: str = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr',
        zarr_path: str = '/fs/nexus-projects/DeepOptic_Ev/16_WeatherControl/res_temp/debug_small_data.zarr',
'''

class ERA5DataLoader:
    def __init__(
        self,
        # zarr_path: str = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr',
        # zarr_path: str = '/fs/nexus-projects/DeepOptic_Ev/16_WeatherControl/res_temp/debug_small_data.zarr',
        zarr_path: str = '/fs/nexus-projects/DeepOptic_Ev/16_WeatherControl/small_dataset.zarr',
        time_interval: str = '12h',
        batch_size: int = 1,
        shuffle: bool = True,
        target_variables: Optional[Tuple[str, ...]] = None,
        cache_size: int = 10,  # Number of samples to keep in memory
        task_config: Optional[graphcast.TaskConfig] = None,
        target_dates: Optional[List[str]] = None  # Add target_dates parameter
    ):
        """
        Args:
            zarr_path: Path to zarr dataset
            time_interval: Time interval between samples ('6h' or '12h')
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle samples
            target_variables: List of variables to load. If None, use default GenCast variables
            cache_size: Number of processed samples to keep in memory
            task_config: GenCast task configuration
            target_dates: List of specific dates to use (format: 'YYYY-MM-DD')
        """
        print(f"\nInitializing ERA5DataLoader with target_dates: {target_dates}")
        # Open dataset lazily with dask and decode timedelta
        self.ds = xr.open_zarr(zarr_path, chunks={'time': 100}, decode_timedelta=True)
        self.time_interval = time_interval
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache_size = cache_size
        self.task_config = task_config or graphcast.TASK_13
        
        # Use default GenCast variables if none specified
        self.target_variables = target_variables or (
            "2m_temperature", "mean_sea_level_pressure", 
            "10m_v_component_of_wind", "10m_u_component_of_wind",
            "temperature", "geopotential", "u_component_of_wind",
            "v_component_of_wind", "vertical_velocity", "specific_humidity"
        )
        
        # Initialize cache
        self.cache = {}
        
        # Filter timestamps based on target_dates if provided
        if target_dates:
            self.target_dates = target_dates
            self.valid_indices = self._filter_timestamps_by_dates(target_dates)
            self.timestamps = self.ds.time.values[self.valid_indices]
        else:
            self.target_dates = None
            self.valid_indices = np.arange(len(self.ds.time))
            self.timestamps = self.ds.time.values
        
        self.num_samples = len(self.timestamps)
        print(f"Found {self.num_samples} valid samples for the specified dates")
        print(f"Valid timestamps: {self.timestamps}")
        
        # Create index list for shuffling
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        self.current_idx = 0

    def _filter_timestamps_by_dates(self, target_dates: List[str]) -> np.ndarray:
        """Filter timestamps to only include those matching target dates at 00:00:00."""
        # Convert target dates to numpy datetime64[D]
        target_dates_np = np.array(target_dates, dtype="datetime64[D]")

        # Convert full timestamp array to day‐precision
        dates = self.ds.time.values.astype("datetime64[D]")
        valid_indices = []

        for i, ts in enumerate(self.ds.time.values):
            if np.isin(dates[i], target_dates_np):
                # Only midnight entries
                if str(ts).endswith("T00:00:00.000000000"):
                    valid_indices.append(i)
        # print(valid_indices)
        # exit()
        if not valid_indices:
            raise ValueError(f"No valid timestamps found for dates {target_dates}")

        return np.array(sorted(valid_indices), dtype=int)

    # def _filter_timestamps_by_dates(self, target_dates: List[str]) -> np.ndarray:
    #     """Filter timestamps to only include those matching target dates at 00:00:00."""
    #     valid_indices = []
        
    #     # Convert target dates to numpy datetime64[D]
    #     target_dates_np = np.array(target_dates, dtype='datetime64[D]')
        
    #     # Convert timestamps to datetime64[D] for date comparison
    #     dates = self.ds.time.values.astype('datetime64[D]')
        
    #     # For each timestamp
    #     for i, (date, timestamp) in enumerate(zip(dates, self.ds.time.values)):
    #         # Check if the date matches any target date
    #         if date in target_dates_np:
    #             # Convert timestamp to string and check if it's midnight
    #             timestamp_str = str(timestamp)
    #             if timestamp_str.endswith('T00:00:00.000000000'):
    #                 valid_indices.append(i)
    #     print(valid_indices)
    #     exit()
    #     if not valid_indices:
    #         raise ValueError(f"No valid timestamps found for the provided target dates: {target_dates}")
        
    #     valid_indices = np.array(sorted(valid_indices))
    #     print(f"Found {len(valid_indices)} valid timestamps for dates: {target_dates}")
    #     return valid_indices

    def select_random_sample(self, era5_ori_data: xr.Dataset, sample_num: int = 32, 
                           time_interval: str = '12h', date_mode: str = 'specific', 
                           target_date: Optional[str] = None) -> xr.Dataset:
        """
        Selects a random consecutive subset of data from era5_ori_data with a fixed interval.

        Requirements:
        1. The original dataset has a 6-hour interval.
        2. The sampled dataset must have a 12-hour or 6-hour interval.
        3. The sampled data must be consecutive.
        4. The first sample must start at "T00:00:00.000000000".

        Args:
            era5_ori_data: The original ERA5 dataset
            sample_num: The number of samples to extract
            time_interval: Time interval between samples ('6h' or '12h')
            date_mode: Mode for selecting dates ('random' or 'specific')
            target_date: Specific date to select (format: 'YYYY-MM-DD')

        Returns:
            Selected subset of the dataset
        """
        # Get all datetime values as strings
        datetimes = era5_ori_data["datetime"].values.astype(str)
        
        if date_mode == 'random':
            # Find indices where the time is "T00:00:00.000000000"
            valid_start_indices = [i for i, dt in enumerate(datetimes) 
                                 if dt.endswith("T00:00:00.000000000")]

            # Ensure there are enough valid indices to select from
            max_valid_start_idx = len(valid_start_indices) - (sample_num // 2)
            if max_valid_start_idx <= 0:
                raise ValueError("Not enough valid starting points with 'T00:00:00.000000000' timestamps.")

            # Choose a random starting index from the valid ones
            start_idx = np.random.choice(valid_start_indices[:max_valid_start_idx])
        
        elif date_mode == 'specific':
            if not target_date or not isinstance(target_date, str) or len(target_date) != 10:
                raise ValueError("Invalid target date. Please provide a valid date string in 'YYYY-MM-DD' format.")
            
            valid_start_indices = [i for i, dt in enumerate(datetimes) 
                                 if dt.endswith("T00:00:00.000000000") and dt.startswith(target_date)]
            
            if not valid_start_indices:
                raise ValueError(f"No valid timestamps found for date {target_date}")
                
            start_idx = valid_start_indices[0]
        
        else:
            raise ValueError("Invalid date_mode. Please select 'random' or 'specific'.")

        # Select time steps based on interval
        if time_interval == '12h':
            # Select every second time step to get a 12-hour interval
            selected_data = era5_ori_data.isel(datetime=slice(start_idx, start_idx + sample_num * 2, 2))
        elif time_interval == '6h':
            # Select consecutive time steps for 6-hour interval
            selected_data = era5_ori_data.isel(datetime=slice(start_idx, start_idx + sample_num))
        else:
            raise ValueError("Invalid time interval. Please select '12h' or '6h'.")

        return selected_data

    def transform_sample(self, sample_ds: xr.Dataset) -> xr.Dataset:
        """Transform a single sample to GenCast format"""
        try:
            # 1. Rename dimensions to match target format
            sample_ds = sample_ds.rename({
                "longitude": "lon", 
                "latitude": "lat", 
                "time": "datetime"
            })

            # 2. Select consecutive time steps ensuring correct intervals
            sample_ds = self.select_random_sample(
                sample_ds, 
                time_interval=self.time_interval,
                date_mode='specific',  # We use specific mode for individual samples
                target_date=str(sample_ds.datetime.values[0].astype('datetime64[D]'))
            )

            # 3. Subsample lat/lon to 1-degree resolution (every 4th point)
            sample_ds = sample_ds.isel(
                lat=slice(0, None, 4), 
                lon=slice(0, None, 4)
            )

            # 3.1 Lat should be inversed
            sample_ds = sample_ds.isel(lat=slice(None, None, -1))

            # 4. Convert datetime to relative time
            if self.time_interval == '12h':
                time_values = np.arange(
                    0, 
                    len(sample_ds["datetime"]) * np.timedelta64(12, 'h'), 
                    np.timedelta64(12, 'h'), 
                    dtype='timedelta64[ns]'
                )
            else:  # '6h'
                time_values = np.arange(
                    0, 
                    len(sample_ds["datetime"]) * np.timedelta64(6, 'h'), 
                    np.timedelta64(6, 'h'), 
                    dtype='timedelta64[ns]'
                )

            # 5. Assign time coordinate and handle datetime
            sample_ds = sample_ds.assign_coords(time=("datetime", time_values))
            
            # Store absolute datetime values
            absolute_datetime_values = np.expand_dims(sample_ds["datetime"].values, axis=0)
            
            # Swap dimensions and handle datetime
            if "datetime" in sample_ds.dims:
                sample_ds = sample_ds.swap_dims({"datetime": "time"})
                sample_ds = sample_ds.drop_vars("datetime", errors="ignore")
            
            # Add datetime as a coordinate with batch dimension
            sample_ds = sample_ds.assign_coords(datetime=(("batch", "time"), absolute_datetime_values))
            
            # 6. Add batch dimension to variables
            for var in sample_ds.data_vars:
                if "batch" not in sample_ds[var].dims and var not in ["geopotential_at_surface", "land_sea_mask"]:
                    sample_ds[var] = sample_ds[var].expand_dims(batch=1)

            return sample_ds
        except Exception as e:
            print(f"Error in transform_sample: {e}")
            print(f"Sample dimensions: {sample_ds.dims}")
            print(f"Sample coordinates: {sample_ds.coords}")
            raise

    def get_sample(self, idx: int, model_mode: str = 'eval') -> xr.Dataset:
        """Get a single sample, using cache if available"""
        if idx in self.cache:
            sample = self.cache[idx]
            return sample

        # Get the timestamp for this index
        timestamp = self.timestamps[idx]
        
        # Calculate the time range for this sample
        if self.time_interval == '12h':
            if model_mode == 'train':
                end_time = timestamp + np.timedelta64(2 * 12, 'h')  # 32 timesteps for GenCast
            else:
                end_time = timestamp + np.timedelta64(32 * 12, 'h')  # 32 timesteps for GenCast
        else:  # '6h'
            end_time = timestamp + np.timedelta64(32 * 6, 'h')  # 64 timesteps for 6h interval
        
        # Load data for this time range
        try:

            sample = self.ds.sel(time=slice(timestamp, end_time))

            # Verify we have enough timesteps
            # if len(sample.time) < 32:
            #     raise ValueError(f"Not enough timesteps for sample at {timestamp}. Got {len(sample.time)}, need 32.")
            
            # Transform the sample
            sample = self.transform_sample(sample)

            # Update cache
            if len(self.cache) >= self.cache_size:
                # Remove oldest item
                self.cache.pop(next(iter(self.cache)))
            self.cache[idx] = sample
            
            return sample
        except Exception as e:
            print(f"Error loading sample at timestamp {timestamp}: {e}")
            raise

    def __len__(self) -> int:
        return self.num_samples

class ERA5TrainingLoader(ERA5DataLoader):
    def __init__(self, target_dates: List[str], *args, **kwargs):
        """
        Args:
            target_dates: List of dates to use for training (format: 'YYYY-MM-DD')
            *args, **kwargs: Arguments passed to ERA5DataLoader
        """
        print("\nInitializing ERA5TrainingLoader")
        super().__init__(target_dates=target_dates, *args, **kwargs)

        # create a psudo typhoon center trajectory. length = self.num_samples. Spatial Shape = self.ds.sizes['lat'], self.ds.sizes['lon']
        # so it is quite like a binary mask.
        # check spatial shape of self.ds 2m_temperature
        T = self.num_samples
        H = 181
        W = 360
        # no boolean dtype in xarray, so we use float
        traj = np.zeros((T, H, W), dtype=float)
        # 1) random start
        lat = np.random.randint(0, H)
        lon = np.random.randint(0, W)
        # 2) for each time step, mark & step
        for t in range(T):
            traj[t, lat, lon] = True
            # choose a random move in {-1, 0, +1} for each axis
            dlat = np.random.choice([-5, 0, 5])
            dlon = np.random.choice([-5, 0, 5])
            # update, clipping to [0, H-1] and [0, W-1]
            lat = np.clip(lat + dlat, 0, H-1)
            lon = np.clip(lon + dlon, 0, W-1)
        # store on the instance
        self.typhoon_center_trajectory = traj



        
    def get_random_batch(self) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        """Get a random batch of training data"""

        # Select random indices from our filtered timestamps
        batch_indices = np.random.choice(np.arange(self.num_samples), size=self.batch_size, replace=False)

        # Load batch samples and compute them immediately
        batch_samples = [self.get_sample(idx, 'train').compute() for idx in batch_indices]
        
        # Combine samples into batch
        batch = xr.concat(batch_samples, dim='batch')


        # 3) Inject fake typhoon trajectory as 2m_temperature mask
        #    self.typhoon_center_trajectory shape: (T, H, W)
        #    Extract for this batch: (batch, H, W)
        traj = self.typhoon_center_trajectory[batch_indices]  # boolean mask per sample
        # Expand to full time dimension (same mask at every timestep)
        time_len = batch.sizes['time']
        traj_expanded = np.repeat(traj[:, np.newaxis, :, :], time_len, axis=1)
        # Assign back into batch
        batch['2m_temperature'] = xr.DataArray(
            traj_expanded,
            coords={
                'batch': batch.batch,
                'time':  batch.time,
                'lat':   batch.lat,
                'lon':   batch.lon,
            },
            dims=('batch', 'time', 'lat', 'lon')
        )


        
        # For training, we want:
        # - inputs: 2 timesteps for context
        # - targets: 1 timestep for prediction
        # - forcings: 1 timestep matching target
        target_lead_times = slice("12h", "12h")  # Single timestep at 12h

        
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            batch,
            target_lead_times=target_lead_times,
            **dataclasses.asdict(self.task_config)
        )
        
        
        return inputs, targets, forcings

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

        # Get batch indices
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size

        # Load batch samples
        batch_samples = [self.get_sample(idx) for idx in batch_indices]
        
        # Combine samples into batch
        batch = xr.concat(batch_samples, dim='batch').compute()
        
        # Extract inputs, targets, and forcings using GenCast utilities
        # Use same configuration as get_random_batch
        target_lead_times = slice("12h", "12h")  # Single timestep at 12h
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            batch,
            target_lead_times=target_lead_times,
            **dataclasses.asdict(self.task_config)
        )

        return inputs, targets, forcings

class ERA5EvalLoader(ERA5DataLoader):
    def __init__(self, eval_dates: List[str], *args, **kwargs):
        """
        Args:
            eval_dates: List of dates to evaluate on (format: 'YYYY-MM-DD')
        """
        print("\nInitializing ERA5EvalLoader")
        super().__init__(target_dates=eval_dates, shuffle=False, *args, **kwargs)
        
    def get_eval_samples(self) -> Tuple[List[xr.Dataset], List[xr.Dataset], List[xr.Dataset]]:
        """Get samples for all evaluation dates"""
        print("\nGetting evaluation samples")
        eval_inputs = []
        eval_targets = []
        eval_forcings = []
        
        for idx in range(self.num_samples):
            try:
                # Get sample and compute immediately
                sample = self.get_sample(idx).compute()
                
                # Extract inputs, targets, and forcings
                num_timesteps = sample.sizes['time']  # Using sizes instead of dims
                target_lead_times = slice("12h", f"{(num_timesteps-2)*12}h")

                
                inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                    sample,
                    target_lead_times=target_lead_times,
                    **dataclasses.asdict(self.task_config)
                )
                 
                eval_inputs.append(inputs)
                eval_targets.append(targets)
                eval_forcings.append(forcings)
            except Exception as e:
                print(f"Error processing date {self.timestamps[idx]}: {e}")
                raise
            
        return eval_inputs, eval_targets, eval_forcings 