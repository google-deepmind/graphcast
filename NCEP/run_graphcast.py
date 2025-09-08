'''
Description: Script to call the graphcast model using gdas products
Author: Sadegh Sadeghi Tabas (sadegh.tabas@noaa.gov)
Revision history:
    -20240307: Sadegh Tabas, initial code
    -20250506: Linlin Cui, moved hard coded model weights to a json file
'''
import os
import argparse
from datetime import timedelta
import dataclasses
import functools
import re
import haiku as hk
import jax
import numpy as np
import xarray
import boto3
import pandas as pd
import pickle

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout

from utils.nc2grib import Netcdf2Grib

class GraphCastModel:
    def __init__(
        self, 
        pretrained_model_path, 
        gdas_data_path, 
        case_name: str, 
        config_file = None, 
        output_dir = None, 
        num_pressure_levels = 13, 
        forecast_length = 64,
    ):
        self.pretrained_model_path = pretrained_model_path
        self.gdas_data_path = gdas_data_path
        self.forecast_length = forecast_length
        self.case_name = case_name
        self.num_pressure_levels = num_pressure_levels
        self.config_file_path = config_file
        
        if output_dir is None:
            self.output_dir = os.getcwd()
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.params = None
        self.state = {}
        self.model_config = None
        self.task_config = None
        self.diffs_stddev_by_level = None
        self.mean_by_level = None
        self.stddev_by_level = None
        self.current_batch = None
        self.inputs = None
        self.targets = None
        self.forcings = None
        self.s3_bucket_name = "noaa-nws-graphcastgfs-pds"
        self.dates = None
        

    def load_pretrained_model(self):
        """Load pre-trained GraphCast model."""
        if self.num_pressure_levels==13:
            model_weights_path = f"{self.pretrained_model_path}/params/GCGFSv2_finetuned - GDAS - ERA5 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz"
        else:
            model_weights_path = f"{self.pretrained_model_path}/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"

        with open(model_weights_path, "rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)
            #self.params = ckpt.params
            self.state = {}
            self.model_config = ckpt.model_config
            self.task_config = ckpt.task_config

            #update params
            if self.config_file_path is not None:
                with open(self.config_file_path, 'rb') as f:
                    self.params = pickle.load(f)
            else:
                self.params = ckpt.params

    def load_gdas_data(self):
        """Load GDAS data."""
        #with open(gdas_data_path, "rb") as f:
        #    self.current_batch = xarray.load_dataset(f).compute()
        self.current_batch = xarray.load_dataset(self.gdas_data_path).compute()
        self.dates =  pd.to_datetime(self.current_batch.datetime.values)
        
        if (self.forecast_length + 2) > len(self.current_batch['time']):
            print('Updating batch dataset to account for forecast length')
            
            diff = int(self.forecast_length + 2 - len(self.current_batch['time']))
            ds = self.current_batch

            # time and datetime update
            curr_time_range = ds['time'].values.astype('timedelta64[ns]')
            new_time_range = (np.arange(len(curr_time_range) + diff) * np.timedelta64(6, 'h')).astype('timedelta64[ns]')
            ds = ds.reindex(time = new_time_range)
            curr_datetime_range = ds['datetime'][0].values.astype('datetime64[ns]')
            new_datetime_range = curr_datetime_range[0] + np.arange(len(curr_time_range) + diff) * np.timedelta64(6, 'h')
            ds['datetime'][0]= new_datetime_range

            self.current_batch = ds
            print('batch dataset updated')
            
        
    def extract_inputs_targets_forcings(self):
        """Extract inputs, targets, and forcings from the loaded data."""
        self.inputs, self.targets, self.forcings = data_utils.extract_inputs_targets_forcings(
            self.current_batch, target_lead_times=slice("6h", f"{self.forecast_length*6}h"), **dataclasses.asdict(self.task_config)
        )

    def load_normalization_stats(self):
        """Load normalization stats."""
        
        diffs_stddev_path = f"{self.pretrained_model_path}/stats/diffs_stddev_by_level.nc"
        mean_path = f"{self.pretrained_model_path}/stats/mean_by_level.nc"
        stddev_path = f"{self.pretrained_model_path}/stats/stddev_by_level.nc"
        
        with open(diffs_stddev_path, "rb") as f:
            self.diffs_stddev_by_level = xarray.load_dataset(f).compute()
        with open(mean_path, "rb") as f:
            self.mean_by_level = xarray.load_dataset(f).compute()
        with open(stddev_path, "rb") as f:
            self.stddev_by_level = xarray.load_dataset(f).compute()
    
    # Jax doesn't seem to like passing configs as args through the jit. Passing it
    # in via partial (instead of capture by closure) forces jax to invalidate the
    # jit cache if you change configs.
    def _with_configs(self, fn):
        return functools.partial(fn, model_config=self.model_config, task_config=self.task_config,)

    # Always pass params and state, so the usage below are simpler
    def _with_params(self, fn):
        return functools.partial(fn, params=self.params, state=self.state)

    # Deepmind models aren't stateful, so the state is always empty, so just return the
    # predictions. This is requiredy by the rollout code, and generally simpler.
    @staticmethod
    def _drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    def load_model(self):
        def construct_wrapped_graphcast(model_config, task_config):
            """Constructs and wraps the GraphCast Predictor."""
            # Deeper one-step predictor.
            predictor = graphcast.GraphCast(model_config, task_config)

            # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
            # from/to float32 to/from BFloat16.
            predictor = casting.Bfloat16Cast(predictor)

            # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
            # BFloat16 happens after applying normalization to the inputs/targets.
            predictor = normalization.InputsAndResiduals(predictor, diffs_stddev_by_level=self.diffs_stddev_by_level, mean_by_level=self.mean_by_level, stddev_by_level=self.stddev_by_level,)

            # Wraps everything so the one-step model can produce trajectories.
            predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True,)
            return predictor

        @hk.transform_with_state
        def run_forward(model_config, task_config, inputs, targets_template, forcings,):
            predictor = construct_wrapped_graphcast(model_config, task_config)
            return predictor(inputs, targets_template=targets_template, forcings=forcings,)
        
        jax.jit(self._with_configs(run_forward.init))
        self.model = self._drop_state(self._with_params(jax.jit(self._with_configs(run_forward.apply))))
    
 
    def get_predictions(self):
        """Run GraphCast and save forecasts to a NetCDF file."""

        print (f"start running GraphCast for {self.forecast_length} steps --> {self.forecast_length*6} hours.")
        self.load_model()
           
        # Call and save f000 in grib2
        ds = self.current_batch
        ds = ds.drop_vars(['geopotential_at_surface','land_sea_mask', 'total_precipitation_6hr'])
        for var in ds.data_vars:
            if 'long_name' in ds[var].attrs:
                del ds[var].attrs['long_name']
        ds = ds.isel(time=slice(1, 2))
        ds['time'] = ds['time'] - pd.Timedelta(hours=6)

        converter = Netcdf2Grib(self.dates[0][1], case_name=self.case_name)
        converter.save_grib2(ds, self.output_dir)

        rollout.chunked_prediction(
            self.output_dir, 
            converter, 
            self.model, 
            rng=jax.random.PRNGKey(0), 
            inputs=self.inputs, 
            targets_template=self.targets * np.nan, 
            forcings=self.forcings,
        )

    def upload_to_s3(self, keep_data):
        s3 = boto3.client('s3')
        
        # Extract date and time information from the input file name
        input_file_name = os.path.basename(self.gdas_data_path)
        
        date_start = input_file_name.find("date-")

        # Check if "date-" is found in the input_file_name
        if date_start != -1:
            date_start += len("date-")  # Move to the end of "date-"
            date = input_file_name[date_start:date_start + 8]  # Extract 8 characters as the date
        
            time_start = date_start + 8  # Move to the character after the date
            time = input_file_name[time_start:time_start + 2]  # Extract 2 characters as the time


        # Define S3 key paths for input and output files
        input_s3_key = f'graphcastgfs.{date}/{time}/input/{self.gdas_data_path}'

        # Upload input file to S3
        s3.upload_file(self.gdas_data_path, self.s3_bucket_name, input_s3_key)
        
        # Upload output files to S3
        # Iterate over all files in the local directory and upload each one to S3
        s3_prefix = f'graphcastgfs.{date}/{time}/forecasts_{self.num_pressure_levels}_levels'
        
        for root, dirs, files in os.walk(self.output_dir):

            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, self.output_dir)
                s3_path = os.path.join(s3_prefix, relative_path)
            
                # Upload the file
                s3.upload_file(local_path, self.s3_bucket_name, s3_path)

        print("Upload to s3 bucket completed.")

        # Delete local files if keep_data is False
        if not keep_data:
            # Remove forecast data from the specified directory
            print("Removing input and forecast data from the specified directory...")
            try:
                os.system(f"rm -rf {self.output_dir}")
                os.remove(self.gdas_data_path)
                print("Local input and output files deleted.")
            except Exception as e:
                print(f"Error removing input and forecast data: {str(e)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GraphCast model.")
    parser.add_argument("-i", "--input", help="input file path (including file name)", required=True)
    parser.add_argument("-w", "--weights", help="parent directory of the graphcast params and stats", required=True)
    parser.add_argument("-l", "--length", help="length of forecast (6-hourly), an integer number in range [1, 40]", required=True)
    parser.add_argument("-n", "--case_name", help="gifs, or gefs member [aigec00, aigep01, ..., aigep30]", required=True)
    #parser.add_argument("-c", "--config", help="GC weight member file", required=True)
    parser.add_argument("-c", "--config", help="GC weight member file", default=None)
    parser.add_argument("-o", "--output", help="output directory", default=None)
    parser.add_argument("-p", "--pressure", help="number of pressure levels", default=13)
    parser.add_argument("-u", "--upload", help="upload input data as well as forecasts to noaa s3 bucket (yes or no)", default = "no")
    parser.add_argument("-k", "--keep", help="keep input and output after uploading to noaa s3 bucket (yes or no)", default = "no")
    
    args = parser.parse_args()
    runner = GraphCastModel(args.weights, args.input, args.case_name, args.config, args.output, int(args.pressure), int(args.length))
    
    runner.load_pretrained_model()
    runner.load_gdas_data()
    runner.extract_inputs_targets_forcings()
    runner.load_normalization_stats()
    runner.get_predictions()
    
    upload_data = args.upload.lower() == "yes"
    keep_data = args.keep.lower() == "yes"
    
    if upload_data:
        runner.upload_to_s3(keep_data)
