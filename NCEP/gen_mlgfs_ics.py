'''
Description
@uthor: Sadegh Sadeghi Tabas (sadegh.tabas@noaa.gov)
Revision history:
    -20231010: Sadegh Tabas, initial code
    -20231204: Sadegh Tabas, calculating toa incident solar radiation, parallelizing, updating units, and resolving memory issues
    -20240112: Sadegh Tabas, (i)removing Pysolar as tisr would be calc through GC, (ii) add NOMADS option for downloading data, (iii) add 37 pressure levels, (iv) configurations for hera
    -20240124: Linlin Cui, added pygrib method to extract variables from grib2 files
    -20240205: Sadegh Tabas, add 37 pressure levels, update s3 bucket
    -20240214: Linlin Cui, update pygrib method to account for 37 pressure levels
    -20240221: Sadegh Tabas, (i) updated acc precip variable IC, (ii) initialize s3 credentials for cloud machines, (iii) updated wgrib2 process, pygrib process, s3 and nomads functions
    -20240425: Sadegh Tabas, (i) update s3 bucket resource, 
'''
import os
import sys
from time import time
import glob
import argparse
import subprocess
from datetime import datetime, timedelta
import re
import boto3
import xarray as xr
import numpy as np
from botocore.config import Config
from botocore import UNSIGNED
#import pygrib
import grib2io
import requests

class GFSDataProcessor:
    def __init__(self, start_datetime, end_datetime, num_pressure_levels=13, download_source='nomads', output_directory=None, download_directory=None, keep_downloaded_data=True, aws=None):
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.num_levels = num_pressure_levels
        self.download_source = download_source
        self.output_directory = output_directory
        self.download_directory = download_directory
        self.keep_downloaded_data = keep_downloaded_data
        self.cycle = self.end_datetime.hour

        if self.download_source == 's3':
            self.s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
        # Specify the S3 bucket name and root directory
        self.bucket_name = 'noaa-gfs-bdp-pds'
        
        self.root_directory = 'gdas'

        # Specify the local directory where you want to save the downloaded files
        if self.download_directory is None:
            self.local_base_directory = os.path.join(os.getcwd(), self.bucket_name+'_'+str(self.num_levels))  # Use current directory if not specified
        else:
            self.local_base_directory = os.path.join(self.download_directory, self.bucket_name+'_'+str(self.num_levels))

        # Specify the output directory where you want to save the processed files
        if self.output_directory is None:
            self.output_directory = os.getcwd()  # Use current directory if not specified
        os.makedirs(self.output_directory, exist_ok=True)
        self.output_netcdf = os.path.join(self.output_directory, f"aigfs.t{self.cycle:02d}z.ic.nc")

        # List of file formats to download
        if self.num_levels == 13:     
            self.file_formats = ['pgrb2.0p25.f000', 'pgrb2.0p25.f006'] # , '0p25.f001'
        else:
            self.file_formats = ['pgrb2.0p25.f000', 'pgrb2b.0p25.f000', 'pgrb2.0p25.f006'] # , '0p25.f001'
    
    def s3bucket(self, date_str, time_str, local_directory):
        # Construct the S3 prefix for the directory
        s3_prefix = f"{self.root_directory}.{date_str}/{time_str}/"

        # get prefix for precip from the previous cycle
        # Convert date_str and time_str to datetime object
        datetime_obj = datetime.strptime(date_str + time_str, "%Y%m%d%H")

        # Get the datetime 6 hours before
        datetime_before = datetime_obj - timedelta(hours=6)

        # Get the date string and time string from datetime objects
        date_str_precip = datetime_before.strftime("%Y%m%d")
        time_str_precip = datetime_before.strftime("%H")

        # Construct the S3 prefix for the directory
        s3_prefix_precip = f"{self.root_directory}.{date_str_precip}/{time_str_precip}/"

        def get_data(s3_prefix, file_format, local_directory):
            # List objects in the S3 directory
            s3_objects = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=s3_prefix)
            for obj in s3_objects.get('Contents', []):
                obj_key = obj['Key']
                if obj_key.endswith(f'.{file_format}'):
                    # Define the local file path
                    local_file_path = os.path.join(local_directory, os.path.basename(obj_key))

                    # Download the file from S3 to the local path
                    self.s3.download_file(self.bucket_name, obj_key, local_file_path)
                    print(f"Downloaded {obj_key} to {local_file_path}")

                 
        for file_format in self.file_formats:
            if file_format !='pgrb2.0p25.f006':
                get_data(s3_prefix, file_format, local_directory)
            else:
                get_data(s3_prefix_precip, file_format, local_directory)

        
    
    def nomads(self, date_str, time_str, local_directory):

        # Convert date_str and time_str to datetime object
        datetime_obj = datetime.strptime(date_str + time_str, "%Y%m%d%H")

        # Get the datetime 6 hours before
        datetime_before = datetime_obj - timedelta(hours=6)

        # Get the date string and time string from datetime objects
        date_str_precip = datetime_before.strftime("%Y%m%d")
        time_str_precip = datetime_before.strftime("%H")
        
        def get_data(date_str, time_str, file_format, local_directory):
            # Construct the URL for the data directory
            gdas_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/{self.root_directory}.{date_str}/{time_str}/atmos/"
            
            # Get the list of files from the URL
            response = requests.get(gdas_url)
            if response.status_code == 200:
                # Parse the HTML content using BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all anchor tags (links) in the HTML
                anchor_tags = soup.find_all('a')
                
                # Extract file URLs from href attributes of anchor tags
                file_urls = [gdas_url + tag['href'] for tag in anchor_tags if tag.get('href')]
    
                for file_url in file_urls: 
                    
                    if file_url.endswith(f'.{file_format}'):
                        
                        # Define the local file path
                        local_file_path = os.path.join(local_directory, os.path.basename(file_url))
                        
                        # Download the file from S3 to the local path
                        try:
                            # Run the wget command
                            subprocess.run(['wget', file_url, '-O', local_file_path], check=True)
                            print(f"Download completed: {file_url} => {local_file_path}")
                        except subprocess.CalledProcessError as e:
                            print(f"Error downloading {file_url}: {e}")

        for file_format in self.file_formats:
            if file_format !='pgrb2.0p25.f006':
                get_data(date_str, time_str, file_format, local_directory)
            else:
                get_data(date_str_precip, time_str_precip, file_format, local_directory)


    
        
    def download_data(self):
        # Calculate the number of 6-hour intervals
        delta = (self.end_datetime - self.start_datetime)
        total_intervals = int(delta.total_seconds() / 3600 / 6)  # 6 hours per interval

        # Loop through the 6-hour intervals
        current_datetime = self.start_datetime
        while current_datetime <= self.end_datetime:
            date_str = current_datetime.strftime("%Y%m%d")
            time_str = current_datetime.strftime("%H")
            
            # Define the local directory path where the file will be saved
            local_directory = os.path.join(self.local_base_directory, date_str, time_str)

            # Create the local directory if it doesn't exist
            os.makedirs(local_directory, exist_ok=True)
            
            if self.download_source == 's3':
                self.s3bucket(date_str, time_str, local_directory)
            else:
                self.nomads(date_str, time_str, local_directory)
                
            

            # Move to the next 6-hour interval
            current_datetime += timedelta(hours=6)

        print("Download completed.")

    def process_data_with_wgrib2(self):
        # Define the directory where your GRIB2 files are located
        data_directory = self.local_base_directory

        # Create a dictionary to specify the variables, levels, and whether to extract only the first time step (if needed)
        variables_to_extract = {
            '.pgrb2.0p25.f000': {
                ':HGT:': {
                    'levels': [':surface:'],
                    'first_time_step_only': True,  # Extract only the first time step
                },
                ':TMP:': {
                    'levels': [':2 m above ground:'],
                },
                ':PRMSL:': {
                    'levels': [':mean sea level:'],
                },
                ':VGRD|UGRD:': {
                    'levels': [':10 m above ground:'],
                },
                ':SPFH|VVEL|VGRD|UGRD|HGT|TMP:': {
                    'levels': [':(50|100|150|200|250|300|400|500|600|700|850|925|1000) mb:'],
                },
            },
            '.pgrb2.0p25.f006': {
                ':LAND:': {
                    'levels': [':surface:'],
                    'first_time_step_only': True,  # Extract only the first time step
                },
                '^(597):': {  # APCP
                    'levels': [':surface:'],
                },
            }
        }
        if self.num_levels == 37:
            variables_to_extract['.pgrb2.0p25.f000'][':SPFH|VVEL|VGRD|UGRD|HGT|TMP:']['levels'] = [':(1|2|3|5|7|10|20|30|50|70|100|150|200|250|300|350|400|450|500|550|600|650|700|750|800|850|900|925|950|975|1000) mb:']
            variables_to_extract['.pgrb2b.0p25.f000'] = {}
            variables_to_extract['.pgrb2b.0p25.f000'][':SPFH|VVEL|VGRD|UGRD|HGT|TMP:'] = {}
            variables_to_extract['.pgrb2b.0p25.f000'][':SPFH|VVEL|VGRD|UGRD|HGT|TMP:']['levels'] = [':(125|175|225|775|825|875) mb:']
       
        # Create an empty list to store the extracted datasets
        extracted_datasets = []
        files = []
        print("Start extracting variables and associated levels from grib2 files:")
        # Loop through each folder (e.g., gdas.yyyymmdd)
        date_folders = sorted(next(os.walk(data_directory))[1])
        for date_folder in date_folders:
            date_folder_path = os.path.join(data_directory, date_folder)

            # Loop through each hour (e.g., '00', '06', '12', '18')
            for hour in ['00', '06', '12', '18']:
                subfolder_path = os.path.join(date_folder_path, hour)

                # Check if the subfolder exists before processing
                if os.path.exists(subfolder_path):
                    # Loop through each GRIB2 file (.f000, .f001, .f006)
                    for file_extension, variable_data in variables_to_extract.items():
                        for variable, data in variable_data.items():
                            levels = data['levels']
                            first_time_step_only = data.get('first_time_step_only', False)  # Default to False if not specified

                            pattern = os.path.join(subfolder_path, f'gdas.t*z{file_extension}')
                            # Use glob to search for files matching the pattern
                            matching_files = glob.glob(pattern)
                            
                            # Check if there's exactly one matching file
                            if len(matching_files) == 1:
                                grib2_file = matching_files[0]
                                print("Found file:", grib2_file)
                            else:
                                print("Error: Found multiple or no matching files.")
                                
                            # Extract the specified variables with levels from the GRIB2 file
                            for level in levels:
                                output_file = f'{variable}_{level}_{date_folder}_{hour}{file_extension}_{self.num_levels}.nc'
                                files.append(output_file)
                                
                                # Extracting levels using regular expression
                                matches = re.findall(r'\d+', level)
                                
                                # Convert the extracted matches to integers
                                curr_levels = [int(match) for match in matches]
                                
                                # Get the number of levels
                                number_of_levels = len(curr_levels)
                                
                                # Use wgrib2 to extract the variable with level
                                wgrib2_command = ['wgrib2', '-nc_nlev', f'{number_of_levels}', grib2_file, '-match', f'{variable}', '-match', f'{level}', '-netcdf', output_file]
                                subprocess.run(wgrib2_command, check=True)

                                # Open the extracted netcdf file as an xarray dataset
                                ds = xr.open_dataset(output_file)

                                # if variable == '^(597):':
                                #    ds['time'] = ds['time'] - np.timedelta64(6, 'h')

                                # If specified, extract only the first time step
                                if variable not in [':LAND:', ':HGT:']:
                                    extracted_datasets.append(ds)
                                else:
                                    if first_time_step_only:
                                        # Append the dataset to the list
                                        ds = ds.isel(time=0)
                                        extracted_datasets.append(ds)
                                        variables_to_extract[file_extension][variable]['first_time_step_only'] = False
                                
                                # Optionally, remove the intermediate GRIB2 file
                                # os.remove(output_file)
        print("Merging grib2 files:")
        ds = xr.merge(extracted_datasets)
        
        print("Merging process completed.")
        
        print("Processing, Renaming and Reshaping the data")
        # Drop the 'level' dimension
        ds = ds.drop_dims('level')

        # Rename variables and dimensions
        ds = ds.rename({
            'latitude': 'lat',
            'longitude': 'lon',
            'plevel': 'level',
            'HGT_surface': 'geopotential_at_surface',
            'LAND_surface': 'land_sea_mask',
            'PRMSL_meansealevel': 'mean_sea_level_pressure',
            'TMP_2maboveground': '2m_temperature',
            'UGRD_10maboveground': '10m_u_component_of_wind',
            'VGRD_10maboveground': '10m_v_component_of_wind',
            'APCP_surface': 'total_precipitation_6hr',
            'HGT': 'geopotential',
            'TMP': 'temperature',
            'SPFH': 'specific_humidity',
            'VVEL': 'vertical_velocity',
            'UGRD': 'u_component_of_wind',
            'VGRD': 'v_component_of_wind'
        })

        # Assign 'datetime' as coordinates
        ds = ds.assign_coords(datetime=ds.time)
        
        # Convert data types
        ds['lat'] = ds['lat'].astype('float32')
        ds['lon'] = ds['lon'].astype('float32')
        ds['level'] = ds['level'].astype('int32')

        # Adjust time values relative to the first time step
        ds['time'] = ds['time'] - ds.time[0]

        # Expand dimensions
        ds = ds.expand_dims(dim='batch')
        ds['datetime'] = ds['datetime'].expand_dims(dim='batch')

        # Squeeze dimensions
        ds['geopotential_at_surface'] = ds['geopotential_at_surface'].squeeze('batch')
        ds['land_sea_mask'] = ds['land_sea_mask'].squeeze('batch')

        # Update geopotential unit to m2/s2 by multiplying 9.80665
        ds['geopotential_at_surface'] = ds['geopotential_at_surface'] * 9.80665
        ds['geopotential'] = ds['geopotential'] * 9.80665

        # Update total_precipitation_6hr unit to (m) from (kg/m^2) by dividing it by 1000kg/m³
        ds['total_precipitation_6hr'] = ds['total_precipitation_6hr'] / 1000
        
        # Define the output NetCDF file
        date = (self.start_datetime + timedelta(hours=6)).strftime('%Y%m%d%H')
        steps = str(len(ds['time']))

        # Save the merged dataset as a NetCDF file
        ds.to_netcdf(self.output_netcdf)
        for file in files:
            os.remove(file)
            
        # Optionally, remove downloaded data
        if not self.keep_downloaded_data:
            self.remove_downloaded_data()

        print(f"Process completed successfully, your inputs for GraphCast model generated at:\n {self.output_netcdf}")

    def process_data_with_grib2io(self):
        # Define the directory where your GRIB2 files are located
        data_directory = self.local_base_directory

        #Get time-varying variables
        variables_to_extract = {
            '.pgrb2.0p25.f000': {
                'TMP': {
                    'level': ['2 m above ground'],
                },
                'PRMSL': {
                    'level': ['mean sea level'],
                },
                'UGRD, VGRD': {
                    'level': ['10 m above ground'],
                },
                'SPFH, VVEL, VGRD, UGRD, HGT, TMP': {
                    'level': [
                        '50 mb', '100 mb', '150 mb', '200 mb', '250 mb', 
                        '300 mb', '400 mb', '500 mb', '600 mb', '700 mb', 
                        '850 mb', '925 mb', '1000 mb'
                    ],
                },
            },
            '.pgrb2.0p25.f006': {
                'APCP': {  # total precipitation 
                    'level': ['surface'],
                },
            }
        }

        if self.num_levels == 37:
            variables_to_extract['.pgrb2.0p25.f000']['w, u, v, q, t, gh']['level'] = [
                1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 
                100, 150, 200, 250, 300, 350, 400,
                450, 500, 550, 600, 650, 700, 750,
                800, 850, 900, 925, 950, 975, 1000,
            ]
            extra_levels = [125, 175, 225, 775, 825, 875]
            file_extension_2b = '.pgrb2b.0p25.f000'

        # Create an empty list to store the extracted datasets
        mergeDSs = []
        print("Start extracting variables and associated levels from grib2 files:")
        # Loop through each folder (e.g., gdas.yyyymmdd)
        date_folders = sorted(next(os.walk(data_directory))[1])
        for date_folder in date_folders:
            date_folder_path = os.path.join(data_directory, date_folder)

            # Loop through each hour (e.g., '00', '06', '12', '18')
            for hour in ['00', '06', '12', '18']:
                subfolder_path = os.path.join(date_folder_path, hour)

                # Check if the subfolder exists before processing
                if os.path.exists(subfolder_path):

                    mergeDAs = []

                    for file_extension, variables in variables_to_extract.items():
                        pattern = os.path.join(subfolder_path, f'gdas.t*z{file_extension}')
                        # Use glob to search for files matching the pattern
                        matching_files = glob.glob(pattern)
                        
                        # Check if there's exactly one matching file
                        if len(matching_files) == 1:
                            fname = matching_files[0]
                            print("Found file:", fname)
                        else:
                            print("Error: Found multiple or no matching files.")

                        #open grib file
                        grbs = grib2io.open(fname)

                        for key, value in variables.items():

                            variable_names = key.split(', ')
                            #levelType = value['typeOfLevel']
                            desired_level = value['level']
                    
                            for var_name in variable_names:

                                print(f'Get variable {var_name} from file {fname}:')
                                if len(desired_level) > 1:
                                    da = self.get_dataarray_3d(grbs, var_name, desired_level)
                                else:
                                    da = self.get_dataarray(grbs, var_name, desired_level[0])
                                mergeDAs.append(da)

                                ##extract variables from pgrb2b
                                #if (levelType == 'isobaricInhPa') & (self.num_levels == 37):
                                #    fname2b = os.path.join(subfolder_path, f'gdas.t{hour}z{file_extension_2b}')
                                #    grbs2b = pygrib.open(fname2b)
                                #    da_extra = self.get_dataarray(grbs2b, var_name, levelType, extra_levels)
                                #    da_combined = da.combine_first(da_extra) 
                                #    mergeDAs.append(da_combined)
                                #else:
                                #    mergeDAs.append(da)

                    ds = xr.merge(mergeDAs)

                    mergeDSs.append(ds)
                    ds.close()

        #Concatenate ds
        ds = xr.concat(mergeDSs, dim='time')

        #Get 2D static variables
        grbfiles = glob.glob(f'{data_directory}/*/*/*.f000')
        grbfiles.sort()
        #Get lsm/orog from the first file
        grbs = grib2io.open(grbfiles[0])
        desired_level = 'surface'
        for var_name in ['LAND', 'HGT']:
            da = self.get_dataarray(grbs, var_name, desired_level)
            ds = xr.merge([ds, da])

        ds = ds.rename({
            'LAND_surface': 'land_sea_mask',
            'HGT_surface': 'geopotential_at_surface',
            'PRMSL_meansealevel': 'mean_sea_level_pressure',
            'TMP_2maboveground': '2m_temperature',
            'UGRD_10maboveground': '10m_u_component_of_wind',
            'VGRD_10maboveground': '10m_v_component_of_wind',
            'APCP_surface': 'total_precipitation_6hr',
            'HGT': 'geopotential',
            'TMP': 'temperature',
            'SPFH': 'specific_humidity',
            'VVEL': 'vertical_velocity',
            'UGRD': 'u_component_of_wind',
            'VGRD': 'v_component_of_wind',
        })

        ds = ds.assign_coords(datetime=ds.time)

        # Adjust time values relative to the first time step
        ds['time'] = ds['time'] - ds.time[0]

        # Expand dimensions
        ds = ds.expand_dims(dim='batch')
        ds['datetime'] = ds['datetime'].expand_dims(dim='batch')

        # Squeeze dimensions
        ds['geopotential_at_surface'] = ds['geopotential_at_surface'].squeeze('batch')
        ds['land_sea_mask'] = ds['land_sea_mask'].squeeze('batch')

        # Update geopotential unit to m2/s2 by multiplying 9.80665
        ds['geopotential_at_surface'] = ds['geopotential_at_surface'] * 9.80665
        ds['geopotential'] = ds['geopotential'] * 9.80665

        # Update total_precipitation_6hr unit to (m) from (kg/m^2) by dividing it by 1000kg/m³
        ds['total_precipitation_6hr'] = ds['total_precipitation_6hr'] / 1000

        # Define the output NetCDF file
        date = (self.start_datetime + timedelta(hours=6)).strftime('%Y%m%d%H')
        steps = str(len(ds['time']))


        #final_dataset = ds.assign_coords(datetime=ds.time)
        ds.to_netcdf(self.output_netcdf)
        ds.close()
        
        # Optionally, remove downloaded data
        if not self.keep_downloaded_data:
            self.remove_downloaded_data()

        print(f"Process completed successfully, your inputs for GraphCast model generated at:\n {self.output_netcdf}")
            
    def remove_downloaded_data(self):
        # Remove downloaded data from the specified directory
        print("Removing downloaded grib2 data...")
        try:
            os.system(f"rm -rf {self.local_base_directory}")
            print("Downloaded data removed.")
        except Exception as e:
            print(f"Error removing downloaded data: {str(e)}")

    def get_dataarray(self, grbfile, var_name, desired_level):

        # Find the matching grib message, return a list
        msg = grbfile.select(shortName=var_name, level=desired_level)
    
        # create a netcdf dataset using the matching grib message
        lats, lons = msg[0].latlons()
        lats = lats[:,0]
        lons = lons[0,:]
    
        #check latitude range
        reverse_lat = False
        if lats[0] > 0:
            reverse_lat = True
            lats = lats[::-1]
    
        steps = msg[0].validDate
        #if var_name=='APCP':
        #    steps = steps + timedelta(hours=6)
        #precipitation rate has two stepType ('instant', 'avg'), use 'instant')
        data = msg[0].data
        if reverse_lat:
            data = data[::-1, :]
    
        var_name2 = f'{var_name}_{"".join(desired_level.split())}'
        if len(data.shape) == 2:
            da = xr.Dataset(
                data_vars={
                    var_name2: (['lat', 'lon'], data.astype('float32'))
                },
                coords={
                    'lon': lons.astype('float32'),
                    'lat': lats.astype('float32'),
                    'time': steps,  
                }
            )
        elif len(data.shape) == 3:
            da = xr.Dataset(
                data_vars={
                    var_name: (['level', 'lat', 'lon'], data.astype('float32'))
                },
                coords={
                    'lon': lons.astype('float32'),
                    'lat': lats.astype('float32'),
                    'level': np.array(desired_level).astype('int32'),
                    'time': steps,  
                }
            )
    
        return da

    def get_dataarray_3d(self, grbfile, var_name, desired_level):

        data, levels = [], []
        for i, level in enumerate(desired_level):
            msg = grbfile.select(shortName=var_name, level=level)
    
            if i == 0:
                lats, lons = msg[0].latlons()
                lats = lats[:,0]
                lons = lons[0,:]
    
                #check latitude range, graphcast needs [-90, 90]
                reverse_lat = False
                if lats[0] > 0:
                    reverse_lat = True
                    lats = lats[::-1]
    
                steps = msg[0].validDate

            data.append(msg[0].data)
            levels.append(int(level.split(' ')[0]))

        data = np.array(data)
        if reverse_lat:
            data = data[:, ::-1, :]
    
        da = xr.Dataset(
            data_vars={
                var_name: (['level', 'lat', 'lon'], data.astype('float32'))
            },
            coords={
                'lon': lons.astype('float32'),
                'lat': lats.astype('float32'),
                'level': np.array(levels).astype('int32'),
                'time': steps,  
            }
        )
    
        return da

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process GDAS data")
    parser.add_argument("start_datetime", help="Start datetime in the format 'YYYYMMDDHH'")
    parser.add_argument("end_datetime", help="End datetime in the format 'YYYYMMDDHH'")
    parser.add_argument("-l", "--levels", help="number of pressure levels, options: 13, 37", default="13")
    parser.add_argument("-m", "--method", help="method to extact variables from grib2, options: wgrib2, grib2io", default="wgrib2")
    parser.add_argument("-s", "--source", help="the source repository to download gdas grib2 data, options: nomads (up-to-date), s3", default="s3")
    parser.add_argument("-o", "--output", help="Output directory for processed data")
    parser.add_argument("-d", "--download", help="Download directory for raw data")
    parser.add_argument("-k", "--keep", help="Keep downloaded data (yes or no)", default="no")

    args = parser.parse_args()

    start_datetime = datetime.strptime(args.start_datetime, "%Y%m%d%H")
    end_datetime = datetime.strptime(args.end_datetime, "%Y%m%d%H")
    num_pressure_levels = int(args.levels)
    download_source = args.source
    method = args.method
    output_directory = args.output
    download_directory = args.download
    keep_downloaded_data = args.keep.lower() == "yes"

    data_processor = GFSDataProcessor(start_datetime, end_datetime, num_pressure_levels, download_source, output_directory, download_directory, keep_downloaded_data)
    data_processor.download_data()
    
    if method == "wgrib2":
      data_processor.process_data_with_wgrib2()
    elif method == "grib2io":
      data_processor.process_data_with_grib2io()
    else:
      raise NotImplementedError(f"Method {method} is not supported!")

