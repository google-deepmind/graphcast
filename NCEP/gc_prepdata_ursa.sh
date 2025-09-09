#!/bin/bash --login

#SBATCH --ntasks=1
#SBATCH --mem=10g
#SBATCH --account=nems
#SBATCH --partition=u1-service
#SBATCH --time=10:00
#SBATCH --job-name=solo-prepdata
#SBATCH --output=slurm/solo_prepdata.out
#SBATCH --error=slurm/solo_prepdata.err

# load necessary modules
module use /contrib/spack-stack/spack-stack-1.9.1/envs/ue-oneapi-2024.2.1/install/modulefiles/Core/
module load stack-oneapi
module load wgrib2
module list

# Activate Conda environment
source /scratch3/NCEPDEV/nems/Linlin.Cui/miniforge3/etc/profile.d/conda.sh
conda activate mlglobal

curr_datetime=${1:-2025090506}
prev_datetime=${2:-2025090500}

echo "Current state: $curr_datetime"
echo "6 hours earlier state: $prev_datetime"

PYD=$(echo $curr_datetime | cut -c1-8)
cyc=$(echo $curr_datetime | cut -c9-10)

forecast_length=64
echo "forecast length: $forecast_length"

num_pressure_levels=13
echo "number of pressure levels: $num_pressure_levels"

start_time=$(date +%s)
echo "start runing gdas utility to generate graphcast inputs for: $curr_datetime"
# Run the Python script gdas.py with the calculated times
python gen_mlgfs_ics.py "$prev_datetime" "$curr_datetime" -l "$num_pressure_levels" -o aigfs."$PYD"/"$cyc"

end_time=$(date +%s)  # Record the end time in seconds since the epoch

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time for gdas_utility.py $execution_time seconds"

## Get TC tracker data
#module load awscli-v2/2.15.53
#aws s3 --profile gcgfs sync s3://noaa-nws-graphcastgfs-pds/hurricanes/syndat tracker/syndat
