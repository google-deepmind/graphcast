#!/bin/bash --login

#SBATCH --nodes=1
#SBATCH --account=gpu-ai4wp
#SBATCH --partition=u1-compute
#SBATCH --cpus-per-task=120
#SBATCH --time=2:00:00
#SBATCH --job-name=solo_fcst
#SBATCH --output=slurm/solo_fcst.out
#SBATCH --error=slurm/solo_fcst.err

# load necessary modules
module use /contrib/spack-stack/spack-stack-1.9.1/envs/ue-oneapi-2024.2.1/install/modulefiles/Core/
module load stack-oneapi
module load wgrib2

source /scratch3/NCEPDEV/nems/Linlin.Cui/miniforge3/etc/profile.d/conda.sh
conda activate mlglobal

PDY=${1:-20250905}
cyc=${2:-06}

forecast_length=64
echo "forecast length: $forecast_length"

num_pressure_levels=13
echo "number of pressure levels: $num_pressure_levels"

model_weights=/scratch3/NCEPDEV/nems/MGFS/graphcast/gc_weights
echo "Model weights and stats are at: $model_weights"

start_time=$(date +%s)
echo "start runing graphcast to get real time 10-days forecasts for: $curr_datetime"

numactl --interleave=all python run_graphcast.py -i aigfs.$PDY/$cyc/aigfs.t${cyc}z.ic.nc -w $model_weights -n aigfs -l "$forecast_length" -p "$num_pressure_levels" -o aigfs.$PDY/$cyc -u no -k yes

end_time=$(date +%s)  # Record the end time in seconds since the epoch

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time for graphcast: $execution_time seconds"
