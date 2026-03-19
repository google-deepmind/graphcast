# GenCast Repository Analysis

## Repository Overview

This is a **modified DeepMind GenCast** repository — a diffusion-based weather forecasting model extended with **storm guidance capabilities**. It can steer, intensify, delete, or spatially shift predicted tropical storms during diffusion sampling via gradient-based latent optimization.

**Framework**: JAX + Haiku | **Grid**: 1440×721 (0.25° ERA5) | **Timestep**: 12h predictions

---

## Architecture

The model follows the Karras et al. (2022) diffusion framework:

1. **Denoiser** — GNN on an icosahedral mesh with Fourier noise-level encoding (`graphcast/denoiser.py`)
2. **Sampler** — DPM-Solver++ 2S with optional guidance hooks (`graphcast/dpm_solver_plus_plus_2s.py`)
3. **Backbone** — Deep typed graph net + sparse transformer attention (`graphcast/deep_typed_graph_net.py`)
4. **Variables** — 6 surface + 6 atmospheric × 13 pressure levels (~84 channels)
5. **Rollout** — Autoregressive multi-day forecasts (`graphcast/rollout.py`)

### 4 Guidance Modes

| Mode | Description |
|------|-------------|
| `none` | Baseline GenCast sampling |
| `direct_optim` | Optimize latent x_t via readout network gradients |
| `input_manipulation` | Modify input fields (intensity scaling + spatial shift) |
| `local_affine` | Optimize affine warp parameters at storm region |

---

## Key Files

| File | Purpose |
|------|---------|
| `10_12-30_...LocalAffine.py` | **Main inference script** (data generation, 5163 lines) |
| `10_12-30_...Backup.py` | Backup version with Chinese comments |
| `graphcast/gencast.py` | Core GenCast model + guidance configs |
| `graphcast/rollout.py` | Multi-step autoregressive rollout |
| `graphcast/data_loader.py` | ERA5 data loading from Zarr/GCS |
| `graphcast/era5_dataset.py` | Typhoon-aware ERA5 PyTorch dataset |
| `sample_config.json` | Example JSON configuration |

---

## How to Run

### 1. Set up the environment

```bash
conda env create -f environment.yml
conda activate gencast
```

Key dependencies: JAX 0.5.2, CUDA 12.8, cuDNN 9.8, dm-haiku, xarray, dask, cartopy, zarr.

### 2. Prerequisites

You need:

- **GPU** with CUDA 12.8 support and high VRAM (the 1440×721 fields are large)
- **GenCast model weights** from GCS bucket `dm_graphcast`
- **ERA5 data** via WeatherBench2 Zarr (`gs://weatherbench2/datasets/era5/...`)
- **Pre-trained readout checkpoint** (e.g. `checkpoint_epoch_020000.pt`)
- **Storm track files** (`tracks_YYYY.txt`) for typhoon center detection

### 3. Configure the script

Edit the **CONFIGURATION SECTION** at the bottom of the main script (`10_12-30_...LocalAffine.py`). Key parameters:

```python
# Paths (update these to your environment)
ROOT_DIR = "/your/project/root"
full_model_path = "/path/to/checkpoint_epoch_020000.pt"

# Guidance method: "none", "direct_optim", "input_manipulation", "local_affine"
GUIDANCE_METHOD = "local_affine"

# Storm targeting
SWEEP_WANT_TIMES = ["2017-09-07 00:00:00"]   # input dates
END_DATES = ["2017-09-08 00:00:00"]           # target dates
MANUAL_TARGETS = [{"lat": 24.0, "lon": 288.0, "radius": 3}]

# Rollout
ROLLOUT_DAYS = 4          # 4-day forecast (8 steps × 12h)
NUM_ROLLOUT_STEPS = 8
```

### 4. Run inference (data generation)

```bash
conda activate gencast
python 10_12-30_gencast_readout_selective_storm_guidance_RollOut_DataOnly_LocalAffine.py
```

This produces NetCDF files under the output directory:

```
output_dir/rollout_data/
  ├── predictions_step00.nc   # Model predictions per step
  ├── gt_step00.nc            # Ground truth
  ├── one_hot.npy             # Storm mask
  └── metadata.json           # Run metadata
```

### 5. Generate visualizations (optional, separate step)

```bash
python Archieve/10_12-30_gencast_readout_selective_storm_guidance_RollOut_visualization.py \
    --root_dir /path/to/output \
    --vis_n_procs 4 \
    --vis_dpi 150 \
    --skip_existing
```

### 6. Training (if needed)

The training script is referenced in `README.md`:

```bash
python gencast_readout_training_script_wFastIO_StormCenter.py
```

---

## Execution Flow

```
ERA5 Zarr (GCS) → Data Loader → Normalization →
GenCast Diffusion Sampling {
    for each denoising step t = T→1:
        1. GNN denoiser predicts x₀ on icosahedral mesh
        2. [GUIDANCE] ∇_{x_t} L(readout, target) → update x_t
        3. DPM-Solver++ 2S step → x_{t-1}
} → 12h Weather Prediction
→ Autoregressive Rollout (multi-day)
→ Save NetCDF → Visualization
```

---

## Module Structure — `graphcast/` Directory

| File | Purpose |
|------|---------|
| `gencast.py` | Core GenCast model — diffusion-based weather predictor using Karras et al. (2022) framework. Defines `ReadOutGuidanceConfig`, task configs with 6 surface + 13 pressure-level atmospheric variables on a 1440×721 grid. |
| `denoiser.py` | Denoiser wrapper — wraps a Predictor (GNN) to act as a denoiser. Contains `FourierFeaturesMLP` for noise-level encoding with sin/cos features. |
| `dpm_solver_plus_plus_2s.py` | DPM-Solver++ 2S sampler — second-order single-step diffusion solver. Contains `Sampler_ReadOut` class with guidance/optimization hooks. |
| `graphcast.py` | Original GraphCast model config — defines `TaskConfig`, `TASK_13`, pressure levels, atmospheric/surface/forcing variable lists. |
| `rollout.py` | Multi-step autoregressive rollout utilities — `chunked_prediction_generator_multiple_runs` for ensemble forecasts with device replication/pmap support. |
| `data_loader.py` | `ERA5DataLoader` — loads ERA5 data from Zarr (GCS or local), supports date filtering, lazy loading via dask, LRU cache. |
| `data_loader_2.py` | Alternative loader that persists filtered subset into RAM. |
| `era5_dataset.py` | `DateMergedERA5TyphoonDataset` (PyTorch Dataset) — merges ERA5 data with typhoon track files. Generates circular one-hot masks centered on storm locations. |
| `era5_dataset_12_01_PureJax.py` | Pure-JAX version of the ERA5 dataset (JIT-compatible training). |
| `deep_typed_graph_net.py` | Deep typed graph neural network (the core GNN backbone). |
| `sparse_transformer.py` | Sparse attention transformer for mesh processing. |
| `transformer.py` | Standard transformer module. |
| `autoregressive.py` | Autoregressive prediction wrapper. |
| `losses.py` | Loss functions including `weighted_mse_per_level`. |
| `icosahedral_mesh.py` | Icosahedral mesh generation. |
| `grid_mesh_connectivity.py` | Grid-to-mesh and mesh-to-grid connectivity. |
| `normalization.py` | Input/output normalization. |
| `casting.py` | Data type casting utilities. |
| `checkpoint.py` | Model checkpoint save/load. |
| `xarray_jax.py` | JAX-backed xarray integration (bridge between xarray and JAX). |
| `vis.py`, `vis_guidance.py` | Visualization utilities for weather maps and guidance results. |
| `samplers_base.py`, `samplers_utils.py` | Base sampler protocol and utilities (noise schedules, rho-CDF, spherical white noise). |

---

## Environment & Dependencies

**Conda environment** named `gencast` (`environment.yml`, 313 lines):

| Category | Key Packages |
|----------|-------------|
| **ML Framework** | JAX 0.5.2, jaxlib 0.5.1, dm-haiku 0.0.13, optax 0.2.4, jraph 0.0.6.dev0, chex 0.1.88, jmp 0.0.4 |
| **CUDA** | CUDA 12.8, cuDNN 9.8, NCCL 2.26 |
| **Scientific** | NumPy 2.2.2, SciPy 1.15.1, xarray 2025.1.2, pandas 2.2.3, scikit-learn 1.6.1 |
| **Data** | zarr 2.18.3, dask 2025.1.0, netCDF4 1.7.2, gcsfs 2025.2.0, tensorstore 0.1.71 |
| **Geospatial** | Cartopy 0.24.1, pyproj, Shapely |
| **Visualization** | Matplotlib 3.10, Pillow, OpenCV |
| **Other** | PyTorch (used for DataSets/one-hot masks only), Jupyter, orbax-checkpoint |
| **Python** | 3.10.16 |

---

> **Note**: The hardcoded paths (e.g. `/fs/nexus-projects/...`) point to a Linux HPC cluster. You'll need to update all paths in the configuration section to match your environment, and ensure access to the ERA5 data and model checkpoints.
