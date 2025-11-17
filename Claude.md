# GraphCast & GenCast

## Project Overview

This repository contains the implementation of two state-of-the-art weather forecasting models from Google DeepMind:

- **GraphCast**: A high-resolution deterministic weather forecasting model using graph neural networks
- **GenCast**: A diffusion-based ensemble forecasting model for probabilistic weather predictions

Both models provide medium-range global weather forecasting using machine learning trained on ERA5 reanalysis data.

## Architecture

The project is built on JAX/Haiku with graph neural networks and transformers for processing weather data on icosahedral meshes.

### Key Technologies

- **JAX**: Core computational framework
- **Haiku**: Neural network library
- **Jraph**: Graph neural network library
- **XArray**: Multi-dimensional data handling
- **Chex**: Testing utilities for JAX

## Project Structure

```
graphcast/
├── graphcast/          # Main source code
│   ├── graphcast.py    # GraphCast model architecture
│   ├── gencast.py      # GenCast model architecture
│   ├── autoregressive.py        # Auto-regressive wrapper for training
│   ├── rollout.py              # Inference-time rollout
│   ├── normalization.py        # Data normalization
│   ├── losses.py               # Loss functions
│   ├── deep_typed_graph_net.py # Deep GNN implementation
│   ├── sparse_transformer.py   # Sparse transformer for mesh processing
│   ├── denoiser.py             # GenCast denoiser
│   ├── dpm_solver_plus_plus_2s.py # DPM-Solver++ sampler
│   ├── grid_mesh_connectivity.py  # Grid-mesh conversion
│   ├── icosahedral_mesh.py     # Mesh definition
│   └── ...
├── docs/               # Documentation and scorecards
├── *.ipynb            # Demo notebooks
├── setup.py           # Package setup
└── README.md          # Main documentation
```

## Core Components

### GraphCast Model

**Main files:**
- `graphcast.py` - One-step prediction architecture
- `casting.py` - BFloat16 precision wrapper
- `solar_radiation.py` - TOA incident solar radiation

**Available pretrained models:**
1. GraphCast (0.25°, 37 levels) - trained on 1979-2017
2. GraphCast_small (1°, 13 levels) - trained on 1979-2015
3. GraphCast_operational (0.25°, 13 levels) - fine-tuned on HRES

### GenCast Model

**Main files:**
- `gencast.py` - Model architecture with denoiser and sampler
- `denoiser.py` - One-step denoiser
- `denoisers_base.py` - Denoiser interface
- `samplers_base.py`, `samplers_utils.py` - Sampling interface and utilities

**Available pretrained models:**
1. GenCast 0.25deg <2019 - High resolution ERA5
2. GenCast 0.25deg Operational <2019 - Fine-tuned on HRES
3. GenCast 1.0deg <2019 - Lower memory footprint
4. GenCast 1.0deg Mini <2019 - Smallest model for demos

### Shared Components

- **Graph Operations**: `typed_graph.py`, `typed_graph_net.py`, `deep_typed_graph_net.py`
- **Mesh Handling**: `icosahedral_mesh.py`, `grid_mesh_connectivity.py`
- **Data Processing**: `data_utils.py`, `xarray_jax.py`, `xarray_tree.py`
- **Training Utils**: `autoregressive.py`, `normalization.py`, `losses.py`
- **Model Utils**: `model_utils.py`, `mlp.py`, `checkpoint.py`

## Data Requirements

### Training Data
- **ERA5**: ECMWF reanalysis dataset (1979-2018)
  - Available via Weatherbench2 as Zarr format
  - https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5

### Fine-tuning Data
- **HRES-fc0**: Operational analysis data (2016-2021)
  - Available via Weatherbench2
  - https://weatherbench2.readthedocs.io/en/latest/data-guide.html#ifs-hres-t-0-analysis

### Pretrained Assets
- Model weights, normalization statistics, and example data available at:
  - Google Cloud Bucket: `gs://dm_graphcast`

## Getting Started

### Demo Notebooks

1. **GraphCast Demo** (`graphcast_demo.ipynb`)
   - Loading pretrained models
   - Making predictions
   - Computing loss and gradients
   - Can run in Colab

2. **GenCast Mini Demo** (`gencast_mini_demo.ipynb`)
   - Running the smallest GenCast model
   - Suitable for free Colab notebooks
   - Good starting point for understanding the code

3. **GenCast Cloud VM Demo** (`gencast_demo_cloud_vm.ipynb`)
   - Running full-size GenCast models
   - Requires Google Cloud TPU VM
   - See `docs/cloud_vm_setup.md` for setup instructions

### Development Workflow

1. **Reading Model Code**
   - Start with `graphcast.py` or `gencast.py` for model architecture
   - Check `predictor_base.py` for the interface all models implement
   - Review `autoregressive.py` to understand training loop

2. **Data Pipeline**
   - `data_utils.py` - Preprocessing utilities
   - `normalization.py` - Historical normalization
   - `grid_mesh_connectivity.py` - Converting grids to mesh

3. **Training**
   - Use `autoregressive.py` wrapper for differentiable rollouts
   - `losses.py` provides latitude-weighted loss computations
   - `checkpoint.py` for serialization

4. **Inference**
   - Use `rollout.py` for longer non-differentiable trajectories
   - Models implement `predictor_base.Predictor` interface

## Dependencies

Key dependencies (see `setup.py` for full list):
- `jax` - Core computation
- `dm-haiku` - Neural network framework
- `jraph` - Graph neural networks
- `xarray` - Multi-dimensional arrays
- `numpy`, `scipy` - Numerical computing
- `trimesh` - Mesh operations
- `dask` - Parallel computing
- `chex` - JAX testing utilities
- `dinosaur-dycore` - Dynamical core

## Testing

Test files follow the pattern `*_test.py`:
- `checkpoint_test.py`
- `data_utils_test.py`
- `grid_mesh_connectivity_test.py`
- `icosahedral_mesh_test.py`
- `solar_radiation_test.py`
- `xarray_jax_test.py`
- `xarray_tree_test.py`

## Important Notes

### Licenses
- **Code**: Apache License 2.0
- **Model Weights**: CC BY-NC-SA 4.0 (non-commercial)

### Data Compliance
- ERA5 and HRES data have separate terms and conditions
- Check compliance with ECMWF data policies before use
- Modified Copernicus Climate Change Service information

### Disclaimers
- Not an officially supported Google product
- Experimental research project
- Not endorsed by meteorological agencies
- Does not replace official weather alerts/warnings

## Citations

When using this code, cite:

**GraphCast:**
```
Lam et al. (2023). Learning skillful medium-range global weather forecasting.
Science, 382(6677), 1416-1421.
```

**GenCast:**
```
Price et al. (2023). GenCast: Diffusion-based ensemble forecasting for
medium-range weather. arXiv preprint arXiv:2312.15796.
```

## Contact

For feedback and questions: gencast@google.com

## Additional Resources

- Blog post: https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/
- GraphCast paper: https://www.science.org/doi/10.1126/science.adi2336
- GenCast paper: https://arxiv.org/abs/2312.15796
- Cloud VM setup: `docs/cloud_vm_setup.md`
- Scorecards: `docs/` directory
