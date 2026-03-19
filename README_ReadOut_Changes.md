# GenCast ReadOut: Changes on Top of DeepMind's GenCast

This document details every modification made to the [original DeepMind GenCast](https://github.com/google-deepmind/graphcast) to add the **ReadOut** system -- a lightweight head that taps internal GNN features to predict storm masks, and then uses those predictions to **guide** the diffusion sampling process toward desired weather outcomes.

---

## Table of Contents

1. [High-Level Summary](#1-high-level-summary)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Modified Files & What Changed](#3-modified-files--what-changed)
4. [ReadOut Head Architecture](#4-readout-head-architecture)
5. [Training Pipeline](#5-training-pipeline)
6. [Cost Function](#6-cost-function)
7. [Guided Inference](#7-guided-inference)
8. [Local Affine Warp](#8-local-affine-warp)
9. [New Configuration Classes](#9-new-configuration-classes)
10. [New Files (Not in Original)](#10-new-files-not-in-original)
11. [Hyperparameter Reference](#11-hyperparameter-reference)

---

## 1. High-Level Summary

The original GenCast is a diffusion-based weather forecasting model that denoises atmospheric states via a GNN on an icosahedral mesh. Our modifications add:

| What | Why |
|------|-----|
| **ReadOut head** | A parallel MLP decoder that reads the same mesh latents and predicts a binary storm mask |
| **Frozen-backbone training** | Only the ReadOut head is trained; the pretrained GenCast denoiser is frozen |
| **Two-channel cross-entropy loss** | Treats (u, v) wind logits as a 2-class classifier (storm vs. no-storm) |
| **Guided sampling** | At inference, optimizes the latent `x_t` to steer the readout prediction toward a target storm mask |
| **Local affine warp** | An alternative guidance mode that translates/rotates/scales weather patterns in a local region |
| **Selective guidance mask** | Restricts the guidance loss to a spatial sub-region |
| **Typhoon-aware data loading** | ERA5 dataset with storm center extraction and circular one-hot mask generation |

---

## 2. Architecture Diagram

### Original GenCast (DeepMind)

```
 Inputs (ERA5)          Noisy Targets
      |                      |
      v                      v
 +---------+          +-----------+
 | Encoder |          | + Noise   |
 +---------+          +-----------+
      |                      |
      +----------+-----------+
                 |
                 v
    +---------------------------+
    |    Grid2Mesh GNN          |    Grid nodes --> Mesh nodes
    |  [65160 grid, 2562 mesh]  |
    +---------------------------+
                 |
                 v
    +---------------------------+
    |    Mesh GNN               |    Message passing on icosahedral mesh
    |  (Sparse Transformer)     |    (16 rounds, with attention)
    +---------------------------+
                 |
                 v
    +---------------------------+
    |    Mesh2Grid GNN          |    Mesh nodes --> Grid nodes
    |  [2562 mesh, 65160 grid]  |    Output: [65160, batch, 84]
    +---------------------------+
                 |
                 v
         Denoised Prediction
         (84 weather channels)
```

### Modified GenCast with ReadOut

```
 Inputs (ERA5)          Noisy Targets
      |                      |
      v                      v
 +---------+          +-----------+
 | Encoder |          | + Noise   |
 +---------+          +-----------+
      |                      |
      +----------+-----------+
                 |
                 v
    +---------------------------+
    |    Grid2Mesh GNN          |
    |  [65160 grid, 2562 mesh]  |
    +---------------------------+
         |               |
         v               |
    latent_grid_nodes    |
    [65160, B, 512]      |
         |               v
         |     +---------------------------+
         |     |    Mesh GNN               |
         |     |  (Sparse Transformer)     |
         |     +---------------------------+
         |               |
         |               v
         |     updated_latent_mesh_nodes
         |     [2562, B, 512]
         |               |
         +-------+-------+
                 |       |
         +-------+       +--------+
         |                        |
         v                        v
+--------------------+   +--------------------+
| Mesh2Grid GNN      |   | Mesh2Grid GNN      |
| (Standard Decoder) |   | (ReadOut Decoder)   |    <-- NEW: parallel path
| _networks_builder  |   | _networks_builder   |         with separate
|                    |   | _4_readout          |         MLP weights
+--------------------+   +--------------------+
         |                        |
         v                        v
  Main Prediction           ReadOut Prediction
  [65160, B, 84]            [65160, B, 84]
  (weather fields)          (storm logits via
                             u,v wind channels)
                                  |
                                  v
                         +------------------+
                         | Softmax over     |
                         | (u, v) channels  |    <-- 2-class classification
                         | --> P(storm)     |
                         +------------------+
                                  |
                                  v
                           Storm Mask
                           (B, 181, 360)
```

### Guided Inference Flow

```
x_T ~ N(0, sigma_max)
  |
  v
for each denoising step i = 0..19:
  |
  |  +-- if i in inner_opt_step_idxs:
  |  |
  |  |   for j = 0..max_opt_steps:
  |  |     +-----------------------------+
  |  |     | x_t ---> Denoiser --------> readout prediction      |
  |  |     |          (frozen)            |                       |
  |  |     |                              v                       |
  |  |     |                     Loss(readout, target_mask)       |
  |  |     |                              |                       |
  |  |     |                     grad_x = dLoss/dx_t              |
  |  |     |                              |                       |
  |  |     |                     x_t = x_t - lr * Adam(grad_x)   |
  |  |     +-----------------------------+
  |  |
  |  +-- Standard DPM-Solver++ 2S step:
  |       x_{i+1} = denoise_and_update(x_i, sigma_i)
  |
  v
x_0 = Final Prediction (steered toward target storm)
```

---

## 3. Modified Files & What Changed

| File | Change Type | Description |
|------|------------|-------------|
| `graphcast/denoiser.py` | **Modified** | Added `ReadOut_flag`, dual-output `__call__`, `_run_mesh2grid_gnn_4_readout` |
| `graphcast/deep_typed_graph_net.py` | **Modified** | Added `readout_out()` method and `_networks_builder_4_readout` |
| `graphcast/gencast.py` | **Modified** | Added `readout_train`, `readout_inference_vis`, `readout_guided_inference_vis`, guidance config dataclasses |
| `graphcast/dpm_solver_plus_plus_2s.py` | **Modified** | Added `Sampler_ReadOut` class with `guided()` method, warp functions, inner optimization loop |
| `graphcast/losses.py` | **Modified** | Added `two_channel_crossentropy_optax`, `debug_two_channel_crossentropy_optax`, `debug_center_mse_optax` |
| `graphcast/normalization.py` | **Modified** | Added `readout_train`, `readout_inference_vis`, `readout_guided_inference_vis` wrappers |
| `graphcast/era5_dataset_12_01_PureJax.py` | **New** | Typhoon-aware ERA5 dataset with circular storm mask generation |
| `graphcast/vis_guidance.py` | **New** | Readout comparison visualization (no-guidance vs guidance) |
| Training/inference scripts (`10_12-30_*.py`) | **New** | Full pipelines for readout training, inference, rollout, and guidance |

---

## 4. ReadOut Head Architecture

### Where Features Are Tapped

The ReadOut head taps the **same internal latent features** as the main decoder, specifically:

- `updated_latent_mesh_nodes` [2562, B, 512] -- output of the Mesh GNN (sparse transformer)
- `latent_grid_nodes` [65160, B, 512] -- output of the Grid2Mesh GNN

These are fed into a **parallel Mesh2Grid GNN** with its own MLP weights.

### Network Structure

The ReadOut decoder (`_networks_builder_4_readout` in `deep_typed_graph_net.py:239`) mirrors the standard decoder architecture:

```
ReadOut Mesh2Grid GNN:
  Embedder:  edge MLP [input --> 512 --> 512]  (with LayerNorm + NormConditioning)
             node MLP [input --> 512 --> 512]
  Processor: N message-passing steps
             edge MLP [512 --> 512]
             node MLP [512 --> 512]
  Decoder:   node MLP [512 --> 84]   (projects to output channels)
```

Each MLP has:
- Hidden size: 512 (= `latent_size`)
- Hidden layers: 1
- Activation: SiLU/Swish
- LayerNorm after each MLP
- Optional norm conditioning on noise level

The key difference: the ReadOut decoder has **separate, independently trainable weights** from the main decoder, created under different Haiku name scopes. During training, only these ReadOut weights are updated.

### Output Interpretation

The 84-channel output is structured as standard weather variables, but only the **u and v wind components** are used as 2-channel logits:

```
readout[u_component_of_wind]  -->  logit for class 0 (no storm)
readout[v_component_of_wind]  -->  logit for class 1 (storm)

softmax(u, v) --> P(storm) per pixel
```

---

## 5. Training Pipeline

### Procedure (in `gencast.py:readout_train`)

1. **Freeze backbone**: All GenCast denoiser weights are frozen; only ReadOut head MLP weights receive gradients
2. **Sample noise**: Draw `sigma` from the rho-inverse CDF distribution, same as original GenCast training
3. **Corrupt targets**: `noisy_targets = targets + noise * sigma`
4. **Forward pass**: Run the full denoiser, get `(main_prediction, readout_prediction)`
5. **Compute loss**: Two-channel cross-entropy between readout prediction and one-hot storm mask
6. **Update**: Adam optimizer (`lr = 1e-4`) updates only ReadOut head parameters

### Storm Mask Generation

The training labels are **circular one-hot masks** generated from typhoon track metadata:

```python
# era5_dataset_12_01_PureJax.py
def data_merge_make_circular_one_hot_varradius_cpu(
    batch_size, height, width, centers_with_radii
):
    # For each storm center (lat_idx, lon_idx, radius):
    #   mask[dist <= radius^2] = 1
    # Combine with OR, convert to one-hot (B, 181, 360, 2)
```

---

## 6. Cost Function

### Training Loss: `two_channel_crossentropy_optax` (`losses.py:80`)

```
Loss = mean over pixels of [ w(pixel) * softmax_CE(logits, one_hot) ]

where:
  logits = stack(u_wind, v_wind)        shape (B, H, W, 2)
  one_hot = storm mask                  shape (B, H, W, 2)

  w(pixel) = { dynamic_pos_weight   if pixel is storm (class 1)
             { 1.0                  if pixel is background (class 0)

  dynamic_pos_weight = clip( (n_neg / n_pos) * scale, min_w, max_w )
```

Default hyperparameters:
- `dynamic_weight_scale = 1.0`
- `min_pos_weight = 1.0`
- `max_pos_weight = 1000.0`

The dynamic weighting addresses extreme class imbalance (storms occupy < 1% of pixels).

### Guidance Loss (at inference)

When `loss_type = "readout_l2"`: same softmax cross-entropy as training, computed on the readout prediction vs. a target storm mask. When a `guidance_mask` is provided, loss is only computed within the masked region.

When `loss_type = "xt_l2"`: simple L2 norm of x_t (`mean(x_t^2)`), used as a regularization baseline.

---

## 7. Guided Inference

### Mechanism (`dpm_solver_plus_plus_2s.py:guided`)

At selected denoising steps, the sampler **pauses** and runs an inner optimization loop on the latent state `x_t`:

```
for each inner optimization iteration:
    1. Forward x_t through frozen denoiser --> readout prediction
    2. Compute loss between readout and target storm mask
    3. grad_x = d(loss) / d(x_t)
    4. x_t = x_t - Adam(lr, clip_grad_norm(grad_x, 1.0))
```

### Key Design Choices

- **Optimizer**: Adam with gradient clipping (`clip_by_global_norm = 1.0`)
- **Per-step control**: `inner_opt_steps_map` specifies how many optimization iterations at each denoising step (e.g., `{4: 15, 8: 10, 12: 5}`)
- **Selective guidance**: A spatial `guidance_mask` (B, H, W) restricts loss computation to a sub-region
- **Loss history tracking**: All per-step losses are recorded and returned for analysis

---

## 8. Local Affine Warp

An alternative guidance mode that optimizes **spatial transformation parameters** instead of (or alongside) pixel values.

### Warp Parameters

```python
warp_params = {
    "translation": [delta_lat, delta_lon],   # shift storm position
    "rotation": theta,                        # rotate storm structure (radians)
    "scale": [scale_lat, scale_lon],          # stretch/compress storm
}
```

### How It Works

1. Define a circular region around the storm center (`center_lat`, `center_lon`, `radius`)
2. Apply an **inverse affine transform** to compute source coordinates
3. Resample the field using `jax.scipy.ndimage.map_coordinates` (bilinear interpolation)
4. **Blend** with smooth Gaussian falloff at the mask boundary:
   ```
   output = mask * falloff * warped + (1 - mask * falloff) * original
   ```
5. Regularization loss keeps warp params close to identity

### Joint Optimization

When warp is enabled, the inner loop jointly optimizes:
- `x_t` (latent state) via its own Adam optimizer
- `warp_params` via a separate Adam optimizer (`warp_lr`, default `1e-2`)

Each warp axis can be independently frozen:
- `optimize_translation`, `optimize_rotation`, `optimize_scale` (bool flags)

---

## 9. New Configuration Classes

All defined in `gencast.py`:

```python
# Extends SamplerConfig with readout step selection
ReadOutSamplerConfig:
    selected_denoising_step: List[int] = [10, 15]

# Extends NoiseConfig
ReadOutNoiseConfig:
    ReadOut_flag: bool = True

# Extends DenoiserArchitectureConfig
ReadOutDenoiserArchitectureConfig:
    ReadOut_flag: bool = True

# Guidance configuration
ReadOutGuidanceConfig:
    inner_opt_step_idxs: List[int]       # which denoising steps to pause at
    inner_opt_steps_map: Dict[int, int]  # {step: num_iterations}
    max_opt_steps: int                   # max iterations (for JAX compilation)
    inner_opt_lr: float = 1e-2           # Adam learning rate for x_t
    loss_type: str = "readout_l2"        # "readout_l2" | "xt_l2"
    target_readout: xarray.Dataset       # target storm mask (one-hot)
    warp_configs: Optional[dict]         # local affine warp settings

# Extends guidance config with spatial mask
ReadOutGuidanceConfigWithMask(ReadOutGuidanceConfig):
    guidance_mask: jnp.ndarray           # (B, H, W) binary mask
```

---

## 10. New Files (Not in Original)

| File | Purpose |
|------|---------|
| `graphcast/era5_dataset_12_01_PureJax.py` | Typhoon-aware ERA5 PyTorch dataset with variable-radius circular mask generation |
| `graphcast/vis_guidance.py` | Side-by-side visualization: no-guidance vs. guided readout predictions |
| `noise_simulation.py` | Noise schedule simulation / debugging utilities |
| `Archieve/gencast_readout_training_script_*.py` | ReadOut head training scripts (frozen backbone + Adam on readout weights) |
| `10_12-30_*_LocalAffine.py` | Main inference script with all 4 guidance modes |
| `10_12-30_*_Backup.py` | Backup inference script |

---

## 11. Hyperparameter Reference

### Training

| Parameter | Value | Location |
|-----------|-------|----------|
| Learning rate | `1e-4` | Training script |
| Optimizer | Adam | Training script |
| Batch size | 1 | Training script |
| Epochs | 100,000 | Training script |
| Dynamic weight scale | `1.0` | Training script |
| Min pos weight | `1.0` | `losses.py` |
| Max pos weight | `1000.0` | `losses.py` |

### Guided Inference

| Parameter | Value | Location |
|-----------|-------|----------|
| Inner opt LR (x_t) | `1.0` | Inference script |
| Gradient clipping | `clip_by_global_norm(1.0)` | `dpm_solver_plus_plus_2s.py` |
| Optimizer | Adam | `dpm_solver_plus_plus_2s.py` |
| Num denoising steps | 20 | `SamplerConfig` |
| Warp LR | `1e-2` | `dpm_solver_plus_plus_2s.py` |
| Warp regularization weight | `1e-3` | `dpm_solver_plus_plus_2s.py` |
| Translation clip | `[-20, 20]` | `dpm_solver_plus_plus_2s.py` |
| Scale clip | `[0.5, 2.0]` | `dpm_solver_plus_plus_2s.py` |
| Rotation clip | `[-pi, pi]` | `dpm_solver_plus_plus_2s.py` |

### Sampler (unchanged from original)

| Parameter | Value |
|-----------|-------|
| Max noise level | 80.0 |
| Min noise level | 0.03 |
| Num noise levels | 20 |
| Rho | 7.0 |
| Stochastic churn rate | 2.5 |
| Noise level inflation | 1.05 |

---

## Summary of Changes vs. Original

```
Original DeepMind GenCast          This Repository
========================          ================
Single decoder output        -->  Dual output (main + readout)
No readout head              -->  Parallel Mesh2Grid GNN with separate weights
MSE loss only                -->  + Two-channel softmax cross-entropy
Standard sampling only       -->  + Guided sampling with inner optimization
No storm awareness           -->  Typhoon track data + circular mask labels
No spatial transforms        -->  Local affine warp (translate/rotate/scale)
No selective guidance         -->  Spatial guidance mask support
```
