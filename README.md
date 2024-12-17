# Google DeepMind GraphCast and GenCast

This package contains example code to run and train the weather models used in the research papers [GraphCast](https://www.science.org/doi/10.1126/science.adi2336) and [GenCast](https://arxiv.org/abs/2312.15796).

It also provides pretrained model weights, normalization statistics and example input data on [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/dm_graphcast).

Full model training requires downloading the
[ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
dataset, available from [ECMWF](https://www.ecmwf.int/). This can best be
accessed as Zarr from [Weatherbench2's ERA5 data](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5).

Data for operational fine-tuning can similarly be accessed at [Weatherbench2's HRES 0th frame data](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#ifs-hres-t-0-analysis).

These datasets may be governed by separate terms and conditions or license provisions. Your use of such third-party materials is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use.

## Overview of files common to models

*   `autoregressive.py`: Wrapper used to run (and train) the one-step predictions
    to produce a sequence of predictions by auto-regressively feeding the
    outputs back as inputs at each step, in JAX a differentiable way.
*   `checkpoint.py`: Utils to serialize and deserialize trees.
*   `data_utils.py`: Utils for data preprocessing.
*   `deep_typed_graph_net.py`: General purpose deep graph neural network (GNN)
    that operates on `TypedGraph`'s where both inputs and outputs are flat
    vectors of features for each of the nodes and edges.
*   `grid_mesh_connectivity.py`: Tools for converting between regular grids on a
    sphere and triangular meshes.
*   `icosahedral_mesh.py`: Definition of an icosahedral multi-mesh.
*   `losses.py`: Loss computations, including latitude-weighting.
*   `mlp.py`: Utils for building MLPs with norm conditioning layers.
*   `model_utils.py`: Utilities to produce flat node and edge vector features
    from input grid data, and to manipulate the node output vectors back
    into a multilevel grid data.
*   `normalization.py`: Wrapper used to normalize inputs according to historical
    values, and targets according to historical time differences.
*   `predictor_base.py`: Defines the interface of the predictor, which models
    and all of the wrappers implement.
*   `rollout.py`: Similar to `autoregressive.py` but used only at inference time
    using a python loop to produce longer, but non-differentiable trajectories.
*   `typed_graph.py`: Definition of `TypedGraph`'s.
*   `typed_graph_net.py`: Implementation of simple graph neural network
    building blocks defined over `TypedGraph`'s that can be combined to build
    deeper models.
*   `xarray_jax.py`: A wrapper to let JAX work with `xarray`s.
*   `xarray_tree.py`: An implementation of tree.map_structure that works with
    `xarray`s.

## GenCast: Diffusion-based ensemble forecasting for medium-range weather

This package provides four pretrained models:

1.  `GenCast 0p25deg <2019`, GenCast model at 0.25deg resolution with 13
pressure levels and a 6 times refined icosahedral mesh. This model is trained on
ERA5 data from 1979 to 2018 (inclusive), and can be causally evaluated on 2019
and later years. This model was described in the paper
`GenCast: Diffusion-based ensemble forecasting for medium-range weather`
(https://arxiv.org/abs/2312.15796)

2.  `GenCast 0p25deg Operational <2022`, GenCast model at 0.25deg resolution, with 13 pressure levels and a 6
times refined icosahedral mesh. This model is trained on ERA5 data from
1979 to 2018, and fine-tuned on HRES-fc0 data from
2016 to 2021 and can be causally evaluated on 2022 and later years.
This model can make predictions in an operational setting (i.e., initialised
from HRES-fc0)

3.  `GenCast 1p0deg <2019`, GenCast model at 1deg resolution, with 13 pressure
levels and a 5 times refined icosahedral mesh. This model is
trained on ERA5 data from 1979 to 2018, and can be causally evaluated on 2019 and later years.
This model has a smaller memory footprint than the 0.25deg models

4. `GenCast 1p0deg Mini <2019`, GenCast model at 1deg resolution, with 13 pressure levels and a
4 times refined icosahedral mesh. This model is trained on ERA5 data
from 1979 to 2018, and can be causally evaluated on 2019 and later years.
This model has the smallest memory footprint of those provided and has been
provided to enable low cost demonstrations (for example, it is runnable in a free Colab notebook).
While its performance is reasonable, it is not representative of the performance
of the GenCast models (1-3) above. For reference, a scorecard comparing its performance to ENS can be found in [docs/](https://github.com/google-deepmind/graphcast/blob/main/docs/GenCast_1p0deg_Mini_ENS_scorecard.png). Note that in this scorecard,
GenCast Mini only uses 8 member ensembles (vs. ENS' 50) so we use the fair (unbiased)
CRPS to allow for fair comparison.

The best starting point is to open `gencast_mini_demo.ipynb` in [Colaboratory](https://colab.research.google.com/github/deepmind/graphcast/blob/master/gencast_mini_demo.ipynb), which gives an
example of loading data, generating random weights or loading a `GenCast 1p0deg Mini <2019`
snapshot, generating predictions, computing the loss and computing gradients.
The one-step implementation of GenCast architecture is provided in
`gencast.py` and the relevant data, weights and statistics are in the `gencast/`
subdir of the Google Cloud Bucket.

### Instructions for running GenCast on Google Cloud compute

[cloud_vm_setup.md](https://github.com/google-deepmind/graphcast/blob/main/docs/cloud_vm_setup.md)
contains detailed instructions on launching a Google Cloud TPU VM. This provides
a means of running models (1-3) in the separate `gencast_demo_cloud_vm.ipynb` through [Colaboratory](https://colab.research.google.com/github/deepmind/graphcast/blob/master/gencast_demo_cloud_vm.ipynb).

The document also provides [instructions](https://github.com/google-deepmind/graphcast/blob/main/docs/cloud_vm_setup.md#running-inference-on-gpu) for running GenCast on a GPU. This requires using a different attention implementation.

### Brief description of relevant library files

*   `denoiser.py`: The GenCast denoiser for one step predictions.
*   `denoisers_base.py`: Defines the interface of the denoiser.
*   `dpm_solver_plus_plus_2s.py`: Sampler using DPM-Solver++ 2S from [1].
*   `gencast.py`: Combines the GenCast model architecture, wrapped as a
    denoiser, with a sampler to generate predictions.
*   `nan_cleaning.py`: Wraps a predictor to allow it to work with data
    cleaned of NaNs. Used to remove NaNs from sea surface temperature.
*   `samplers_base.py`: Defines the interface of the sampler.
*   `samplers_utils.py`: Utility methods for the sampler.
*   `sparse_transformer.py`: General purpose sparse transformer that
    operates on `TypedGraph`'s where both inputs and outputs are flat vectors of
    features for each of the nodes and edges. `predictor.py` uses one of these
    for the mesh GNN.
*   `sparse_transformer_utils.py`: Utility methods for the sparse
    transformer.
*   `transformer.py`: Wraps the mesh transformer, swapping the leading
    two axes of the nodes in the input graph.

[1] DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic
  Models, https://arxiv.org/abs/2211.01095

## GraphCast: Learning skillful medium-range global weather forecasting

This package provides three pretrained models:

1.  `GraphCast`, the high-resolution model used in the GraphCast paper (0.25 degree
resolution, 37 pressure levels), trained on ERA5 data from 1979 to 2017,

2.  `GraphCast_small`, a smaller, low-resolution version of GraphCast (1 degree
resolution, 13 pressure levels, and a smaller mesh), trained on ERA5 data from
1979 to 2015, useful to run a model with lower memory and compute constraints,

3.  `GraphCast_operational`, a high-resolution model (0.25 degree resolution, 13
pressure levels) pre-trained on ERA5 data from 1979 to 2017 and fine-tuned on
HRES data from 2016 to 2021. This model can be initialized from HRES data (does
not require precipitation inputs).

The best starting point is to open `graphcast_demo.ipynb` in [Colaboratory](https://colab.research.google.com/github/deepmind/graphcast/blob/master/graphcast_demo.ipynb), which gives an
example of loading data, generating random weights or load a pre-trained
snapshot, generating predictions, computing the loss and computing gradients.
The one-step implementation of GraphCast architecture, is provided in
`graphcast.py` and the relevant data, weights and statistics are in the `graphcast/`
subdir of the Google Cloud Bucket.

WARNING: For backwards compatibility, we have also left GraphCast data in the top level of the bucket. These will eventually be deleted in favour of the `graphcast/` subdir.

### Brief description of relevant library files:

*   `casting.py`: Wrapper used around GraphCast to make it work using
    BFloat16 precision.
*   `graphcast.py`: The main GraphCast model architecture for one-step of
    predictions.
*   `solar_radiation.py`: Computes Top-Of-the-Atmosphere (TOA) incident solar
    radiation compatible with ERA5. This is used as a forcing variable and thus
    needs to be computed for target lead times in an operational setting.

## Dependencies.

[Chex](https://github.com/deepmind/chex),
[Dask](https://github.com/dask/dask),
[Dinosaur](https://github.com/google-research/dinosaur),
[Haiku](https://github.com/deepmind/dm-haiku),
[JAX](https://github.com/google/jax),
[JAXline](https://github.com/deepmind/jaxline),
[Jraph](https://github.com/deepmind/jraph),
[Numpy](https://numpy.org/),
[Pandas](https://pandas.pydata.org/),
[Python](https://www.python.org/),
[SciPy](https://scipy.org/),
[Tree](https://github.com/deepmind/tree),
[Trimesh](https://github.com/mikedh/trimesh),
[XArray](https://github.com/pydata/xarray) and
[XArray-TensorStore](https://github.com/google/xarray-tensorstore).


## License and Disclaimers

The Colab notebooks and the associated code are licensed under the Apache License, Version 2.0. You may obtain a copy of the License at: https://www.apache.org/licenses/LICENSE-2.0.

The model weights are made available for use under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). You may obtain a copy of the License at: https://creativecommons.org/licenses/by-nc-sa/4.0/.

This is not an officially supported Google product.

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY-NC-SA 4.0 licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

GenCast and GraphCast are part of an experimental research project. You are solely responsible for determining the appropriateness of using or distributing GenCast, GraphCast or any outputs generated and assume all risks associated with your use or distribution of GenCast, GraphCast and outputs and your exercise of rights and permissions granted by Google to you under the relevant License. Use discretion before relying on, publishing, downloading or otherwise using GenCast, GraphCast or any outputs generated. GenCast, GraphCast or any outputs generated (i) are not based on data published by; (ii) have not been produced in collaboration with; and (iii) have not been endorsed by any government meteorological agency or department and in no way replaces official alerts, warnings or notices published by such agencies.

Copyright 2024 DeepMind Technologies Limited.


## Citations

If you use this work, consider citing our papers ([blog post](https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/), [Science](https://www.science.org/doi/10.1126/science.adi2336), [arXiv](https://arxiv.org/abs/2212.12794), [arxiv GenCast](https://arxiv.org/abs/2312.15796)):

```latex
@article{lam2023learning,
  title={Learning skillful medium-range global weather forecasting},
  author={Lam, Remi and Sanchez-Gonzalez, Alvaro and Willson, Matthew and Wirnsberger, Peter and Fortunato, Meire and Alet, Ferran and Ravuri, Suman and Ewalds, Timo and Eaton-Rosen, Zach and Hu, Weihua and others},
  journal={Science},
  volume={382},
  number={6677},
  pages={1416--1421},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```


```latex
@article{price2023gencast,
  title={GenCast: Diffusion-based ensemble forecasting for medium-range weather},
  author={Price, Ilan and Sanchez-Gonzalez, Alvaro and Alet, Ferran and Andersson, Tom R and El-Kadi, Andrew and Masters, Dominic and Ewalds, Timo and Stott, Jacklynn and Mohamed, Shakir and Battaglia, Peter and Lam, Remi and Willson, Matthew},
  journal={arXiv preprint arXiv:2312.15796},
  year={2023}
}
```

## Acknowledgements

The (i) GenCast and GraphCast communicate with and/or reference with the following separate libraries and packages and the colab notebooks include a few examples of ECMWF’s ERA5 and HRES data that can be used as input to the models.
Data and products of the European Centre for Medium-range Weather Forecasts (ECMWF), as modified by Google.
Modified Copernicus Climate Change Service information 2023. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.
ECMWF HRES datasets
Copyright statement: Copyright "© 2023 European Centre for Medium-Range Weather Forecasts (ECMWF)".
Source: www.ecmwf.int
License Statement: ECMWF open data is published under a Creative Commons Attribution 4.0 International (CC BY 4.0). https://creativecommons.org/licenses/by/4.0/
Disclaimer: ECMWF does not accept any liability whatsoever for any error or omission in the data, their availability, or for any loss or damage arising from their use.

Use of the third-party materials referred to above may be governed by separate terms and conditions or license provisions. Your use of the third-party materials is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use.


## Contact

For feedback and questions, contact us at gencast@google.com.
