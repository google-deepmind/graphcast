# WeatherNext

## WeatherNext 2

This repo contains the code for WeatherNext 2 (WN2), the global, medium-range
atmospheric and cyclone forecasting model developed by Google DeepMind and
Google Research.

It also contains code for prior generation models
[GraphCast](https://deepmind.google/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/)
and
[GenCast](https://deepmind.google/blog/gencast-predicts-weather-and-the-risks-of-extreme-conditions-with-sota-accuracy/).

### Accessing Forecast Data Feeds
If you are interested in directly accessing daily data feeds of WN2 model
outputs rather than running the model yourself, we provide them across multiple
platforms:

*   [Google Cloud](https://developers.google.com/weathernext/guides/access-forecast)
(including Earth Engine, BigQuery, and Vertex AI).
*   [WeatherLab](https://deepmind.google.com/science/weatherlab) (including cyclone tracks).
*   [OpenMeteo](https://open-meteo.com/en/docs/google-weathernext-api) (including an API and interactive builder).

### Learn More

*   **Model Guide & Documentation:** [Google Developers WeatherNext Guide](https://developers.google.com/weathernext/guides/models)
*   **WeatherNext Cyclones Paper:** [Operational tropical cyclone forecasting
    with AI](https://www.nature.com/articles/s41586-026-10953-2)
*   **FGN/WN2 Technical Report:** [Skillful joint probabilistic weather
    forecasting from marginals
    (arXiv:2506.10772)](https://arxiv.org/abs/2506.10772)
*   **WeatherNext 2 Blog Post:** [WeatherNext 2: Our most advanced weather forecasting model](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/weathernext-2/)
*   **WeatherNext Cyclones Blog Post:** [WeatherNext: AI model achieves breakthrough in forecasting cyclones](https://deepmind.google/blog/weathernext-ai-model-achieves-breakthrough-in-forecasting-cyclones/)

### Older Models

This repository serves as the primary home for the WeatherNext family models.
Alongside WN2, this repository also hosts the code and documentation for our
legacy and specialized models:

*   [WeatherNext Graph](docs/weathernext1_graph/README.md): Deterministic
    medium-range weather forecasting using graph neural networks. Published as
    GraphCast.
*   [WeatherNext Gen](docs/weathernext1_gen/README.md): Diffusion-based ensemble
    forecasting for medium-range weather. Published as GenCast.

## Provided Pretrained Models

This repository provides code to run the different versions of WeatherNext 2 and
WeatherNext Cyclones. The only difference between them is that WN2 can also
predict 100m wind. In particular, WN2 also forecasts cyclones with the exact
same algorithm as WN Cyclones. Their weights are different due to independent
training runs.

### WeatherNext 2

1.  **WeatherNext2_<2025** (Used Operationally): 0.25° resolution (~30km).
    Fine-tuned on ECMWF HRES data and designed to be initialized directly from
    operational HRES initial conditions rather than ERA5 reanalysis. Trained on
    data through 2024. Corresponding weights files:
    `WeatherNext2_<2025_model{1,2,3,4}.npz`.

### WeatherNext Cyclones - models which reproduce the results in paper

1.  **WeatherNextCyclones_<2025** (Used Operationally): 0.25° resolution. The
    model that ran live during the 2025 Atlantic hurricane season, publicly
    referred to as FNV3 (NHC's postprocessed version was called GDMI). Trained
    on data through 2024. The paper appendix contains a partial evaluation of
    2025 in NHC basins for this model checkpoint, and how the tracker
    improvement in September 2025 improved results. Corresponding weights files:
    `WeatherNextCyclones_<2025_model{1,2,3,4}.npz`.
2.  **WeatherNextCyclones_<2024**: 0.25° resolution. Reproduces results from the
    paper on 2024. Trained on data through 2023. Corresponding weights files:
    `WeatherNextCyclones_<2024_model{1,2,3,4}.npz`.
3.  **WeatherNextCyclones_<2023**: 0.25° resolution. Reproduces results from the
    paper on 2023. Trained on data through 2022. Corresponding weights files:
    `WeatherNextCyclones_<2023_model{1,2,3,4}.npz`.

### WeatherNext Cyclones Mini

1.  **WeatherNextCyclones_Mini_<2024**: 1° resolution. A lightweight version
    suitable for lower memory and compute constraints (e.g., local testing or
    single TPUs or GPUs). Not expected to match the performance of the larger
    versions. Forecasts the same things as WeatherNext2_<2025, including
    cyclones. Trained on data through 2023. Corresponding weights file:
    `WeatherNextCyclones_Mini_<2024.npz`.
2.  **WeatherNextCyclones_Mini_<2023**: As above, but only trained on data through
    2022. Corresponding weights file: `WeatherNextCyclones_Mini_<2023.npz`.

Evaluation results for WeatherNextCyclones_Mini can be found in the appendix
of the
[WeatherNext Cyclones Paper](https://www.nature.com/articles/s41586-026-10953-2).

## Quick Start Guide

The easiest way to get started with WeatherNext 2 is by running our interactive
[Colab Notebook](docs/weathernext2/wn2_demo.ipynb), which can be opened from
[Colaboratory](https://colab.research.google.com/github/google-deepmind/weathernext/blob/master/docs/weathernext2/wn2_demo.ipynb).
This notebook defaults to WeatherNext Cyclones Mini, which we recommend running
using the `v5e-1` runtime, available for free as a Colab runtime. However, the
notebook can also be used to run the other models enumerated above (but these
will require a `v5p` accelerator).

In general, we recommend running WeatherNext 2 on TPU where possible, since its
implementation has been optimised for it. However, if choosing to run on GPU,
the attention implementation must be switched, as shown in the demo notebook.
The non-Mini models require H100 for sufficient VRAM. The Mini models should
manage inference on a P100.

Pre-trained weights and sample data are available on our [Google Cloud
Bucket](https://console.cloud.google.com/storage/browser/dm_graphcast).

**Inside the notebook, you will learn how to:**

1.  Automatically load the required model weights from our storage bucket.
2.  Load initial state weather data (e.g., HRES initial conditions).
3.  Initialize the WN2 (FGN) architecture.
4.  Run auto-regressive rollout steps to generate a forecast prediction.
5.  Visualize the outputs (e.g., temperature, wind speed, geopotential height).
6.  Run the direct tracker on model outputs to obtain track data for cyclones.
7.  Compute the training loss on model predictions and targets, and take a
    gradient step.

## Setup

### Installation

> [!NOTE] This is research code provided as-is for the purpose of running and
> experimenting with the published models. There are no guarantees of API
> stability and future updates may introduce breaking changes without notice. We
> recommend pinning to a specific release.

E.g.:

```bash
pip install git+https://github.com/google-deepmind/weathernext.git@v0.3.0
```

### Model Weights

To run WeatherNext 2 or WeatherNext Cyclones, you will need to download the
pre-trained model weights. You can access the weights on [Google Cloud
Bucket](https://console.cloud.google.com/storage/browser/dm_graphcast).

### Shared Utilities

The `utils/` directory contains shared libraries used by multiple WeatherNext
models, providing common infrastructure for autoregressive rollouts, input
normalization, graph building blocks, loss computation, and JAX-compatible
xarray utilities. See the per-model READMEs for model-specific code.

### Training Data

Full model training requires downloading the
[ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)
dataset from [ECMWF](https://www.ecmwf.int/), best accessed as Zarr via
[WeatherBench2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5).

Operational fine-tuning data is available via [WeatherBench2's HRES
data](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#ifs-hres-t-0-analysis).

These datasets may be governed by separate terms and conditions. Check that you
can comply with any applicable restrictions before use.

## License

Copyright 2026 Google LLC.

The Colab notebooks and the associated code are licensed under the Apache
License, Version 2.0 (Apache 2.0); you may not use these materials except in
compliance with the Apache 2.0 license. You may obtain a copy of the License at:
https://www.apache.org/licenses/LICENSE-2.0.

All other materials are licensed under the Creative Commons Attribution 4.0
International (CC BY 4.0). You may obtain a copy of the License at:
[https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY 4.0 licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

## Disclaimers

This is not an officially supported Google product.

The WeatherNext models are part of an experimental research project. You are
solely responsible for determining the appropriateness of using or distributing
these models or any outputs they generate, and you assume all risks associated
with such use or distribution and your exercise of rights and permissions
granted by Google under the relevant license. Use discretion before relying on,
publishing, downloading, or otherwise using these models or any of their
outputs.

The WeatherNext models have not been produced in collaboration with nor endorsed
by any government meteorological agency or department, and in no way replaces
official alerts, warnings or notices published by such agencies.

## Citations

If you use WeatherNext 2 in your research, please cite our papers:

<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->

```latex
@article{Alet2026,
  title={Operational Tropical Cyclone Forecasting with AI},
  author={Alet, Ferran and Andersson, Tom R. and Price, Ilan and Markou, Stratis and El-Kadi, Andrew and Masters, Dominic and Li, Amy and Merchant, Samier and Williams, Natalie and Thornton, Gregory and MacKay, Ken and Graham, Olivia and Uddin, Akib and Gaiarin, Ben and Shah, Devaja and Kruse, Elinor and Hogsett, Wallace and Zelinsky, David and Cangialosi, John and Martinez, Jonathan and Franklin, James and DeMaria, Mark and Musgrave, Kate and Bain, Caroline L. and Titley, Helen and Stott, Jacklynn and Lam, Remi and Bell, Aaron and Komarek, Paul and Willson, Matthew and Sanchez-Gonzalez, Alvaro and Battaglia, Peter},
  journal={Nature},
  year={2026},
  issn={1476-4687},
  doi={10.1038/s41586-026-10953-2},
  url={https://doi.org/10.1038/s41586-026-10953-2}
}
```

```latex
@article{alet2025skillful,
  title={Skillful joint probabilistic weather forecasting from marginals},
  author={Alet, Ferran and Price, Ilan and El-Kadi, Andrew and Masters, Dominic and Markou, Stratis and Andersson, Tom R and Stott, Jacklynn and Lam, Remi and Willson, Matthew and Sanchez-Gonzalez, Alvaro and Battaglia, Peter},
  journal={arXiv preprint arXiv:2506.10772},
  year={2025}
}
```

## Acknowledgements

The WeatherNext models communicate with the following separate libraries and
packages:.

*   Data and products of the European Centre for Medium-range Weather Forecasts
    (ECMWF), as modified by Google.
*   Modified Copernicus Climate Change Service information 2023\.
*   NOAA's International Best Track Archive for Climate Stewardship (IBTrACS)
    data, first accessed on 1 Dec 2022\.

Additionally, the colab notebooks include a few examples of ECMWF’s ERA5 and
HRES data that can be used as input to the models.

Neither the European Commission nor ECMWF is responsible for any use that may be
made of the Copernicus information or data it contains. ECMWF HRES datasets
Copyright statement: Copyright "© 2023 European Centre for Medium-Range Weather
Forecasts (ECMWF)". Source: [www.ecmwf.int](http://www.ecmwf.int/) License
Statement: ECMWF open data is published under a Creative Commons Attribution 4.0
International (CC BY 4.0).
[https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)
Disclaimer: ECMWF does not accept any liability whatsoever for any error or
omission in the data, their availability, or for any loss or damage arising from
their use.

Use of the third-party materials referred to above may be governed by separate
terms and conditions or license provisions. Your use of the third-party
materials is subject to any such terms and you should check that you can comply
with any applicable restrictions or terms and conditions before use.

## Contact

For feedback and questions regarding the codebase or models, contact us at
`weathernext@google.com`.

Any information collected via email will be used in accordance with [Google's
privacy policy](http://policies.google.com/privacy).
