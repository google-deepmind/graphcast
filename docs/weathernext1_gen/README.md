## WeatherNext 1 Gen (GenCast)

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

This model was published as [GenCast: Diffusion-based ensemble forecasting for
medium-range weather](https://arxiv.org/abs/2312.15796). File and checkpoint
names retain the original "GenCast" naming for backward compatibility.

This package provides four pretrained models:

1.  `GenCast 0p25deg <2019`, GenCast model at 0.25deg resolution with 13
    pressure levels and a 6 times refined icosahedral mesh. This model is
    trained on ERA5 data from 1979 to 2018 (inclusive), and can be causally
    evaluated on 2019 and later years. This model was described in the paper
    `GenCast: Diffusion-based ensemble forecasting for medium-range weather`
    (https://arxiv.org/abs/2312.15796)

2.  `GenCast 0p25deg Operational <2022`, GenCast model at 0.25deg resolution,
    with 13 pressure levels and a 6 times refined icosahedral mesh. This model
    is trained on ERA5 data from 1979 to 2018, and fine-tuned on HRES-fc0 data
    from 2016 to 2021 and can be causally evaluated on 2022 and later years.
    This model can make predictions in an operational setting (i.e., initialised
    from HRES-fc0)

3.  `GenCast 1p0deg <2019`, GenCast model at 1deg resolution, with 13 pressure
    levels and a 5 times refined icosahedral mesh. This model is trained on ERA5
    data from 1979 to 2018, and can be causally evaluated on 2019 and later
    years. This model has a smaller memory footprint than the 0.25deg models

4.  `GenCast 1p0deg Mini <2019`, GenCast model at 1deg resolution, with 13
    pressure levels and a 4 times refined icosahedral mesh. This model is
    trained on ERA5 data from 1979 to 2018, and can be causally evaluated on
    2019 and later years. This model has the smallest memory footprint of those
    provided and has been provided to enable low cost demonstrations (for
    example, it is runnable in a free Colab notebook). While its performance is
    reasonable, it is not representative of the performance of the GenCast
    models (1-3) above. For reference, a scorecard comparing its performance to
    ENS can be found in
    [docs/](GenCast_1p0deg_Mini_ENS_scorecard.png).
    Note that in this scorecard, GenCast Mini only uses 8 member ensembles (vs.
    ENS' 50) so we use the fair (unbiased) CRPS to allow for fair comparison.

The best starting point is to open `gencast_mini_demo.ipynb` in
[Colaboratory](https://colab.research.google.com/github/google-deepmind/weathernext/blob/master/docs/weathernext1_gen/gencast_mini_demo.ipynb),
which gives an example of loading data, generating random weights or loading a
`GenCast 1p0deg Mini <2019` snapshot, generating predictions, computing the loss
and computing gradients. The one-step implementation of GenCast architecture is
provided in `gencast.py` and the relevant data, weights and statistics are in
the `gencast/` subdir of the Google Cloud Bucket.

### Instructions for running GenCast on Google Cloud compute

[cloud_vm_setup.md](cloud_vm_setup.md) contains
detailed instructions on launching a Google Cloud TPU VM. This provides a means
of running models (1-3) in the separate `gencast_demo_cloud_vm.ipynb` through
[Colaboratory](https://colab.research.google.com/github/google-deepmind/weathernext/blob/master/docs/weathernext1_gen/gencast_demo_cloud_vm.ipynb).

The document also provides
[instructions](cloud_vm_setup.md#running-inference-on-gpu)
for running GenCast on a GPU. This requires using a different attention
implementation.

### Brief description of relevant library files

In addition to the model-specific files below, shared library files
(`autoregressive.py`, `normalization.py`, `rollout.py`, etc.) are in the
`utils/` directory.

*   `denoiser.py`: The GenCast denoiser for one step predictions.
*   `denoisers_base.py`: Defines the interface of the denoiser.
*   `dpm_solver_plus_plus_2s.py`: Sampler using DPM-Solver++ 2S from [1].
*   `gencast.py`: Combines the GenCast model architecture, wrapped as a
    denoiser, with a sampler to generate predictions.
*   `nan_cleaning.py`: Wraps a predictor to allow it to work with data cleaned
    of NaNs. Used to remove NaNs from sea surface temperature.
*   `samplers_base.py`: Defines the interface of the sampler.
*   `samplers_utils.py`: Utility methods for the sampler.
*   `sparse_transformer.py`: General purpose sparse transformer that operates on
    `TypedGraph`'s where both inputs and outputs are flat vectors of features
    for each of the nodes and edges. `predictor.py` uses one of these for the
    mesh GNN.
*   `sparse_transformer_utils.py`: Utility methods for the sparse transformer.
*   `transformer.py`: Wraps the mesh transformer, swapping the leading two axes
    of the nodes in the input graph.

[1] DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic
Models, https://arxiv.org/abs/2211.01095

## License and Disclaimers

Copyright 2024-2026 Google LLC.

The Colab notebooks and the associated code are licensed under the Apache
License, Version 2.0 (Apache 2.0); you may not use these materials except in
compliance with the Apache 2.0 license. You may obtain a copy of the License at:
[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0).

All other materials are licensed under the terms of the Creative Commons
Attribution 4.0 International (CC BY 4.0). You may obtain a copy of the License
at:
[https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

***August 6, 2026 UPDATE:** The license for the model weights in this repository
has been updated to permit commercial use. The previous license is now replaced
by the new terms outlined in this README. By continuing to access, download, or
use the weights (including those previously sourced from this repository), you
agree to the updated terms.*

This is not an officially supported Google product.

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY 4.0 licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

GenCast is part of an experimental research project. You are solely responsible
for determining the appropriateness of using or distributing GenCast or any
outputs generated and assume all risks associated with your use or distribution
of GenCast and outputs and your exercise of rights and permissions granted by
Google to you under the relevant License. Use discretion before relying on,
publishing, downloading or otherwise using GenCast or any outputs generated.
\[GenCast outputs have not been produced in collaboration with nor endorsed by
any government meteorological agency or department, and in no way replaces
official alerts, warnings or notices published by such agencies.\]

## Citations

If you use this work, consider citing our papers ([blog
post](https://deepmind.google/blog/gencast-predicts-weather-and-the-risks-of-extreme-conditions-with-sota-accuracy/),
[Nature](https://www.nature.com/articles/s41586-024-08252-9)):

<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->

```latex
@article{price2024gencast,
  title={Probabilistic weather forecasting with machine learning},
  author={Price, Ilan and Sanchez-Gonzalez, Alvaro and Alet, Ferran and Andersson, Tom R and El-Kadi, Andrew and Masters, Dominic and Ewalds, Timo and Stott, Jacklynn and Mohamed, Shakir and Battaglia, Peter and Lam, Remi and Willson, Matthew},
  journal={Nature},
  volume={637},
  number={8044},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.1038/s41586-024-08252-9}
}
```

## Acknowledgements

GenCast communicates with the following separate libraries and packages:.

*   Data and products of the European Centre for Medium-range Weather Forecasts
    (ECMWF), as modified by Google.
*   Modified Copernicus Climate Change Service information 2023\.

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
