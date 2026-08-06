## WeatherNext 1 Graph (GraphCast)

This model was published as [GraphCast: Learning skillful medium-range global
weather forecasting](https://www.science.org/doi/10.1126/science.adi2336). File
and checkpoint names retain the original "GraphCast" naming for backward
compatibility.

This package provides three pretrained models:

1.  `GraphCast`, the high-resolution model used in the GraphCast paper (0.25
    degree resolution, 37 pressure levels), trained on ERA5 data from 1979 to
    2017,

2.  `GraphCast_small`, a smaller, low-resolution version of GraphCast (1 degree
    resolution, 13 pressure levels, and a smaller mesh), trained on ERA5 data
    from 1979 to 2015, useful to run a model with lower memory and compute
    constraints,

3.  `GraphCast_operational`, a high-resolution model (0.25 degree resolution, 13
    pressure levels) pre-trained on ERA5 data from 1979 to 2017 and fine-tuned
    on HRES data from 2016 to 2021. This model can be initialized from HRES data
    (does not require precipitation inputs).

The best starting point is to open `graphcast_demo.ipynb` in
[Colaboratory](https://colab.research.google.com/github/google-deepmind/weathernext/blob/master/docs/weathernext1_graph/graphcast_demo.ipynb),
which gives an example of loading data, generating random weights or load a
pre-trained snapshot, generating predictions, computing the loss and computing
gradients. The one-step implementation of GraphCast architecture, is provided in
`graphcast.py` and the relevant data, weights and statistics are in the
`graphcast/` subdir of the Google Cloud Bucket.

WARNING: For backwards compatibility, we have also left GraphCast data in the
top level of the bucket. These will eventually be deleted in favour of the
`graphcast/` subdir.

### Brief description of relevant library files:

In addition to the model-specific files below, shared library files
(`autoregressive.py`, `normalization.py`, `rollout.py`, etc.) are in the
`utils/` directory.

*   `casting.py`: Wrapper used around GraphCast to make it work using BFloat16
    precision.
*   `graphcast.py`: The main GraphCast model architecture for one-step of
    predictions.
*   `solar_radiation.py`: Computes Top-Of-the-Atmosphere (TOA) incident solar
    radiation compatible with ERA5. This is used as a forcing variable and thus
    needs to be computed for target lead times in an operational setting.

## License and Disclaimers

Copyright 2024-2026 Google LLC.

The Colab notebooks and the associated code are licensed under the Apache
License, Version 2.0 (Apache 2.0); you may not use these materials except in
compliance with the Apache 2.0 license. You may obtain a copy of the License at:
[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0).

All other materials are licensed under the Creative Commons Attribution 4.0
International (CC BY 4.0). You may obtain a copy of the License at:
[https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

***August 6, 2026 UPDATE:** The license for the model weights in this repository
has been updated to permit commercial use. The previous license is now replaced
by the new terms outlined in this README By continuing to access, download, or
use the weights (including those previously sourced from this repository), you
agree to the updated terms.*

This is not an officially supported Google product.

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY 4.0 licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

GraphCast is part of an experimental research project. You are solely
responsible for determining the appropriateness of using or distributing
GraphCast or any outputs generated and assume all risks associated with your use
or distribution of GraphCast and outputs and your exercise of rights and
permissions granted by Google to you under the relevant License. Use discretion
before relying on, publishing, downloading or otherwise using GraphCast or any
outputs generated. \[GraphCast outputs have not been produced in collaboration
with nor endorsed by any government meteorological agency or department, and in
no way replaces official alerts, warnings or notices published by such
agencies.\]

## Citations

If you use this work, consider citing our papers ([blog
post](https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/),
[Science](https://www.science.org/doi/10.1126/science.adi2336)):

<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->

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

## Acknowledgements

GraphCast communicates with the following separate libraries and packages:.

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
