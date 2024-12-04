# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module setuptools script."""

from setuptools import setup

description = (
    "GraphCast: Learning skillful medium-range global weather forecasting"
)

setup(
    name="graphcast",
    version="0.2.0.dev",
    description=description,
    long_description=description,
    author="DeepMind",
    license="Apache License, Version 2.0",
    keywords="GraphCast Weather Prediction",
    url="https://github.com/deepmind/graphcast",
    packages=["graphcast"],
    install_requires=[
        "cartopy",
        "chex",
        "colabtools",
        "dask",
        "dinosaur-dycore",
        "dm-haiku",
        "dm-tree",
        "jax",
        "jraph",
        "matplotlib",
        "numpy",
        "pandas",
        "rtree",
        "scipy",
        "trimesh",
        "typing_extensions",
        "xarray",
        "xarray_tensorstore"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
