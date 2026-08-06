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
import setuptools

description = "This is Google DeepMind's library to run WeatherNext models"

setuptools.setup(
    name="weathernext",
    version="0.3.0",
    description=description,
    long_description=description,
    author="DeepMind",
    license="Apache License, Version 2.0",
    keywords="WeatherNext",
    url="https://github.com/google-deepmind/weathernext",
    packages=setuptools.find_packages(),
    package_data={
        "weathernext.weathernext2": ["configs/*.json"],
    },
    install_requires=[
        "chex",
        "colabtools",
        "dask",
        "dinosaur-dycore",
        "dm-haiku",
        "dm-tree",
        "gdm-xarray-jax @ git+https://github.com/google-deepmind/xarray_jax.git@v0.1.1",  # pylint: disable=line-too-long
        "fiddle",
        "h5netcdf",
        "jax",
        "jraph",
        "matplotlib",
        "numpy",
        "pandas",
        "pyproj",
        "rtree",
        "scipy",
        "trimesh",
        "typing_extensions",
        "xarray<=2026.2.0",
        "xarray_tensorstore",
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
