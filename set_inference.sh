#!/bin/bash

# Upgrade packages
pip install -U importlib_metadata

# Install GraphCast and dependencies
pip install --upgrade https://github.com/deepmind/graphcast/archive/master.zip

# Install other required packages
pip install google-cloud-storage matplotlib xarray
