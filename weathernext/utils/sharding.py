# Copyright 2026 Google LLC.
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

"""Utilities for sharding the training and inference of weather models."""

from collections.abc import Sequence
import math

import jax
from jax.interpreters import pxla

# Conventions for device mesh axis names:

# Used for standard data parallelism.
BATCH_AXIS = 'batch'

# Used when generating multiple ensemble members or 'samples' for each example
# in the batch.
# At inference time these ensemble members don't interact at all, however if
# multiple samples are used for training, they will interact in the training
# loss, generally in a more complicated way than a plain sum-of-losses that we
# see with batch/data parallelism.
# We also need to differentiate between local and process-sharded ensemble
# axes as the data readers are not sharded across processes and we therefore
# have to put data on to just the local process and replicate it across
# processes.
LOCAL_ENSEMBLE_AXIS = 'sample_local'
ENSEMBLE_PROCESS_AXIS = 'sample_process'

# All ensemble axes, including local and global.
ENSEMBLE_AXES = (ENSEMBLE_PROCESS_AXIS, LOCAL_ENSEMBLE_AXIS)

# It's common to stack batch and ensemble dimensions into a single batch
# dimension, which we shard over both BATCH_AXIS and ENSEMBLE_AXIS.
BATCH_LIKE_AXES = (BATCH_AXIS, *ENSEMBLE_AXES)

# For partitioning spatial data and/or spatial activations within a single
# process.
# We also need to differentiate between local and process-sharded axes here too.
LOCAL_SPATIAL_AXIS = 'spatial_local'
SPATIAL_PROCESS_AXIS = 'spatial_process'

# All spatial axes, including local and global.
SPATIAL_AXES = (SPATIAL_PROCESS_AXIS, LOCAL_SPATIAL_AXIS)

# Used for vmapping a leading dimension, which can be useful to be able
# to feed different rng seeds, and helps replacing former uses of pmap.
OUTER_VMAP_AXIS = 'outer_vmap'


def get_global_mesh() -> jax.sharding.Mesh:
  """Returns pjit global mesh."""
  return pxla.thread_resources.env.physical_mesh


def is_global_mesh_defined() -> bool:
  """Checks if pjit global mesh is defined."""
  return get_global_mesh().devices.shape != ()  # pylint: disable=g-explicit-bool-comparison


def get_num_shards(axis_name: str | Sequence[str]) -> int:
  if isinstance(axis_name, str):
    axis_name = [axis_name]
  if is_global_mesh_defined():
    return math.prod(get_global_mesh().shape[name] for name in axis_name)
  else:
    return 1
