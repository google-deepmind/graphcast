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
"""Utilities for working with trees of xarray.DataArray (including Datasets).

Note that xarray.Dataset doesn't work out-of-the-box with the `tree` library;
it won't work as a leaf node since it implements Mapping, but also won't work
as an internal node since tree doesn't know how to re-create it properly.

To fix this, we reimplement a subset of `map_structure`, exposing its
constituent DataArrays as leaf nodes. This means it can be mapped over as a
generic container of DataArrays, while still preserving the result as a Dataset
where possible.

This is useful because in a few places we need to handle a general
Mapping[str, DataArray] (where the coordinates might not be compatible across
the constituent DataArrays) but also the special case of a Dataset nicely.

For the result e.g. of a tree.map_structure(fn, dataset), if fn returns None for
some of the child DataArrays, they will be omitted from the returned dataset. If
any values other than DataArrays or None are returned, then we don't attempt to
return a Dataset and just return a plain dict of the results. Similarly if
DataArrays are returned but with non-matching coordinates, it will just return a
plain dict of DataArrays.

Note xarray datatypes are registered with `jax.tree_util` by xarray_jax.py,
but `jax.tree_util.tree_map` is distinct from the `xarray_tree.map_structure`.
as the former exposes the underlying JAX/numpy arrays as leaf nodes, while the
latter exposes DataArrays as leaf nodes.
"""

from typing import Any, Callable

import xarray


def map_structure(func: Callable[..., Any], *structures: Any) -> Any:
  """Maps func through given structures with xarrays. See tree.map_structure."""
  if not callable(func):
    raise TypeError(f'func must be callable, got: {func}')
  if not structures:
    raise ValueError('Must provide at least one structure')

  first = structures[0]
  if isinstance(first, xarray.Dataset):
    data = {k: func(*[s[k] for s in structures]) for k in first.keys()}
    if all(isinstance(a, (type(None), xarray.DataArray))
           for a in data.values()):
      data_arrays = [v.rename(k) for k, v in data.items() if v is not None]
      try:
        return xarray.merge(data_arrays, join='exact')
      except ValueError:  # Exact join not possible.
        pass
    return data
  if isinstance(first, dict):
    return {k: map_structure(func, *[s[k] for s in structures])
            for k in first.keys()}
  if isinstance(first, (list, tuple, set)):
    return type(first)(map_structure(func, *s) for s in zip(*structures))
  return func(*structures)
