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

"""Accelerated Gather Scatter operations."""

from collections.abc import Callable
import functools
from typing import Any

from absl import logging
import chex
from weathernext.utils import sharding
import jax
import jax.numpy as jnp
import jraph
import numpy as np


def is_sorted(indices: np.ndarray) -> bool:
  """Returns True if the indices are sorted."""
  indices = indices[indices >= 0]
  return all([a == b for a, b in zip(indices, sorted(indices))])


def global_to_shard_local_indexing(
    input_indices: np.ndarray, spatial_shards: int, global_index_range: int
) -> np.ndarray:
  """Reindex the indices to be local to the spatial shard.

  Reindex the indices to be relative to the starting index of the shard. This
  means that the indices can be used independently per shard in a shard map.

  Args:
    input_indices: The indices to be reindexed.
    spatial_shards: The number of spatial shards.
    global_index_range: Range of index values for the input indices. For a
      scatter this relates to the size of the output, for a gather the size of
      the input.

  Returns:
    Indices reindexed to be shard local.
  """
  logging.info('Reindex indices to be shard local...')
  # Padding should be applied at the graph level to ensure this is true.
  if global_index_range % spatial_shards != 0:
    raise ValueError(
        f'global_index_range ({global_index_range}) must be a multiple of'
        f' spatial_shards ({spatial_shards}).'
    )
  # First we reshape the indices to get the indices per spatial shard.
  indices = input_indices.reshape(spatial_shards, -1)
  padding_mask = indices == -1  # Assumes we use -1 for padding indices.
  # Determine the range of values for each shard.
  shard_local_index_range = global_index_range // spatial_shards
  # Reindex relative to the starting index of the shard.
  shard_starting_idx = np.arange(
      0, global_index_range, shard_local_index_range
  )[..., np.newaxis]
  indices = indices - shard_starting_idx
  indices[padding_mask] = -1  # Reset padding indices.
  return indices.reshape(-1)


def indices_all_in_range(indices: np.ndarray, index_range: int) -> bool:
  """Check if all indices are in range [0, index_range).

  Args:
    indices: The indices to be checked.
    index_range: Range of index values allowed. For a scatter this relates to
      the size of the output, for a gather it is the size of the input.

  Returns:
    True if all indices are in range, False otherwise.
  """
  padding_mask = indices == -1  # Assumes we use -1 for padding indices.
  # Check if all indices fall within the index_range.
  in_range = np.logical_or(
      np.logical_and(indices >= 0, indices < index_range), padding_mask
  )
  all_in_range = np.all(in_range)
  logging.info(
      'all_in_range: %s, indices_size: %d, in_range_count: %d (%2.1f%%)',
      all_in_range,
      in_range.size,
      np.sum(in_range),
      100 * np.sum(in_range) / in_range.size,
  )
  return all_in_range  # pyrefly: ignore[bad-return]


def maybe_to_shard_local_indexing(
    indices: np.ndarray,
    spatial_shards: int,
    global_index_range: int,
    remove_collectives_if_local: bool = True,
) -> tuple[jnp.ndarray, int, bool]:
  """Reindex indices to be local to their spatial shard if all are local.

  If remove_collectives_if_local is True, we will try and reindex the indices
  from "global" indexing to "shard local" indexing. This means that the indices
  will be defined relative to the start of the shard rather than the start of
  the global indices. If in, the shard local indexing, all indices are in range
  [0, local_index_range), then they are all 'local' and we can remove the
  collectives in the shard map. If not we return the original globally indexed
  indices.

  Args:
    indices: The indices to be preprocessed.
    spatial_shards: The number of spatial shards.
    global_index_range: Range of index values for the input indices. For a
      scatter this relates to the size of the output, for a gather the size of
      the input.
    remove_collectives_if_local: If True, and all indices only reference values
      on the same shard then we define the indices as local and can remove the
      collectives in the shard map.

  Returns:
    A tuple containing the preprocessed indices, the new index range, and a
    boolean indicating whether the indices are local (which indicates if indices
    and index_range have been modified).
  """
  index_range = global_index_range
  is_local = False
  if remove_collectives_if_local:
    shard_local_indices = global_to_shard_local_indexing(
        indices, spatial_shards, global_index_range)
    local_index_range = global_index_range // spatial_shards
    is_local = indices_all_in_range(shard_local_indices, local_index_range)
    if is_local:
      indices = shard_local_indices
      index_range = local_index_range
  indices = jnp.array(indices)  # pyrefly: ignore[bad-assignment]
  return indices, index_range, is_local  # pyrefly: ignore[bad-return]


def gather_with_fill(
    x: jnp.ndarray, indices: jnp.ndarray, indices_are_sorted: bool = False
) -> jnp.ndarray:
  """Wrapper around jax.lax.gather with out of bounds values filled with zero.

  For cases with -1 used for padding, this is a safer option than x[idx].

  Note that the performance of the gather does not improve by setting
  indices_are_sorted=True, however the performance of the corresponding
  backwards segment_sum will be.

  Args:
    x: Matrix to gather rows from
    indices: Rows to be gathered
    indices_are_sorted: If indices are sorted

  Returns:
    Matrix of gathered rows
  """
  indices = indices[:, jnp.newaxis]
  offset_dims = tuple(range(1, x.ndim))
  dnums = jax.lax.GatherDimensionNumbers(
      offset_dims=offset_dims, collapsed_slice_dims=(0,), start_index_map=(0,)
  )
  return jax.lax.gather(
      x,
      indices,
      dimension_numbers=dnums,
      slice_sizes=[1, *x.shape[1:]],
      unique_indices=False,
      indices_are_sorted=indices_are_sorted,
      mode='fill',
      fill_value=0,
  )


def sorted_gather(x: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
  return gather_with_fill(x, indices, indices_are_sorted=True)


def create_static_segment_sum(
    indices: chex.Array,
    num_output_rows: int,
    indices_are_sorted: bool = False,
    check_is_sorted: bool = False,
    explicit_sharding: bool = True,
    remove_collectives_if_local: bool = True,
    _use_custom_vjp: bool = True,  # pylint: disable=invalid-name
) -> Callable[..., jnp.ndarray]:
  """Constructor for a segment_sum op with static indices.

  Optionally sets indices_are_sorted=True and additionally checks the indices
  are sorted AOT and fixes them for future calls.

  Any indices given to the produced op at runtime are ignored.

  Args:
    indices: indices of the output rum to sum with
    num_output_rows: number of output rows
    indices_are_sorted: Set indices_are_sorted in the segment_sum op
    check_is_sorted: If True check that the indieces are sorted
    explicit_sharding: If True, explicitly shard with a shard_map rather than
      allowing PJIT to determine the sharding.
    remove_collectives_if_local: Check if indices imply only local connectivity,
      if True reindex indices and remove collectives.
    _use_custom_vjp: If True, define a custom VJP for the segment sum using a
      static gather for the backward pass.

  Returns:
    Static sorted segment sum function with fixed indices.
  """
  logging.info('Building static_segment_sum op...')
  if indices_are_sorted and check_is_sorted:
    # Convert to numpy array to ensure this is done AOT and isn't JITed.
    indices = np.array(indices)
    if not is_sorted(indices):
      raise ValueError('Indices are not sorted.')

  static_indices = indices

  def segment_sum_fun(
      x: jnp.ndarray,
      segment_ids: chex.Array | None = None,
      num_segments: int | None = None,
  ):
    # Warn but do not fail with additional arguments to maintain compatibility
    # with dynamic ops.
    if segment_ids is not None or num_segments is not None:
      if segment_ids is not None and isinstance(segment_ids, np.ndarray):
        np.testing.assert_equal(
            segment_ids,
            static_indices,
            'Runtime indices should match the compile time indices.',
        )
        np.testing.assert_equal(
            num_segments,
            num_output_rows,
            'Runtime num_segments should match the compile time num_segments.',
        )
      logging.warning(
          'Only the first input of the sorted segment sum operation is being'
          ' used, others are being ignored at runtime.'
      )
    inner_segment_sum_fun = functools.partial(
        jraph.segment_sum,
        indices_are_sorted=indices_are_sorted,
    )

    if sharding.is_global_mesh_defined() and explicit_sharding:
      data_pspec = jax.sharding.PartitionSpec(
          sharding.SPATIAL_AXES, sharding.BATCH_LIKE_AXES
      )
      indices_pspec = jax.sharding.PartitionSpec(sharding.SPATIAL_AXES)
      global_mesh = sharding.get_global_mesh()
      spatial_shards = sharding.get_num_shards(sharding.SPATIAL_AXES)
      indices, num_segments, is_local = maybe_to_shard_local_indexing(
          indices=static_indices,  # pyrefly: ignore[bad-argument-type]
          spatial_shards=spatial_shards,
          global_index_range=num_output_rows,
          remove_collectives_if_local=remove_collectives_if_local)
      @functools.partial(jax.shard_map,
                         mesh=global_mesh,
                         in_specs=(data_pspec, indices_pspec),
                         out_specs=data_pspec,
                         axis_names={*sharding.SPATIAL_AXES,
                                     *sharding.BATCH_LIKE_AXES,
                                     sharding.OUTER_VMAP_AXIS},
                         check_vma=False)
      def shmapped_segment_sum_fun(x, shard_indices):
        y = inner_segment_sum_fun(x, shard_indices, num_segments)
        if not is_local:
          y = jax.lax.psum_scatter(
              y, sharding.SPATIAL_AXES, scatter_dimension=0, tiled=True
          )
        return y

      return shmapped_segment_sum_fun(x, indices)
    else:
      return inner_segment_sum_fun(
          x,
          segment_ids=jnp.array(static_indices),
          num_segments=num_output_rows)
  if not _use_custom_vjp:
    return segment_sum_fun

  # Explicitly define the backward pass as a gather op. This avoids
  # differentiating through the shard_map body (including collectives) which
  # may not produce the most efficient backward pass.
  gather_op = create_static_gather(
      indices,
      num_input_rows=num_output_rows,
      indices_are_sorted=indices_are_sorted,
      explicit_sharding=explicit_sharding,
      remove_collectives_if_local=remove_collectives_if_local,
      _use_custom_vjp=False,
  )

  @jax.custom_vjp
  def f(
      x: jnp.ndarray,
      segment_ids: chex.Array | None = None,
      num_segments: int | None = None,
  ):
    return segment_sum_fun(x, segment_ids, num_segments)

  def f_fwd(x, segment_ids=None, num_segments=None):
    return f(x, segment_ids, num_segments), ()

  def f_bwd(_, g):
    return gather_op(g), None, None

  f.defvjp(f_fwd, f_bwd)
  return f


def create_static_sorted_segment_sum(
    indices: chex.Array,
    num_output_rows: int,
    check_is_sorted: bool = True,
    explicit_sharding: bool = True,
    remove_collectives_if_local: bool = True
) -> Callable[..., jnp.ndarray]:
  return create_static_segment_sum(
      indices=indices,
      num_output_rows=num_output_rows,
      indices_are_sorted=True,
      check_is_sorted=check_is_sorted,
      explicit_sharding=explicit_sharding,
      remove_collectives_if_local=remove_collectives_if_local,
  )


def create_static_gather(
    indices: chex.Array,
    num_input_rows: Any = None,  # pylint: disable=unused-argument
    indices_are_sorted: bool = False,
    check_is_sorted: bool = True,
    explicit_sharding: bool = True,
    remove_collectives_if_local: bool = True,
    _use_custom_vjp: bool = True,  # pylint: disable=invalid-name
) -> Callable[..., jnp.ndarray]:
  """Constructor for a gather op with static indices.

  Additionally checks the indices are sorted AOT and fixes them for future
  calls.

  Any indices given to the produced op at runtime are ignored.

  Args:
    indices: indices of the rows to gather
    num_input_rows: Unused argument required to match signature of other gather
      constructors
    indices_are_sorted: If True use a sorted version of the gather. This has no
      performance benefit in the forwards pass but allows a more efficient
      sorted segment sum to be used in the backwards pass.
    check_is_sorted: If True check that the indices are actually sorted.
    explicit_sharding: If True, explicitly shard with a shard_map rather than
      allowing PJIT to determine the sharding.
    remove_collectives_if_local: Check if indices imply only local connectivity,
      if True reindex indices and remove collectives.
    _use_custom_vjp: If True, define a custom VJP for the gather using a static
      segment sum for the backward pass.

  Returns:
    Static gather function with fixed indices.
  """
  logging.info('Building static_gather op...')
  if indices_are_sorted and check_is_sorted:
    # Convert to numpy array to ensure this is done AOT and isn't JITed.
    indices = np.array(indices)
    if not is_sorted(indices):
      raise ValueError('Indices are not sorted.')

  static_indices = indices
  logging.warning(
      'Only the first input of the static gather operation is being'
      ' used, others are being ignored at runtime.'
  )

  def gather_fun(x: jnp.ndarray, indices: chex.Array | None = None):
    if indices is not None and isinstance(indices, np.ndarray):
      np.testing.assert_equal(
          static_indices,
          indices,
          'Runtime indices should match the compile time indices.',
      )

    inner_gather_fun = functools.partial(gather_with_fill,
                                         indices_are_sorted=indices_are_sorted)

    if sharding.is_global_mesh_defined() and explicit_sharding:
      data_pspec = jax.sharding.PartitionSpec(
          sharding.SPATIAL_AXES, sharding.BATCH_LIKE_AXES
      )
      indices_pspec = jax.sharding.PartitionSpec(sharding.SPATIAL_AXES)
      global_mesh = sharding.get_global_mesh()
      spatial_shards = sharding.get_num_shards(sharding.SPATIAL_AXES)
      indices, _, is_local = maybe_to_shard_local_indexing(
          indices=static_indices,  # pyrefly: ignore[bad-argument-type]
          spatial_shards=spatial_shards,
          global_index_range=num_input_rows,
          remove_collectives_if_local=remove_collectives_if_local)
      @functools.partial(jax.shard_map,
                         mesh=global_mesh,
                         in_specs=(data_pspec, indices_pspec),
                         out_specs=data_pspec,
                         axis_names={*sharding.SPATIAL_AXES,
                                     *sharding.BATCH_LIKE_AXES,
                                     sharding.OUTER_VMAP_AXIS},
                         check_vma=False)
      def shmapped_gather_fun(x, shard_indices):
        # Reshaping the input here helps remove unnecessary padding in the
        # layout after the all_gather.
        feature_shape = x.shape[1:]
        x = x.reshape(x.shape[0], -1)
        if not is_local:
          x = jax.lax.all_gather(x, sharding.SPATIAL_AXES, tiled=True)
        y = inner_gather_fun(x, shard_indices)
        y = y.reshape(y.shape[0], *feature_shape)
        return y

      return shmapped_gather_fun(x, indices)
    else:
      return inner_gather_fun(x, indices=jnp.array(static_indices))

  if not _use_custom_vjp or num_input_rows is None:
    return gather_fun

  # Explicitly define the backward pass as a segment_sum op. This avoids
  # differentiating through the shard_map body (including collectives) which
  # may not produce the most efficient backward pass.
  segment_sum_op = create_static_segment_sum(
      indices=indices,
      num_output_rows=num_input_rows,
      indices_are_sorted=indices_are_sorted,
      explicit_sharding=explicit_sharding,
      remove_collectives_if_local=remove_collectives_if_local,
      _use_custom_vjp=False,
  )

  @jax.custom_vjp
  def f(x: jnp.ndarray, indices: chex.Array | None = None):
    return gather_fun(x, indices)

  def f_fwd(x, indices=None):
    return f(x, indices), ()

  def f_bwd(_, g):
    return segment_sum_op(g), None

  f.defvjp(f_fwd, f_bwd)
  return f


def create_static_sorted_gather(
    indices: chex.Array,
    num_input_rows: Any = None,
    check_is_sorted: bool = True,
    explicit_sharding: bool = True,
    remove_collectives_if_local: bool = True,
) -> Callable[..., jnp.ndarray]:
  return create_static_gather(
      indices=indices,
      num_input_rows=num_input_rows,
      indices_are_sorted=True,
      check_is_sorted=check_is_sorted,
      explicit_sharding=explicit_sharding,
      remove_collectives_if_local=remove_collectives_if_local,
  )
