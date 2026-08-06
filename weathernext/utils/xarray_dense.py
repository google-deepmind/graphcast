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

"""Dense layers for xarrays."""

from collections.abc import Hashable, Mapping, Sequence
import re
from typing import TypeVar

from absl import logging
import chex
from weathernext.utils import dense
from weathernext.utils import sharding_utils
import haiku as hk
from jax import sharding
import jax.numpy as jnp
import numpy as np
import xarray as xr
import xarray_jax


def coord_to_str(data_array: xr.DataArray, dim: str) -> str:
  """Coordinates to string {dim_name}={comma_separated_coord_values}."""
  # Note the input array, may or may not have the "dim", specifically
  # if the dimension has been split, we expect the dimension not to exist, but
  # the associated coordinates to still exist as scalars.
  coord = data_array.coords[dim].data

  # Note that even when a dim does not have a coord, xarray still return an
  # integer index coord, so we raise a warning in that case.
  if dim not in data_array.coords:
    logging.warning(
        "No coordinates found for dim %s of array %s., using index for param "
        "name. This may make model surgery more error prone if changing "
        "coordinates at fine-tuning time.",
        dim,
        data_array,
    )

  if np.issubdtype(coord.dtype, np.timedelta64):
    # Cast to seconds for timedeltas, so they render nicely.
    coord = coord.astype("timedelta64[s]").astype(np.int64)

  # Increase the rank of the "scalar" case, for the iterator below.
  if coord.ndim == 0:
    coord = np.array([coord])

  assert coord.ndim == 1  # Should always be true at this point.

  # Cast each value of the coordinate to a string, and join them.
  if len(coord) > 1 and np.all(coord == np.arange(len(coord))):
    # For coordinates that are just a range, use a more compact string.
    coord_str = f"range({len(coord)})"
  else:
    coord_str = ",".join(map(str, list(coord)))
  return f"{dim}={coord_str}"


DataArrayMapping = TypeVar(
    "DataArrayMapping", bound=Mapping[Hashable, xr.DataArray])

# Normally in haiku, initializers are not called directly, but by
# `hk.get_parameter`. Even though a a `dtype` argument is passed to
# `get_parameter`, haiku allows setting up custom getters which can change the
# actual dtype during initialization, like we do in our `Bfloat16Cast` wrapper.
#
# In this module we are calling the initializer explicitly, and then passing
# the initialized arrays to a `Constant` initializer. So to play it safe
# numerically, we initialize everything in float32, but pass the correct
# dtype to the `hg.get_parameter`. This way the returned parameter will always
# have the correct dtype, but if there are any custom getters, like
# `Bfloat16Cast`, which would have promoted the dtype during initialization
# to float32, we know we do not lose precision during initialization.
INITIALIZER_DTYPE = np.float32


class DataArrayDictDenseEncoder(hk.Module):
  """Dense encoder for a mapping of data arrays.

  This module is designed to encode a mapping of data arrays using a dense
  layer. The parameters of the first matrix multiply of the linear layer are
  defined such that there is at least one separate parameter array for each
  element in the input mapping, with the possibility to split the parameter
  array of a single element further along dimensions of the element. Crucially,
  even though the parameter arrays for each input are split, the actual matrix
  multiply is done by concatenating the data, and concatenating the parameters,
  so the cost of the matrix multiply is the same as if the inputs had been
  provided pre-concatenated. Also the initialization of the array will be
  performed in the concatenated space (which is crucial for initializers
  that depend on the input/output size).

  For each variable the dimensions are split into three groups:
  * preserved_dims: The dimensions that will be preseved in the output as
    leading dimensions. Coordinates and sizes must be consistent in size and
    main coordinates across all data arrays in the mapping. If a preserved dim
    does not exist in a data array, it will be broadcasted.
  * dims_to_split: Array dimensions across which the variables will have
    separate parameter arrays.
  * dims_to_keep: Remaining dimensions which will be kept together in the
    same parameter array.

  Note the sizes and coordinates for any `dims_to_split` and `dims_to_keep`
  don't need to be consistent across all data arrays in the mapping, which is
  why the input is provided as a mapping, and not as a `xr.Dataset`.

  For example for an input mapping with shapes:
      {"geopotential": [batch, lat, lon, time:(-10, 0), level:(1, 2)],
       "2m_temperature": [batch, lat, lon, time:(10, 20)]}
  and setting:
    preserved_dims = ["batch", "lat", "lon"]
    dims_to_split = ["time"]
    dims_to_keep = ["level"] (implied by the other two).

  The first linear layer will have separate weight parameters with shapes:
    "w_geopotential_time=-10_level=1,2" (2, hidden_size)
    "w_geopotential_time=0_level=1,2" (2, hidden_size)
    "w_2m_temperature_time=10" (1, hidden_size)
    "w_2m_temperature_time=20" (1, hidden_size)

  while all remaining biases and weights will be similar to any other dense
  layer.

  """

  def __init__(
      self,
      *,
      name: str,
      preserved_dims: Sequence[str],
      dims_to_split: Sequence[str] | None,
      hidden_size: int,
      output_size: int,
      num_hidden_layers: int,
      w_init: hk.initializers.Initializer | None = None,
      drop_first_matmul: bool = False,
      remat: bool = False,
      partition_spec: sharding.PartitionSpec | None = None,
      _emulate_dims_to_split_bug: bool = False,
      preserved_dims_sizes: Mapping[str, int] | None = None,
      **dense_kwargs,
  ):
    """Initializes the module.

    Args:
      name: Name of the module.
      preserved_dims: Dimensions that will be preserved in the output as
        leading dimensions. Coordinates and sizes must be consistent in size and
        main coordinates across all data arrays in the mapping. If a preserved
        dim does not exist in a data array, it will be broadcasted. It is
        mandatory that all dimensions that are not preserved dims have a
        corresponding coordinate.
      dims_to_split: Array dimensions across which the variables will have
        separate parameter arrays. These can be regex patterns. A value of None
        means that all dimensions will be split. A an empty tuple means that no
        dimensions will be split.
      hidden_size: See DenseLayer.
      output_size: See DenseLayer.
      num_hidden_layers: See DenseLayer.
      w_init: See DenseLayer.
      drop_first_matmul: See DenseLayer. Must be False.
      remat: Remat the whoel DataArrayDictDenseEncoder layer.
      partition_spec: sharding.PartitionSpec to set the sharding of the arrays.
      _emulate_dims_to_split_bug: See _SplitInputMatMul.
      preserved_dims_sizes: Allows specifying sizes of `preserved_dims`, to
        handle the case where they are not present in any of the input arrays.
      **dense_kwargs: other arguments to pass to DenseLayer.
    """

    super().__init__(name=name)
    self._dense_kwargs = dense_kwargs
    self._output_size = output_size
    self._hidden_size = hidden_size
    self._num_hidden_layers = num_hidden_layers
    self._w_init = w_init
    self._remat = remat
    self._preserved_dims = preserved_dims
    self._dims_to_split = dims_to_split
    self._partition_spec = partition_spec
    self._emulate_dims_to_split_bug = _emulate_dims_to_split_bug
    self._preserved_dims_sizes = preserved_dims_sizes
    # TODO(dominicmasters): Check that only the preserved dimensions are
    # assigned sharding axes.

    # We pass this argument explicitly so it never gets folded into
    # `dense_kwargs` and we can raise an error if it is set to True.
    if drop_first_matmul:
      raise ValueError("`drop_first_matmul` is not supported.")

  def __call__(
      self,
      data_array_mapping: Mapping[Hashable, xr.DataArray],
      norm_conditioning: chex.Array | None = None,
      ) -> chex.Array:
    """Encodes the data array mapping using a dense layer.

    Args:
      data_array_mapping: The data array mapping to encode. Sizes and
          coordinates (if they exist) for preserved dims must be consistent
          across all data arrays in the mapping. Coordinates must exist for
          any dims that are not preserved dims, but their sizes and values
          are not required to be consistent across all data arrays.
      norm_conditioning: The conditioning array to use for norm conditioning
          normalization.

    Returns:
      The encoded data array mapping with shape
      `preserved_dims_shape + [output_size]`.

    """

    def call_function(data_array_mapping: Mapping[Hashable, xr.DataArray],
                      norm_conditioning: chex.Array | None = None):
      if self._num_hidden_layers > 0:
        first_matmul_output_size = self._hidden_size
      else:
        first_matmul_output_size = self._output_size

      after_first_matmul = _SplitInputMatMul(
          name="split_input_matmul",
          preserved_dims=self._preserved_dims,
          dims_to_split=self._dims_to_split,
          output_size=first_matmul_output_size,
          w_init=self._w_init,
          _emulate_dims_to_split_bug=self._emulate_dims_to_split_bug,
          preserved_dims_sizes=self._preserved_dims_sizes,
      )(data_array_mapping)

      if self._partition_spec is not None:
        # TODO(dominicmasters): Consider whether this should be pushed down into
        # _SplitInputMatMul.
        after_first_matmul = sharding_utils.set_sharding(
            after_first_matmul, partition_spec=self._partition_spec)

      return dense.DenseLayer(
          name="shared_dense",
          drop_first_matmul=True,  # First matmul was done above.
          hidden_size=self._hidden_size,
          output_size=self._output_size,
          num_hidden_layers=self._num_hidden_layers,
          w_init=self._w_init,
          **self._dense_kwargs,
      )(after_first_matmul, norm_conditioning=norm_conditioning)

    if self._remat:
      call_function = hk.remat(call_function)

    return call_function(data_array_mapping, norm_conditioning)


class DataArrayDictDenseDecoder(hk.Module):
  """A decoder with the same design principles as DataArrayDictDenseEncoder.

  Similar to `DataArrayDictDenseEncoder`, but now applied to the last linear
  layer, and including both the weight and bias of that last linear layer.

  Because now the input to the linear decoder is a regular array, it is
  required to pass a template of what the output would look like, and the
  output size and splits of the params will be inferred from that.

  For example for an output template with shapes:
      {"geopotential": [batch, lat, lon, time:(-10, 0), level:(1, 2)],
       "2m_temperature": [batch, lat, lon, time:(10, 20)]}
  and setting:
    preserved_dims = ["batch", "lat", "lon"]
    dims_to_split = ["time"]
    dims_to_keep = ["level"] (implied by the other two).

  The last linear layer will have separate bias and weight parameters with
  shapes:
    "w_geopotential_time=-10_level=1,2" (hidden_size, 2)
    "w_geopotential_time=0_level=1,2" (hidden_size, 2)
    "w_2m_temperature_time=10" (hidden_size, 1)
    "w_2m_temperature_time=20" (hidden_size, 1)
    "b_geopotential_time=-10_level=1,2" (2,)
    "b_geopotential_time=0_level=1,2" (2,)
    "b_2m_temperature_time=10" (1,)
    "b_2m_temperature_time=20" (1,)
  """

  def __init__(self,
               *,
               name: str,
               preserved_dims: Sequence[str],
               dims_to_split: Sequence[str] | None,
               hidden_size: int,
               num_hidden_layers: int,
               w_init: hk.initializers.Initializer | None = None,
               b_init: hk.initializers.Initializer | None = None,
               activate_final: bool = False,
               activation_normalization: str | None = None,
               remat: bool = False,
               **dense_kwargs):
    """Initializes the module.

    Args:
      name: Name of the module.
      preserved_dims: Names of the leading dimensions that will be preserved.
      dims_to_split: Array dimensions across which the variables will have
        separate parameter arrays. These can be regex patterns. A value of None
        means that all dimensions will be split. A an empty tuple means that no
        dimensions will be split.
      hidden_size: See DenseLayer.
      num_hidden_layers: See DenseLayer.
      w_init: See DenseLayer.
      b_init: See DenseLayer.
      activate_final: See DenseLayer. Must be False.
      activation_normalization: See DenseLayer. Must be None.
      remat: Remat the whoel DataArrayDictDenseDecoder layer.
      **dense_kwargs: other arguments to pass to DenseLayer.
    """
    super().__init__(name=name)
    self._dense_kwargs = dense_kwargs
    self._hidden_size = hidden_size
    self._num_hidden_layers = num_hidden_layers
    self._w_init = w_init
    self._b_init = b_init
    self._activate_final = activate_final
    self._activation_normalization = activation_normalization
    self._preserved_dims = preserved_dims
    self._dims_to_split = dims_to_split
    self._remat = remat

    # We pass these argument explicitly so they never gets folded into
    # `dense_kwargs` and we can raise an error if the value is incompatible.
    if self._activation_normalization is not None:
      raise ValueError("`activation_normalization` is not supported.")

    if self._activate_final:
      raise ValueError("`activate_final` is not supported.")

  def __call__(
      self,
      inputs: chex.Array,
      output_template: DataArrayMapping) -> DataArrayMapping:
    """Decodes the inputs array into a template-like output with a dense layer.

    Args:
      inputs: The array to decode. Must be of rank len(preserved_dims) + 1.
          Where the leading dimensions are the preserved dims, and the last
          dimension is the hidden size.
      output_template: The template of the output. All arrays must
          have dimensions corresponding to preserved_dims. Any other dimensions
          will be decoded as separate channels.

    Returns:
      The decoded data array mapping.
    """
    def call_function(inputs: chex.Array,
                      output_template: DataArrayMapping):
      sharding_utils.inspect_array_sharding(
          inputs, name=f"{self.name}.input"
      )
      if self._num_hidden_layers > 0:
        before_last_linear = dense.DenseLayer(
            name="shared_dense",
            hidden_size=self._hidden_size,
            output_size=self._hidden_size,
            num_hidden_layers=self._num_hidden_layers - 1,
            activate_final=True,
            activation_normalization=None,
            w_init=self._w_init,
            b_init=self._b_init,
            **self._dense_kwargs,
        )(inputs)
      else:
        before_last_linear = inputs

      output_data_array_mapping = _SplitOutputLinear(
          name="split_output_linear",
          preserved_dims=self._preserved_dims,
          dims_to_split=self._dims_to_split,
          w_init=self._w_init,
          b_init=self._b_init,
      )(before_last_linear, output_template)
      sharding_utils.inspect_xarray_sharding(
          output_data_array_mapping, name=f"{self.name}.output"
      )
      return output_data_array_mapping
    if self._remat:
      call_function = hk.remat(call_function)
    return call_function(inputs, output_template)


class _SplitInputMatMul(hk.Module):
  """First matmul of DataArrayDictDenseEncoder."""

  def __init__(
      self,
      *,
      name: str,
      preserved_dims: Sequence[str],
      dims_to_split: Sequence[str] | None,
      output_size: int,
      w_init: hk.initializers.Initializer | None = None,
      _emulate_dims_to_split_bug: bool = False,
      preserved_dims_sizes: Mapping[str, int] | None = None,
  ):
    """Inits the module.

    Args:
      name: Name of the module.
      preserved_dims: See DataArrayDictDenseEncoder.
      dims_to_split: See DataArrayDictDenseEncoder.
      output_size: Output size of the matmul.
      w_init: See DenseLayer.
      _emulate_dims_to_split_bug: To be used for testing only.
      preserved_dims_sizes: Allows specifying sizes of `preserved_dims`, to
        handle the case where they are not present in any of the input arrays.
    """
    super().__init__(name=name)
    self._output_size = output_size
    self._w_init = w_init
    self._preserved_dims = preserved_dims
    self._dims_to_split = dims_to_split
    self._emulate_dims_to_split_bug = _emulate_dims_to_split_bug
    self._preserved_dims_sizes = preserved_dims_sizes

  def __call__(self,
               data_array_mapping: Mapping[Hashable, xr.DataArray],
               ) -> chex.Array:
    """Runs the matmul."""
    size_dict, coord_dict = _get_preserved_dims_sizes_and_coords(
        data_array_mapping,
        self._preserved_dims,
        require_preserved_dims_in_all_arrays=False,
        preserved_dims_sizes=self._preserved_dims_sizes,
    )
    (num_channels, flat_arrays, dims_to_split_dict, dims_to_keep_dict,
     ) = _flatten_data_array_mapping(
         data_array_mapping, self._preserved_dims, self._dims_to_split,
         size_dict, coord_dict)

    # Parameter initialization.
    weight_sequence = self._initialize_and_get_params(
        num_channels, flat_arrays, dims_to_split_dict, dims_to_keep_dict)

    # Forward pass.
    # During forward pass, rather than splitting the dims of the inputs arrays
    # and applying their weights, instead we:
    # 1. Transpose the input arrays to be fully flattened to go from
    #  preserved_dims + dims_to_keep + [channels_for_dims_to_split]
    # to:
    #  preserved_dims + [all_channels_for_dims_to_split_and_dims_to_keep]
    inputs_to_concat = []
    for unused_name, flat_data_array in flat_arrays.items():
      if self._emulate_dims_to_split_bug:
        flat_data_array = flat_data_array.transpose(
            *self._preserved_dims, ..., "channels")
      flat_array = xarray_jax.unwrap_data(flat_data_array)
      flat_array = flat_array.reshape(
          flat_array.shape[:len(self._preserved_dims)] +  (-1,))
      inputs_to_concat.append(flat_array)

    # 2. Concatenate all the inputs arrays, and all the parameters.
    concat_inputs = jnp.concatenate(inputs_to_concat, axis=-1)
    concat_w = jnp.concatenate(weight_sequence, axis=0)

    # 3. Apply the matmul. in concatenated space.
    return jnp.dot(concat_inputs, concat_w)

  def _initialize_and_get_params(
      self,
      num_channels: int,
      flat_arrays: Mapping[str, xr.DataArray],
      dims_to_split_dict: Mapping[str, Sequence[str]],
      dims_to_keep_dict: Mapping[str, Sequence[str]],
      ) -> list[chex.Array]:
    """Initializes and returns the parameters."""

    # All code on this initialization section will not be part of the compiled
    # function when using jax.jit on ".apply", since once the model
    # has been initialized `hk.get_parameter` simply returns the already
    # initialized parameters, and all computation that goes into the arguments
    # of `hk.get_parameter` become dead branches.

    # 1. We initialize the parameter array as a single array for the total
    # number of channels for non-preserved dims.
    w_init = self._w_init
    if w_init is None:
      stddev = 1. / np.sqrt(num_channels)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    concat_w_init = w_init([num_channels, self._output_size], INITIALIZER_DTYPE)

    dtype = _get_shared_dtype(
        flat_arrays,  # pyrefly: ignore[bad-argument-type]
        # Because bool casts nicely to all other types, we can ignore it.
        ignore_bool=True,
        )

    # 2. Then we split the initialized array into the individual parameters
    # using as initialization a slice of the previous array.
    channel_index = 0
    weight_sequence = []
    for name, flat_data_array in flat_arrays.items():

      if (self._emulate_dims_to_split_bug and
          len(dims_to_split_dict[name]) >= 1 and
          len(dims_to_keep_dict[name]) > 1):
        raise ValueError(
            "Affected by the dims_to_split bug in a way that won't be fixable "
            "after the bug fix is applied by default."
            "If this is a new experiment, please wait until "
            "`_emulate_dims_to_split_bug` becomes False by default (or "
            "comment-out this check for short-lived experiments). If this "
            "is an existing experiment, please get in touch ASAP with alvarosg")

      # The goal is to build a param name of the form:
      # "w_{name}_keep_dim_1=1,2_keep_dim_2=1,2,3_split_dim_1=10_split_dim_2=20"
      # where split dims will always have a single coordinate value associated
      # with them, and keep dims will have as many coordinates as their original
      # size.

      # Do a little of prep work for the dims to keep, counting how many extra
      # channels will corresponds to the dims to keep, and building their part
      # of the param name ""_keep_dim_1=1,2_keep_dim_2=1,2,3".
      num_channels_in_dims_to_keep = 1
      param_name_for_dims_to_keep = ""
      for dim in dims_to_keep_dict[name]:
        num_channels_in_dims_to_keep *= flat_data_array.sizes[dim]
        param_name_for_dims_to_keep += "_" + coord_to_str(flat_data_array, dim)

      # Now we need a separate param for each channel (which corresponds to the
      # product of all of the dims that we decided to split).
      # Each iterational in the loop will correspond to a specific combination
      # of the coordinates of the split dims.
      for var_channel_i in range(flat_data_array.sizes["channels"]):
        channel_data = flat_data_array.isel(channels=var_channel_i)
        param_name = f"w_{name}" + param_name_for_dims_to_keep

        # Append the suffix for with the coordinate value for each split dim.
        if dims_to_split_dict[name]:
          for split_dim in dims_to_split_dict[name]:
            param_name += "_" + coord_to_str(channel_data, split_dim)
        else:
          # Should be guaranteed at this point, that variables without any
          # split have a single channel dimension.
          assert var_channel_i == 0

        # Initialize the param using the slice of the concatenated initialized
        # array, note the number of channels for this param is not just 1, but
        # it is the size of any dimensions we have decided to keep, instead of
        # split.
        init = hk.initializers.Constant(
            concat_w_init[
                channel_index:channel_index+num_channels_in_dims_to_keep])
        weight = hk.get_parameter(
            param_name, init.constant.shape, dtype, init=init)  # pytype: disable=attribute-error
        weight_sequence.append(weight)

        channel_index += num_channels_in_dims_to_keep
    # Should be guaranteed that we have used all the channels at this point.
    assert channel_index == num_channels

    return weight_sequence


class _SplitOutputLinear(hk.Module):
  """Last linear of matmul of DataArrayDictDenseDecoder."""

  def __init__(self,
               *,
               name: str,
               preserved_dims: Sequence[str],
               dims_to_split: Sequence[str] | None,
               w_init: hk.initializers.Initializer | None = None,
               b_init: hk.initializers.Initializer | None = None,
               ):
    """Inits the module.

    Args:
      name: Name of the module.
      preserved_dims: See DataArrayDictDenseEncoder.
      dims_to_split: See DataArrayDictDenseEncoder.
      w_init: See DenseLayer.
      b_init: See DenseLayer.
    """
    super().__init__(name=name)
    self._w_init = w_init
    self._b_init = b_init or jnp.zeros
    self._preserved_dims = preserved_dims
    self._dims_to_split = dims_to_split

  def __call__(
      self,
      inputs: chex.Array,
      output_template: DataArrayMapping) -> DataArrayMapping:
    """Runs the linear. See DataArrayDictDenseDecoder for more details."""

    # Sanity checks.
    if inputs.ndim != len(self._preserved_dims) + 1:
      raise ValueError(
          "Inputs must have a single feature dimension, but got shape"
          f"{inputs.shape} for preseved_dims={self._preserved_dims}"
      )
    num_input_channels = inputs.shape[-1]
    dtype = _get_shared_dtype(output_template)
    # We are disabling this to support the "keep_targets_in_fp32" option in the
    # bfloat16 wrapper, but ideally the targets_template would already come in
    # the dtype that we expect the outputs.
    # TODO(dominicmasters, alvarosg): Find a better way to handle the
    # target_templates dtype.
    # if dtype != inputs.dtype:
    #   raise ValueError(
    #       "Dtype of inputs and data_array_mapping_template must match. Got "
    #       f"{dtype} and {inputs.dtype}")

    # Preparation of the template.
    size_dict, coord_dict = _get_preserved_dims_sizes_and_coords(
        output_template, self._preserved_dims,
        require_preserved_dims_in_all_arrays=True)
    (num_output_channels, flat_arrays, dims_to_split_dict, dims_to_keep_dict,
     ) = _flatten_data_array_mapping(
         output_template, self._preserved_dims,
         self._dims_to_split, size_dict, coord_dict)

    # Parameter initialization.
    weight_sequence, bias_sequence = self._initialize_and_get_params(
        num_input_channels, num_output_channels, flat_arrays,
        dims_to_split_dict, dims_to_keep_dict, dtype)

    # Forward pass.
    # During forward pass, rather than splitting the dims of the inputs arrays
    # and applying their weights, instead we:
    # 1. Concatenate all the parameters and apply the layer.
    concat_w = jnp.concatenate(weight_sequence, axis=1)
    concat_b = jnp.concatenate(bias_sequence, axis=0)
    concat_outputs = jnp.dot(inputs, concat_w) + concat_b

    # 2. Slice and reshape the output channels into parts of the template.
    channel_index = 0
    outputs = {}
    for name, flat_data_array in flat_arrays.items():
      # Get the channels that will correspond to this array.
      # The total number of channels for this array, will be the product of the
      # sizes of everything but the preserved dims.
      this_num_channels = 1
      for dim in flat_data_array.dims:
        if dim in self._preserved_dims:
          continue
        this_num_channels *= flat_data_array.sizes[dim]
      data_slice = concat_outputs[
          ..., channel_index:channel_index+this_num_channels]

      # Reshape and set the data.
      flat_data_array = flat_data_array.copy()
      flat_data_array.data = xarray_jax.wrap(
          data_slice.reshape(flat_data_array.data.shape))

      # Unstack any split dimensions that are still stacked in the channels dim,
      # and transpose back to expected order.
      if dims_to_split_dict[name]:
        data_array = flat_data_array.unstack("channels")
      else:
        data_array = flat_data_array.squeeze("channels")
      data_array = data_array.transpose(*output_template[name].dims)

      # We replace all the coords and indices, since the stacking/unstacking
      # operations, may have broadcasted some of the coords, and added some
      # integer index coords that did not exist explicitly.
      data_array = data_array.drop_indexes(data_array.coords, errors="ignore")
      data_array = data_array.reset_coords(data_array.coords, drop=True)
      data_array = data_array.assign_coords(output_template[name].coords)

      outputs[name] = data_array
      channel_index += this_num_channels

    # Should be guaranteed that we have used all the channels at this point.
    assert channel_index == num_output_channels

    if isinstance(output_template, xr.Dataset):
      outputs = xr.Dataset(outputs)

    return outputs  # pyrefly: ignore[bad-return]

  def _initialize_and_get_params(
      self,
      num_input_channels: int,
      num_output_channels: int,
      flat_arrays: Mapping[str, xr.DataArray],
      dims_to_split_dict: Mapping[str, Sequence[str]],
      dims_to_keep_dict: Mapping[str, Sequence[str]],
      dtype: np.dtype,
      ) -> tuple[list[chex.Array], list[chex.Array]]:

    # Very similar to `_SplitInputMatMul._initialize_and_get_params`, except:
    # (1) Both weights and biases are initialized.
    # (2) The shape of the weights is [latent_size, channels_to_split],
    #     instead of [channels_to_split, latent_size].
    # TODO(alvarosg): Consider refactoring the initialize functions to avoid
    # code duplication.
    w_init = self._w_init
    if w_init is None:
      stddev = 1. / np.sqrt(num_input_channels)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    concat_w_init = w_init([num_input_channels, num_output_channels],
                           INITIALIZER_DTYPE)
    concat_b_init = self._b_init([num_output_channels],
                                 INITIALIZER_DTYPE)

    channel_index = 0
    weight_sequence = []
    bias_sequence = []
    for name, flat_data_array in flat_arrays.items():
      num_channels_in_dims_to_keep = 1
      param_name_for_dims_to_keep = ""
      for dim in dims_to_keep_dict[name]:
        num_channels_in_dims_to_keep *= flat_data_array.sizes[dim]
        param_name_for_dims_to_keep += "_" + coord_to_str(flat_data_array, dim)

      for var_channel_i in range(flat_data_array.sizes["channels"]):
        param_name = str(name) + param_name_for_dims_to_keep
        channel_data = flat_data_array.isel(channels=var_channel_i)

        if dims_to_split_dict[name]:
          for trailing_dim in dims_to_split_dict[name]:
            param_name += "_" + coord_to_str(channel_data, trailing_dim)
        else:
          assert var_channel_i == 0

        init_w = hk.initializers.Constant(
            concat_w_init[
                :, channel_index:channel_index+num_channels_in_dims_to_keep])
        init_b = hk.initializers.Constant(
            concat_b_init[
                channel_index:channel_index+num_channels_in_dims_to_keep])

        weight_sequence.append(hk.get_parameter(
            "w_" + param_name, init_w.constant.shape, dtype, init=init_w))  # pytype: disable=attribute-error
        bias_sequence.append(hk.get_parameter(
            "b_" + param_name, init_b.constant.shape, dtype, init=init_b))  # pytype: disable=attribute-error
        channel_index += num_channels_in_dims_to_keep

    assert channel_index == num_output_channels

    return weight_sequence, bias_sequence


def _get_preserved_dims_sizes_and_coords(
    data_array_mapping: Mapping[Hashable, xr.DataArray],
    preserved_dims: Sequence[str],
    require_preserved_dims_in_all_arrays: bool,
    preserved_dims_sizes: Mapping[str, int] | None = None,
) -> tuple[dict[str, int], dict[str, xr.DataArray | None]]:
  """Get sizes and coords for each of the preserved dims."""
  # Search for coordinates and sizes for the preserved dims, to verify they
  # are the same for all arrays, and to return them in a clean dict.
  size_dict = {}
  coord_dict = {}
  for preserved_dim in preserved_dims:
    for name, data_array in data_array_mapping.items():
      if preserved_dim not in data_array.dims:
        if require_preserved_dims_in_all_arrays:
          raise ValueError(
              f"Preserved dim {preserved_dim} not found in array {name}: "
              f"{data_array}.")
        continue

      # Get size and check consistency.
      dim_size = data_array.sizes[preserved_dim]
      if preserved_dim not in size_dict:
        size_dict[preserved_dim] = dim_size
      if size_dict[preserved_dim] != dim_size:
        raise ValueError(
            f"Sizes for preserved dim {preserved_dim} must be the same for all"
            f" arrays. Got {data_array_mapping}."
        )

      # Get coord and check consistency. Only keep coordinates that are
      # proper 1D indices for the dimension (i.e. dims == (preserved_dim,)).
      # Multi-dimensional coordinates (e.g. curvilinear lat/lon with
      # dims=("lat", "lon")) are not stored, since they are not valid as
      # dimension indices for expand_dims.
      if preserved_dim in data_array.coords:
        raw_coord = data_array.coords[preserved_dim]
        if raw_coord.dims == (preserved_dim,):
          dim_coord = raw_coord
        else:
          dim_coord = None
      else:
        dim_coord = None
      if preserved_dim not in coord_dict:
        coord_dict[preserved_dim] = dim_coord
      if (coord_dict[preserved_dim] is None and dim_coord is not None or
          coord_dict[preserved_dim] is not None and (
              dim_coord is None or
              not np.array_equal(
                  coord_dict[preserved_dim].data, dim_coord.data))):
        raise ValueError(
            f"Coordinates for preserved dim {preserved_dim} must be the same "
            f"for all arrays. Got {data_array_mapping}."
        )

    if preserved_dim not in size_dict:
      if preserved_dims_sizes and preserved_dim in preserved_dims_sizes:
        size_dict[preserved_dim] = preserved_dims_sizes[preserved_dim]
        coord_dict[preserved_dim] = None
      else:
        raise ValueError(
            f"Preserved dim {preserved_dim} not found in any array in "
            f"{data_array_mapping}."
        )
    else:
      if (preserved_dims_sizes and preserved_dim in preserved_dims_sizes and
          size_dict[preserved_dim] != preserved_dims_sizes[preserved_dim]):
        raise ValueError(
            f"Size for preserved dim {preserved_dim} must be the same as "
            f"specified in preserved_dims_sizes. Got "
            f"{size_dict[preserved_dim]} and "
            f"{preserved_dims_sizes[preserved_dim]}."
        )

  return size_dict, coord_dict


def _get_shared_dtype(
    data_array_mapping: Mapping[Hashable, xr.DataArray],
    ignore_bool: bool = False,
    ) -> np.dtype:
  """Returns the dtype of all arrays in the mapping."""

  dtype = None
  for name, data_array in data_array_mapping.items():
    if ignore_bool and data_array.dtype == bool:
      continue

    if dtype is None:
      dtype = data_array.dtype

    if dtype != data_array.dtype:
      raise ValueError(
          "Dtype of all arrays in data_array_mapping must match. Got "
          f"{dtype} and {data_array.dtype} for {name}")
  assert dtype is not None  # Make pytype happy.
  return dtype


def _flatten_data_array_mapping(
    data_array_mapping: Mapping[Hashable, xr.DataArray],
    preserved_dims: Sequence[str],
    dims_to_split: Sequence[str] | None,
    size_dict: Mapping[str, int],
    coord_dict: Mapping[str, xr.DataArray | None],
) -> tuple[int,
           Mapping[str, xr.DataArray],
           Mapping[str, Sequence[str]],
           Mapping[str, Sequence[str]],
           ]:
  """Flattens all arrays in the data_array_mapping while doing book-keeping.

  Args:
    data_array_mapping: The mapping to flatten.
    preserved_dims: See DataArrayDictDenseEncoder.
    dims_to_split: See DataArrayDictDenseEncoder. Can be regex patterns.
    size_dict: As returned by _get_preserved_dims_sizes_and_coords.
    coord_dict: As returned by _get_preserved_dims_sizes_and_coords.

  Returns:
    A tuple of:
      - The total number of channels that would result from concatenating all
        the features along non preserved dims for all input arrays.
      - The flattened arrays, each with shape:
        preserved_dims_shape + keep_dims_shape + [stacked_keep_dims_size]
        such that all of the dims to split have been stacked into a single
        dimension.
      - A mapping from array name to the dims dims_split for that array.
      - A mapping from array name to the dims dims_keep for that array.WW

  """

  flat_arrays = {}
  num_channels = 0
  dims_to_split_dict = {}
  dims_to_keep_dict = {}
  # Force the iteration ordering to be deterministic. Otherwise we can get
  # inconsistent parameter initialization across hosts if data_array_mapping is
  # constructed with a non-deterministic ordering, for example by iterating
  # through a set.
  for name in sorted(data_array_mapping.keys()):  # pyrefly: ignore[bad-specialization]
    data_array = data_array_mapping[name]
    name = str(name)  # Make pytype happy.

    # Broadcast any preserved dims that may not be present in the array.
    for axis, preserved_dim in enumerate(preserved_dims):
      if preserved_dim not in data_array.dims:
        if coord_dict[preserved_dim] is not None:
          data_array = data_array.expand_dims(
              axis=axis, **{preserved_dim: coord_dict[preserved_dim]})  # pyrefly: ignore[bad-argument-type]
        else:
          data_array = data_array.expand_dims(
              axis=axis, **{preserved_dim: size_dict[preserved_dim]})  # pyrefly: ignore[bad-argument-type]

    # Determine the dims to split for this array specifically.
    if dims_to_split is None:
      # All remaining non-preserved dims are split.
      dims_to_split_this_array = [
          dim for dim in data_array.dims if dim not in preserved_dims]
    else:
      # Find dimensions in `data_array.dims` that match any of the
      # regex patterns in `dims_to_split`.
      dims_to_split_this_array = []
      for split_pattern in dims_to_split:
        for darray_dim in data_array.dims:
          if (
              darray_dim not in dims_to_split_this_array
              and re.fullmatch(split_pattern, darray_dim)  # pyrefly: ignore[no-matching-overload]
          ):
            dims_to_split_this_array.append(darray_dim)
    dims_to_split_dict[name] = dims_to_split_this_array

    # Transpose the array to put the preserved dims first, and the dims to split
    # at the end. We put them at the end because the `stack` operation later
    # will stack them at the end no matter what, so it is easier to reason about
    # if they are already at the end.
    # TODO(alvarosg): Find a way for `stack` to add the new dimension not at
    # the end, but in the middle, so we don't need a double transpose to then
    # put the `channels` after the preserved dims, and instead directly do:
    # `.transpose(*preserved_dims, *dims_to_split_this_array, ...)` here.
    data_array = data_array.transpose(
        *preserved_dims, ..., *dims_to_split_this_array)

    # Stack it into shape: preserved_dims + dims_to_keep + ["channels"].
    # where channels are the product of the dims to split.
    if dims_to_split_this_array:
      for dim in dims_to_split_this_array:
        if dim not in data_array.coords:
          # TODO(alvarosg): Consider raising an error instead.
          logging.warning(
              "No coordinates found for dim %s of array %s., this will use "
              "indices as coordinates and may make model surgery more error "
              "prone if changing coordinates at fine-tuning time.",
              dim, data_array)

      flat_array = data_array.stack(channels=list(dims_to_split_this_array))
    else:
      # If there are not dims to split we will need to add a channel at the end.
      flat_array = data_array.expand_dims("channels", axis=-1)

    # Now we finally transpose the array to put the preserved dims first,
    # then the dims to split, and then any other remaining dims.
    # This is is because we will be concatenating arrays with the non split dims
    # to produce a single flattened array, so we need the channels axis to
    # come before the non split dims.
    flat_array = flat_array.transpose(*preserved_dims, "channels", ...)
    flat_arrays[name] = flat_array

    # Get a list of the dims to keep (any remaining ones) for easy book-keeping.
    dims_to_keep_this_array = [
        dim for dim in flat_array.dims
        if dim not in list(preserved_dims) + ["channels"]]
    dims_to_keep_dict[name] = dims_to_keep_this_array

    # The total number of channels for this array, will be the product of the
    # sizes of the "channels" dim, times any of the other dims to keep (e.g.
    # everything but the preserved dims).
    this_num_channels = 1
    for dim in flat_array.dims:
      if dim in preserved_dims:
        continue
      this_num_channels *= flat_array.sizes[dim]
    num_channels += this_num_channels

  return num_channels, flat_arrays, dims_to_split_dict, dims_to_keep_dict  # pytype: disable=bad-return-type
