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
"""Utils for rolling out models."""

from typing import Iterator, Optional, Sequence

from absl import logging
import chex
import dask.array
from graphcast import xarray_jax
from graphcast import xarray_tree
import jax
import numpy as np
import typing_extensions
import xarray


class PredictorFn(typing_extensions.Protocol):
  """Functional version of base.Predictor.__call__ with explicit rng."""

  def __call__(
      self, rng: chex.PRNGKey, inputs: xarray.Dataset,
      targets_template: xarray.Dataset,
      forcings: xarray.Dataset,
      **optional_kwargs,
      ) -> xarray.Dataset:
    ...


def _replicate_dataset(
    data: xarray.Dataset, replica_dim: str,
    replicate_to_device: bool,
    devices: Sequence[jax.Device],
    ) -> xarray.Dataset:
  """Used to prepare for xarray_jax.pmap."""

  def replicate_variable(variable: xarray.Variable) -> xarray.Variable:
    if replica_dim in variable.dims:
      # TODO(pricei): call device_put_replicated when replicate_to_device==True
      return variable.transpose(replica_dim, ...)
    else:
      data = len(devices) * [variable.data]
      if replicate_to_device:
        assert devices is not None
        # TODO(pricei): Refactor code to use "device_put_replicated" instead of
        # device_put_sharded.
        data = jax.device_put_sharded(data, devices)
      else:
        data = np.stack(data, axis=0)
      return xarray_jax.Variable(
          data=data, dims=(replica_dim,) + variable.dims, attrs=variable.attrs
      )

  def replicate_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    if dataset is None:
      return None
    data_variables = {
        name: replicate_variable(var)
        for name, var in dataset.data_vars.variables.items()
    }
    coords = {name: coord.variable for name, coord in dataset.coords.items()}
    return xarray.Dataset(data_variables, coords=coords, attrs=dataset.attrs)

  return replicate_dataset(data)


def chunked_prediction_generator_multiple_runs(
    predictor_fn: PredictorFn,
    rngs: chex.PRNGKey,
    inputs: xarray.Dataset,
    targets_template: xarray.Dataset,
    forcings: Optional[xarray.Dataset],
    num_samples: Optional[int],
    pmap_devices: Optional[Sequence[jax.Device]] = None,
    **chunked_prediction_kwargs,
) -> Iterator[xarray.Dataset]:
  """Outputs a trajectory of multiple samples by yielding chunked predictions.

  Args:
    predictor_fn: Function to use to make predictions for each chunk.
    rngs: RNG sequence to be used for each ensemble member.
    inputs: Inputs for the model.
    targets_template: Template for the target prediction, requires targets
        equispaced in time.
    forcings: Optional forcing for the model.
    num_samples: The number of runs / samples to rollout.
    pmap_devices: List of devices over which predictor_fn is pmapped, or None if
      it is not pmapped.
    **chunked_prediction_kwargs:
      See chunked_prediction, some of these are required arguments.

  Yields:
    The predictions for each chunked step of the chunked rollout, such that
    if all predictions are concatenated in time and sample dimension squeezed,
    this would match the targets template in structure.

  """
  if pmap_devices is not None:
    assert (
        num_samples % len(pmap_devices) == 0
    ), "num_samples must be a multiple of len(pmap_devices)"

    def predictor_fn_pmap_named_args(rng, inputs, targets_template, forcings):
      targets_template = _replicate_dataset(
          targets_template,
          replica_dim="sample",
          replicate_to_device=True,
          devices=pmap_devices,
      )
      return predictor_fn(rng, inputs, targets_template, forcings)

    for i in range(0, num_samples, len(pmap_devices)):
      sample_idx = slice(i, i + len(pmap_devices))
      logging.info("Samples %s out of %s", sample_idx, num_samples)
      logging.flush()
      sample_group_rngs = rngs[sample_idx]

      if "sample" not in inputs.dims:
        sample_inputs = inputs
      else:
        sample_inputs = inputs.isel(sample=sample_idx, drop=True)

      sample_inputs = _replicate_dataset(
          sample_inputs,
          replica_dim="sample",
          replicate_to_device=True,
          devices=pmap_devices,
      )

      if forcings is not None:
        if "sample" not in forcings.dims:
          sample_forcings = forcings
        else:
          sample_forcings = forcings.isel(sample=sample_idx, drop=True)

        # TODO(pricei): We are replicating the full forcings for all rollout
        # timesteps here, rather than inside `predictor_fn_pmap_named_args` like
        # the targets_template above, because the forcings are concatenated with
        # the inputs which will already be replicated. We should refactor this
        # so that chunked prediction is aware of whether it is being run with
        # pmap, and if so do the replication and device_put only of the
        # necessary timesteps, as part of the chunked prediction function.
        sample_forcings = _replicate_dataset(
            sample_forcings,
            replica_dim="sample",
            replicate_to_device=False,
            devices=pmap_devices,
        )
      else:
        sample_forcings = None

      for prediction_chunk in chunked_prediction_generator(
          predictor_fn=predictor_fn_pmap_named_args,
          rng=sample_group_rngs,
          inputs=sample_inputs,
          targets_template=targets_template,
          forcings=sample_forcings,
          pmap_devices=pmap_devices,
          **chunked_prediction_kwargs,
      ):
        prediction_chunk.coords["sample"] = np.arange(
            sample_idx.start, sample_idx.stop, sample_idx.step
        )
        yield prediction_chunk
        del prediction_chunk
  else:
    for i in range(num_samples):
      logging.info("Sample %d/%d", i, num_samples)
      logging.flush()
      this_sample_rng = rngs[i]

      if "sample" in inputs.dims:
        sample_inputs = inputs.isel(sample=i, drop=True)
      else:
        sample_inputs = inputs

      sample_forcings = forcings
      if sample_forcings is not None:
        if "sample" in sample_forcings.dims:
          sample_forcings = sample_forcings.isel(sample=i, drop=True)

      for prediction_chunk in chunked_prediction_generator(
          predictor_fn=predictor_fn,
          rng=this_sample_rng,
          inputs=sample_inputs,
          targets_template=targets_template,
          forcings=sample_forcings,
          **chunked_prediction_kwargs):
        prediction_chunk.coords["sample"] = i
        yield prediction_chunk
        del prediction_chunk


def chunked_prediction(
    predictor_fn: PredictorFn,
    rng: chex.PRNGKey,
    inputs: xarray.Dataset,
    targets_template: xarray.Dataset,
    forcings: xarray.Dataset,
    num_steps_per_chunk: int = 1,
    verbose: bool = False,
) -> xarray.Dataset:
  """Outputs a long trajectory by iteratively concatenating chunked predictions.

  Args:
    predictor_fn: Function to use to make predictions for each chunk.
    rng: Random key.
    inputs: Inputs for the model.
    targets_template: Template for the target prediction, requires targets
        equispaced in time.
    forcings: Optional forcing for the model.
    num_steps_per_chunk: How many of the steps in `targets_template` to predict
        at each call of `predictor_fn`. It must evenly divide the number of
        steps in `targets_template`.
    verbose: Whether to log the current chunk being predicted.

  Returns:
    Predictions for the targets template.

  """
  chunks_list = []
  for prediction_chunk in chunked_prediction_generator(
      predictor_fn=predictor_fn,
      rng=rng,
      inputs=inputs,
      targets_template=targets_template,
      forcings=forcings,
      num_steps_per_chunk=num_steps_per_chunk,
      verbose=verbose):
    chunks_list.append(jax.device_get(prediction_chunk))
  return xarray.concat(chunks_list, dim="time")


def chunked_prediction_generator(
    predictor_fn: PredictorFn,
    rng: chex.PRNGKey,
    inputs: xarray.Dataset,
    targets_template: xarray.Dataset,
    forcings: xarray.Dataset,
    num_steps_per_chunk: int = 1,
    verbose: bool = False,
    pmap_devices: Optional[Sequence[jax.Device]] = None
) -> Iterator[xarray.Dataset]:
  """Outputs a long trajectory by yielding chunked predictions.

  Args:
    predictor_fn: Function to use to make predictions for each chunk.
    rng: Random key.
    inputs: Inputs for the model.
    targets_template: Template for the target prediction, requires targets
        equispaced in time.
    forcings: Optional forcing for the model.
    num_steps_per_chunk: How many of the steps in `targets_template` to predict
        at each call of `predictor_fn`. It must evenly divide the number of
        steps in `targets_template`.
    verbose: Whether to log the current chunk being predicted.
    pmap_devices: List of devices over which predictor_fn is pmapped, or None if
      it is not pmapped.

  Yields:
    The predictions for each chunked step of the chunked rollout, such as
    if all predictions are concatenated in time this would match the targets
    template in structure.

  """

  # Create copies to avoid mutating inputs.
  inputs = xarray.Dataset(inputs)
  targets_template = xarray.Dataset(targets_template)
  forcings = xarray.Dataset(forcings)

  if "datetime" in inputs.coords:
    del inputs.coords["datetime"]

  if "datetime" in targets_template.coords:
    output_datetime = targets_template.coords["datetime"]
    del targets_template.coords["datetime"]
  else:
    output_datetime = None

  if "datetime" in forcings.coords:
    del forcings.coords["datetime"]

  num_target_steps = targets_template.dims["time"]
  num_chunks, remainder = divmod(num_target_steps, num_steps_per_chunk)
  if remainder != 0:
    raise ValueError(
        f"The number of steps per chunk {num_steps_per_chunk} must "
        f"evenly divide the number of target steps {num_target_steps} ")

  if len(np.unique(np.diff(targets_template.coords["time"].data))) > 1:
    raise ValueError("The targets time coordinates must be evenly spaced")

  # Our template targets will always have a time axis corresponding for the
  # timedeltas for the first chunk.
  targets_chunk_time = targets_template.time.isel(
      time=slice(0, num_steps_per_chunk))

  current_inputs = inputs

  def split_rng_fn(rng):
    # Note, this is *not* equivalent to `return jax.random.split(rng)`, because
    # by assigning to a tuple, the single numpy array returned by
    # `jax.random.split` actually gets split into two arrays, so when calling
    # the function with pmap the output is Tuple[Array, Array], where the
    # leading axis of each array is `num devices`.
    rng1, rng2 = jax.random.split(rng)
    return rng1, rng2

  if pmap_devices is not None:
    split_rng_fn = jax.pmap(split_rng_fn, devices=pmap_devices)

  for chunk_index in range(num_chunks):
    if verbose:
      logging.info("Chunk %d/%d", chunk_index, num_chunks)
      logging.flush()

    # Select targets for the time period that we are predicting for this chunk.
    target_offset = num_steps_per_chunk * chunk_index
    target_slice = slice(target_offset, target_offset + num_steps_per_chunk)
    current_targets_template = targets_template.isel(time=target_slice)

    # Replace the timedelta, by the one corresponding to the first chunk, so we
    # don't recompile at every iteration, keeping the
    actual_target_time = current_targets_template.coords["time"]
    current_targets_template = current_targets_template.assign_coords(
        time=targets_chunk_time).compute()

    current_forcings = forcings.isel(time=target_slice)
    current_forcings = current_forcings.assign_coords(time=targets_chunk_time)
    current_forcings = current_forcings.compute()
    # Make predictions for the chunk.
    rng, this_rng = split_rng_fn(rng)
    predictions = predictor_fn(
        rng=this_rng,
        inputs=current_inputs,
        targets_template=current_targets_template,
        forcings=current_forcings)

    # In the pmapped case, profiling reveals that the predictions, forcings and
    # inputs are all copied onto a single TPU, causing OOM. To avoid this
    # we pull all of the input/output data off the devices. This will have
    # some performance impact, but maximise the memory efficiency.
    # TODO(aelkadi): Pmap `_get_next_inputs` when running under pmap, and
    # remove the device_get.
    if pmap_devices is not None:
      predictions = jax.device_get(predictions)
      current_forcings = jax.device_get(current_forcings)
      current_inputs = jax.device_get(current_inputs)

    next_frame = xarray.merge([predictions, current_forcings])

    next_inputs = _get_next_inputs(current_inputs, next_frame)

    # Shift timedelta coordinates, so we don't recompile at every iteration.
    next_inputs = next_inputs.assign_coords(time=current_inputs.coords["time"])
    current_inputs = next_inputs

    # At this point we can assign the actual targets time coordinates.
    predictions = predictions.assign_coords(time=actual_target_time)
    if output_datetime is not None:
      predictions.coords["datetime"] = output_datetime.isel(
          time=target_slice)
    yield predictions
    del predictions


def _get_next_inputs(
    prev_inputs: xarray.Dataset, next_frame: xarray.Dataset,
    ) -> xarray.Dataset:
  """Computes next inputs, from previous inputs and predictions."""

  # Make sure are are predicting all inputs with a time axis.
  non_predicted_or_forced_inputs = list(
      set(prev_inputs.keys()) - set(next_frame.keys()))
  if "time" in prev_inputs[non_predicted_or_forced_inputs].dims:
    raise ValueError(
        "Found an input with a time index that is not predicted or forced.")

  # Keys we need to copy from predictions to inputs.
  next_inputs_keys = list(
      set(next_frame.keys()).intersection(set(prev_inputs.keys())))
  next_inputs = next_frame[next_inputs_keys]

  # Apply concatenate next frame with inputs, crop what we don't need.
  num_inputs = prev_inputs.dims["time"]
  return (
      xarray.concat(
          [prev_inputs, next_inputs], dim="time", data_vars="different")
      .tail(time=num_inputs))


def extend_targets_template(
    targets_template: xarray.Dataset,
    required_num_steps: int) -> xarray.Dataset:
  """Extends `targets_template` to `required_num_steps` with lazy arrays.

  It uses lazy dask arrays of zeros, so it does not require instantiating the
  array in memory.

  Args:
    targets_template: Input template to extend.
    required_num_steps: Number of steps required in the returned template.

  Returns:
    `xarray.Dataset` identical in variables and timestep to `targets_template`
    full of `dask.array.zeros` such that the time axis has `required_num_steps`.

  """

  # Extend the "time" and "datetime" coordinates
  time = targets_template.coords["time"]

  # Assert the first target time corresponds to the timestep.
  timestep = time[0].data
  if time.shape[0] > 1:
    assert np.all(timestep == time[1:] - time[:-1])

  extended_time = (np.arange(required_num_steps) + 1) * timestep

  if "datetime" in targets_template.coords:
    datetime = targets_template.coords["datetime"]
    extended_datetime = (datetime[0].data - timestep) + extended_time
  else:
    extended_datetime = None

  # Replace the values with empty dask arrays extending the time coordinates.
  datetime = targets_template.coords["time"]

  def extend_time(data_array: xarray.DataArray) -> xarray.DataArray:
    dims = data_array.dims
    shape = list(data_array.shape)
    shape[dims.index("time")] = required_num_steps
    dask_data = dask.array.zeros(
        shape=tuple(shape),
        chunks=-1,  # Will give chunk info directly to `ChunksToZarr``.
        dtype=data_array.dtype)

    coords = dict(data_array.coords)
    coords["time"] = extended_time

    if extended_datetime is not None:
      coords["datetime"] = ("time", extended_datetime)

    return xarray.DataArray(
        dims=dims,
        data=dask_data,
        coords=coords)

  return xarray_tree.map_structure(extend_time, targets_template)
