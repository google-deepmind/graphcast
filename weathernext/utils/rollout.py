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
"""Utils for rolling out models."""

import functools
from typing import Iterator, Optional, Sequence, Callable, Protocol, TypeVar

from absl import logging
import chex
import dask.array
import jax
import jax.numpy as jnp
import numpy as np
import xarray
import xarray_jax


# POP_TRICK_EXPLANATION
# When the caller site or generator yield does not longer need a reference
# we use a pop trick of the form, to avoid situations like
# `yield var` or `fn(var)`
# followed by `del var``
# Where the caller/yield site would still have a reference to the array.
# Instead we do:
# `var = [var]`
# followed by `yield var.pop()` or `fn(var.pop())
# Which straight away gets rid of the reference at the caller/yield site.


# TODO(alvarosg): Consider moving to a separate module.
def device_put_sharded(
    xs: Sequence[jax.Array],
    devices: Sequence[jax.Device],
    axis_name: str,
) -> jax.Array:
  """Stack data and put on devices with consistent sharding.

  Creates a mesh with axis_name to ensure JIT cache consistency with pmap.

  Args:
    xs: List of data to stack and put on devices. Note any array with more than
      one dimension also counts as a sequence of arrays.
    devices: List of devices to put the data on.
    axis_name: Name of the axis to use for sharding.

  Returns:
    Data put on devices with consistent sharding.
  """

  mesh = jax.sharding.Mesh(np.array(devices), (axis_name,))
  spec = jax.sharding.PartitionSpec(axis_name)
  sharding = jax.NamedSharding(mesh, spec)
  if isinstance(xs, (jax.Array, np.ndarray)):
    stacked = xs  # If it is an array it must be already stacked.
  else:
    is_jax = all(isinstance(x, jax.Array) for x in xs)
    stack_fn = jnp.stack if is_jax else np.stack
    stacked = stack_fn(xs, axis=0)

  if stacked.ndim == 0 or stacked.shape[0] != len(devices):
    raise ValueError(
        f"Invalid shape: {stacked.shape} expected leading {len(devices)}.")

  return jax.device_put(stacked, sharding)


class PredictorFn(Protocol):
  """Functional version of base.Predictor.__call__ with explicit rng."""

  def __call__(
      self, rng: chex.PRNGKey, inputs: xarray.Dataset,
      targets_template: xarray.Dataset,
      forcings: Optional[xarray.Dataset],
      **optional_kwargs,
      ) -> xarray.Dataset:
    ...


# TODO(alvarosg): Consider moving to a separate module.
def replicate_dataset(
    data: xarray.Dataset | None, replica_dim: str,
    replicate_to_device: bool,
    devices: Sequence[jax.Device] | None = None,
    num_replicas: int | None = None
    ) -> xarray.Dataset:
  """Used to prepare for xarray_jax.pmap."""

  if devices is not None and num_replicas is not None:
    if len(devices) != num_replicas:
      raise ValueError(f"devices: {len(devices)} != replicas: {num_replicas}")
  elif devices is not None:
    num_replicas = len(devices)
  else:
    if num_replicas is None:
      raise ValueError("num_replicas must be specified.")
    if replicate_to_device:
      raise ValueError(
          "devices must be specified when replicate_to_device is True.")

  def replicate_variable(variable: xarray.Variable) -> xarray.Variable:
    if replica_dim in variable.dims:
      if variable.sizes[replica_dim] != num_replicas:
        raise ValueError(
            f"Variable {variable} has {variable.sizes[replica_dim]} "
            f"replicas, but {num_replicas} were requested."
            )
      variable = variable.transpose(replica_dim, ...)
      if replicate_to_device:
        # Note device_put_sharded should be no-op if the array is already
        # sharded the same way.
        data = device_put_sharded(
            variable.data, devices, replica_dim)  # pyrefly: ignore[bad-argument-type]
        variable = xarray_jax.Variable(
            data=data, dims=variable.dims, attrs=variable.attrs
            )
      return variable
    else:
      data = num_replicas * [variable.data]
      if replicate_to_device:
        assert devices is not None
        data = device_put_sharded(data, devices, replica_dim)
      else:
        data = np.stack(data, axis=0)
      return xarray_jax.Variable(
          data=data, dims=(replica_dim,) + variable.dims, attrs=variable.attrs
          )

  def replicate_dataset_(dataset: xarray.Dataset) -> xarray.Dataset:
    if dataset is None: return None  # pyrefly: ignore[bad-return]
    data_variables = {
        name: replicate_variable(var)
        for name, var in dataset.data_vars.variables.items()}
    coords = {}
    for name, coord in dataset.coords.items():
      if coord.attrs.get(xarray_jax.core.JAX_COORD_ATTR_NAME, False):
        # JAX coordinates will be exposed as leaf array data when flattened
        # for the pmap so will need to have the replica axis added.
        coords[name] = replicate_variable(coord.variable)
      else:
        # Other coordinates will be static data and won't need replicating here.
        coords[name] = coord.variable
    return xarray.Dataset(data_variables, coords=coords, attrs=dataset.attrs)

  return replicate_dataset_(data)  # pyrefly: ignore[bad-argument-type]


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
  """Multiple rollouts by yielding time and sample chunks as they are ready.

  This is useful for stochastic models where you want an ensemble of multiple
  runs to evaluate using probabilistic / ensemble metrics.

  It works similarly to ensemble.MultipleRunsLooped but is necessary because
  ensemble.MultipleRunsLooped will not work with chunked rollouts.

  Args:
    predictor_fn: For chunked_prediction.
    rngs: RNG sequence to be used for each ensemble member.
    inputs: Input dataset for `chunked_prediction`. If a "sample" axis is
      present, each rollout will be done with a different sample
    targets_template: Targets template for `chunked_prediction'.
    forcings: Forcings for `chunked_prediction'.
    num_samples: The number of runs / samples to rollout. If provided, must be
      consistent with the size of the "sample" axis of the inputs. It is not
      optional if the inputs don't have a sample axis.
    pmap_devices: List of devices over which predictor_fn is pmapped, or None if
      it is not pmapped.
    **chunked_prediction_kwargs:
      See chunked_prediction, some of these are required arguments.

  Yields:
    Yields chunks of predictions each with `num_samples_per_chunk` samples
    and `num_steps_per_chunk` leadtimes. Where `num_samples_per_chunk` matches
    the number of devices in `pmap_devices` if provided, or 1 otherwise.
    Chunks for all leadtimes for a group of samples,  will be returned first
    before going to the next group of samples.
  """
  if num_samples is None:
    if "sample" not in inputs.dims:
      raise ValueError(
          "The number of samples must be passed when `inputs` don't have a "
          "`sample` dim.")
    num_samples = inputs.sizes["sample"]

  if "sample" in inputs.dims and num_samples != inputs.sizes["sample"]:
    raise ValueError(
        "Inconsistent number of samples requested for inputs"
        f"{num_samples} != {inputs.sizes['sample']}.")

  if num_samples != rngs.shape[0]:
    raise ValueError(
        f"Inconsistent number of rngs passed. {num_samples} != {len(rngs)}.")

  if forcings:
    if "sample" in forcings.dims and num_samples != forcings.sizes["sample"]:
      raise ValueError(
          "Inconsistent number of samples requested for forcings"
          f"{num_samples} != {forcings.sizes['sample']}.")

  if pmap_devices is not None:
    num_samples_per_chunk = len(pmap_devices)
  else:
    num_samples_per_chunk = 1

  if num_samples % num_samples_per_chunk != 0:
    raise ValueError(
        f"{num_samples} must multiple of {num_samples_per_chunk}")

  if pmap_devices is not None:
    replicate_fn = functools.partial(
        replicate_dataset,
        replica_dim="sample",
        devices=None,
        num_replicas=num_samples_per_chunk,
        replicate_to_device=False,  # There will be a separate device_put.
        )

    # Technically we would not need the `replicate` part, just the device_put
    # part however by the time this is called it would already be replicated
    # anyway, so `replicate_xarray` all will do is device_put.
    device_put_fn = functools.partial(
        replicate_dataset,
        replica_dim="sample",
        devices=pmap_devices,
        replicate_to_device=True,
        )

    # `xarray_jax.pmap/vmap` are not happy about passing args by name.
    def predictor_fn_wrapped(rng, inputs, targets_template, forcings):
      return predictor_fn(
          rng, inputs, targets_template, forcings)

    for i in range(0, num_samples, num_samples_per_chunk):
      sample_idx = slice(i, i + num_samples_per_chunk)
      logging.info("Samples (%s, %s) out of %s",
                   sample_idx.start,
                   sample_idx.stop,
                   num_samples)
      logging.flush()
      sample_group_rngs = device_put_sharded(
          rngs[sample_idx], pmap_devices, "sample")  # pyrefly: ignore[bad-argument-type]
      sample_inputs, sample_forcings = _slice_sample_if_present(  # pyrefly: ignore[bad-specialization]
          inputs, forcings, sample_idx)

      sample_inputs = [sample_inputs]  # See POP_TRICK_EXPLANATION
      sample_forcings = [sample_forcings]
      for prediction_chunk in chunked_prediction_generator(
          predictor_fn_wrapped,  # pyrefly: ignore[bad-argument-type]
          sample_group_rngs,
          inputs=sample_inputs.pop(),
          targets_template=targets_template,
          forcings=sample_forcings.pop(),
          pmap_devices=pmap_devices,  # May be None.
          replica_axis="sample",
          replicate_fn=replicate_fn,
          device_put_fn=device_put_fn,
          **chunked_prediction_kwargs,
          ):
        prediction_chunk.coords["sample"] = np.arange(
            sample_idx.start, sample_idx.stop, sample_idx.step
        )
        prediction_chunk = [prediction_chunk]  # See POP_TRICK_EXPLANATION.
        yield prediction_chunk.pop()

  else:
    assert num_samples_per_chunk == 1  # Should be guaranteed by above.
    for i in range(num_samples):
      logging.info("Sample %d/%d", i, num_samples)
      logging.flush()
      this_sample_rng = rngs[i]
      sample_inputs, sample_forcings = _slice_sample_if_present(  # pyrefly: ignore[bad-specialization]
          inputs, forcings, sample_idx=i)
      sample_inputs = [sample_inputs]  # See POP_TRICK_EXPLANATION
      sample_forcings = [sample_forcings]
      for prediction_chunk in chunked_prediction_generator(
          predictor_fn,
          this_sample_rng,
          inputs=sample_inputs.pop(),
          targets_template=targets_template,
          forcings=sample_forcings.pop(),
          **chunked_prediction_kwargs):
        prediction_chunk.coords["sample"] = i
        prediction_chunk = [prediction_chunk]  # See POP_TRICK_EXPLANATION
        yield prediction_chunk.pop()
      logging.info("Completed sample %d/%d", i, num_samples)
      logging.flush()


DatasetOrNone = TypeVar("DatasetOrNone", xarray.Dataset, None)


def _slice_sample_if_present(
    inputs: xarray.Dataset,
    forcings: DatasetOrNone,
    sample_idx: slice | int
) -> tuple[xarray.Dataset, DatasetOrNone]:
  """Slices inputs and forcings if 'sample' dimension is present."""
  if "sample" in inputs.dims:
    inputs = inputs.isel(sample=sample_idx, drop=True)
  if forcings is not None:
    if "sample" in forcings.dims:  # pyrefly: ignore[missing-attribute]
      forcings = forcings.isel(sample=sample_idx, drop=True)  # pyrefly: ignore[missing-attribute]
  return inputs, forcings


def chunked_prediction(
    predictor_fn: PredictorFn,
    rng: chex.PRNGKey,
    inputs: xarray.Dataset,
    targets_template: xarray.Dataset,
    forcings: Optional[xarray.Dataset] = None,
    num_steps_per_chunk: int = 1,
    **kwargs,
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
    **kwargs: To be passed to the generator.

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
      **kwargs,
  ):
    chunks_list.append(jax.device_get(prediction_chunk))
    del prediction_chunk
  return xarray.concat(chunks_list, dim="time")  # pyrefly: ignore[bad-return]


def chunked_prediction_generator(
    predictor_fn: PredictorFn,
    rng: chex.PRNGKey,
    inputs: xarray.Dataset,
    targets_template: xarray.Dataset,
    num_steps_per_chunk: int,
    forcings: Optional[xarray.Dataset] = None,
    verbose: bool = False,
    pmap_devices: Sequence[jax.Device] | None = None,
    replica_axis: str | None = None,
    device_put_fn: Callable[[xarray.Dataset], xarray.Dataset] | None = None,
    replicate_fn: Callable[[xarray.Dataset], xarray.Dataset] | None = None,
) -> Iterator[xarray.Dataset]:
  """Outputs a long trajectory by yielding chunked predictions.

  The treatment of the "time" coordinate, and any coordinates associated
  with the time dim, follows the rules described in the docstring of
  `autoregressive.Predictor`.

  Args:
    predictor_fn: Function to use to make predictions for each chunk.
    rng: Random key.
    inputs: Inputs for the model.
    targets_template: Template for the target prediction, requires targets
        equispaced in time.
    num_steps_per_chunk: How many of the steps in `targets_template` to predict
        at each call of `predictor_fn`. It must evenly divide the number of
        steps in `targets_template`.
    forcings: Optional forcing for the model.
    verbose: Whether to log the current chunk being predicted.
    pmap_devices: List of devices over which predictor_fn is pmapped, or None if
      it is not pmapped.
    replica_axis: Dimension name to use for the replicas.
    device_put_fn: Device put fn.
    replicate_fn: Function to replicate data, required if replica_axis is
      passed.

  Yields:
    The predictions for each chunked step of the chunked rollout, such as
    if all predictions are concatenated in time this would match the targets
    template in structure.

  """
  if pmap_devices is not None and replica_axis is None:
    raise ValueError("Must provide replica_axis when pmap_devices is provided.")

  if (replicate_fn is None) ^ (replica_axis is None):
    raise ValueError("Must provide replicate_fn when replica_axis is provided.")

  if forcings is None:
    forcings = xarray.Dataset({}, coords={
        n: c for n, c in targets_template.coords if "time" in c.dims})  # pytype: disable=attribute-error

  # Create copies to avoid mutating inputs.
  inputs = inputs.copy()
  targets_template = targets_template.copy()
  forcings = forcings.copy()

  if "datetime" in inputs.coords:
    del inputs.coords["datetime"]

  if "datetime" in targets_template.coords:
    output_datetime = targets_template.coords["datetime"]
    del targets_template.coords["datetime"]
  else:
    output_datetime = None

  if "datetime" in forcings.coords:
    del forcings.coords["datetime"]

  num_target_steps = targets_template.sizes["time"]
  num_chunks, remainder = divmod(num_target_steps, num_steps_per_chunk)
  if remainder != 0:
    raise ValueError(
        f"The number of steps per chunk {num_steps_per_chunk} must "
        f"evenly divide the number of target steps {num_target_steps} ")

  if len(np.unique(np.diff(targets_template.coords["time"].data))) > 1:
    raise ValueError("The targets time coordinates must be evenly spaced")

  # Similar to autoregressive.py, the static time coordinates we will pass for
  # the time axis at each iteration will be the same as that of the first chunk.
  # This not only prevents recompilation, but also helps thinking of the
  # "time" coord as relative to the individual chunk steps. Any time jax
  # coordinates  will be different for each iteration (see autoregressive.py
  # for more details.)
  chunk_inputs_time_array = inputs.coords["time"].data
  chunk_targets_time_array = targets_template.isel(
      # ".data" is important so we only set the "time" coordinate, but not
      # other coordinates associated with the time dim.
      time=slice(0, num_steps_per_chunk)).coords["time"].data

  current_inputs = inputs
  if replicate_fn is not None:
    current_inputs = replicate_fn(current_inputs)
  if device_put_fn is not None:
    current_inputs = device_put_fn(current_inputs)

  del inputs  # Don't leave the reference hanging during the rollout.

  if pmap_devices is not None:
    # Note that while we are defining a new pmapped function every time we go
    # throuh this code, `pmap` will not actually retrace/recompile unless inputs
    # change, since `pmap` will tie the compilation to `_get_next_inputs`.
    get_next_inputs_fn = xarray_jax.pmap(
        _get_next_inputs, replica_axis, devices=pmap_devices)  # pyrefly: ignore[bad-argument-type]
  elif replica_axis is not None:
    # Note `replica_axis` is a static kwarg, so this will retrace/recompile if
    # `replica_axis` changes.
    get_next_inputs_fn = functools.partial(
        _get_next_inputs_fn_vmapped_jitted, replica_axis=replica_axis)
  else:
    get_next_inputs_fn = _get_next_inputs_jitted

  if pmap_devices is not None:
    # Note that while we are defining a new pmapped function every time we go
    # throuh this code, `pmap` will not actually retrace/recompile unless inputs
    # change, since `pmap` will tie the compilation to `_split_rng_fn`.
    split_rng_fn = jax.pmap(
        _split_rng_fn, devices=pmap_devices, axis_name=replica_axis
    )
  elif replica_axis is not None:
    split_rng_fn = _split_rng_fn_vmapped_jitted
  else:
    split_rng_fn = _split_rng_fn_jitted

  for chunk_index in range(num_chunks):
    if verbose:
      logging.info("Chunk %d/%d", chunk_index, num_chunks)
      logging.flush()

    # Select targets for the time period that we are predicting for this chunk.
    target_offset = num_steps_per_chunk * chunk_index
    target_slice = slice(target_offset, target_offset + num_steps_per_chunk)
    current_targets_template = targets_template.isel(
        time=target_slice).compute()
    current_forcings = forcings.isel(
        time=target_slice).compute()

    # Coordinates that we will add later to the predictions.
    # We get this before we replicate the targets template data, to avoid
    # replicated jax coordinates on the outputs, also because we don't trust
    # the predictor to have kept all of them.
    time_coords_to_override = {
        n: c for n, c in current_targets_template.coords.items()
        if "time" in c.dims}

    if replicate_fn is not None:
      current_forcings = replicate_fn(current_forcings)
      current_targets_template = replicate_fn(current_targets_template)
    if device_put_fn is not None:
      current_forcings = device_put_fn(current_forcings)
      current_targets_template = device_put_fn(current_targets_template)

    # Make predictions for the chunk.
    rng, this_rng = split_rng_fn(rng)

    # We make it so the "time" coord of all arguments is fixed and the same
    # as the first chunk for all iterations, as it is relative to the chunk.
    # Jax coordinates containing information about the leadtime will be
    # different.
    current_inputs = current_inputs.assign_coords(
        time=chunk_inputs_time_array)
    current_forcings = current_forcings.assign_coords(
        time=chunk_targets_time_array)
    current_targets_template = current_targets_template.assign_coords(
        time=chunk_targets_time_array)
    predictions = predictor_fn(
        rng=this_rng,
        inputs=current_inputs,
        targets_template=current_targets_template,
        forcings=current_forcings)
    del current_targets_template

    # Build the inputs for the next step, while we are stil in relative
    # coordinates to the first step.
    if chunk_index == num_chunks - 1:
      # No need to call `_get_next_inputs` on the last iteration.
      current_inputs = None
    else:
      # Note how we are doing this with the current_inputs and
      # current_forcings that are relative to the first step.
      next_frame = predictions.assign(current_forcings)
      current_inputs = get_next_inputs_fn(current_inputs, next_frame)
      del next_frame
    del current_forcings

    # At this point we can assign the actual time coordinates.
    predictions = predictions.assign_coords(time_coords_to_override)

    if output_datetime is not None:
      predictions.coords["datetime"] = output_datetime.isel(
          time=target_slice)
    # We yield as a device array, and leave it up to the caller site to
    # decide if it needs to keep it in device or not. Most caller sites will
    # want to convert it to a numpy array right away, which wll release the
    # device memory.
    predictions = [predictions]  # See POP_TRICK_EXPLANATION
    yield predictions.pop()


def _split_rng_fn(rng):
  # Note, this is *not* equivalent to `return jax.random.split(rng)`, because
  # by assigning to a tuple, the single numpy array returned by
  # `jax.random.split` actually gets split into two arrays, so when calling
  # the function with pmap the output is Tuple[Array, Array], where the
  # leading axis of each array is `num devices`.
  rng1, rng2 = jax.random.split(rng)
  return rng1, rng2

_split_rng_fn_jitted = jax.jit(_split_rng_fn)
_split_rng_fn_vmapped_jitted = jax.jit(jax.vmap(_split_rng_fn))


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

  # Apply concatenate next frame with inputs and crop what we don't need.
  num_inputs = prev_inputs.sizes["time"]
  return (
      xarray.concat(
          [prev_inputs, next_inputs], dim="time",
          data_vars="different", compat="equals")
      .tail(time=num_inputs))


_get_next_inputs_jitted = jax.jit(_get_next_inputs)


def _get_next_inputs_fn_vmapped(*args, replica_axis: str, **kwargs):
  return xarray_jax.vmap(_get_next_inputs, replica_axis)(*args, **kwargs)


_get_next_inputs_fn_vmapped_jitted = jax.jit(
    _get_next_inputs_fn_vmapped, static_argnames=["replica_axis"])


def extend_targets_template(
    targets_template: xarray.Dataset,
    required_num_steps: int,
    value: float | None = None,
    ) -> xarray.Dataset:
  """Extends `targets_template` to `required_num_steps` with lazy arrays.

  It uses lazy dask arrays of `value`, so it does not require instantiating the
  array in memory.

  Args:
    targets_template: Input template to extend.
    required_num_steps: Number of steps required in the returned template.
    value: Value to use for the extended steps. If None, use zeros.

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
    if value is None:
      # TODO(alvarosg): Consider setting value = 0, and use `dask.array.full`
      # below, if we can show that it does not impact performance.
      dask_data = dask.array.zeros(
          shape=tuple(shape),
          chunks=-1,  # Will give chunk info directly to `ChunksToZarr``.
          dtype=data_array.dtype)
    else:
      dask_data = dask.array.full(
          shape=tuple(shape),
          chunks=-1,  # Will give chunk info directly to `ChunksToZarr``.
          fill_value=value,
          dtype=data_array.dtype)

    coords = dict(data_array.coords)
    coords["time"] = extended_time

    if extended_datetime is not None:
      coords["datetime"] = ("time", extended_datetime)

    return xarray.DataArray(
        dims=dims,
        data=dask_data,
        coords=coords)

  return targets_template.map(extend_time)
