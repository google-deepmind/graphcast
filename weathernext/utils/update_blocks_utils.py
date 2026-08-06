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

"""Utils for update blocks."""

from typing import Any, Protocol, TypeVar

from weathernext.utils import data_modalities
import haiku as hk


DataModalityOrAny = data_modalities.Data | Any


class DataModalitiesCallable(Protocol):
  def __call__(
      self, *args: DataModalityOrAny, **kwargs: DataModalityOrAny
      ) ->  DataModalityOrAny | tuple[DataModalityOrAny, ...]:
    pass


DataModalitiesCallableVar = TypeVar(
    "DataModalitiesCallableVar", bound=DataModalitiesCallable)


# TODO(alvarosg): Consider adding some unit-test for this rather than relying
# on unit tests of call sites.
def modality_data_remat(
    fn: DataModalitiesCallableVar) -> DataModalitiesCallableVar:
  """Applies remat to the "data" of a function that takes data modalities.

  The idea is that the "remat" is applied with respect to the input and output
  `data` fields of the data modalities, but makes other modalities fields like
  lat/lon coordinates pass through as if the remat did not exist. This is so
  things like coordinates don't become "tracer" objects.

  It is implemented by passing anything that is not the data field as a closure,
  and returning everything but the data field by a non-local variable. The only
  limitation of this approach is that:
  * If a wrapped function makes some pure transformations to the coordinates,
    between inputs and outputs those transformations will not be rematerialized.
  * If a wrapped function makes a transformation to the output coordinates
    that depends on the input data, then a jax leaker error will be thrown.

  In practice, we plan to mostly use this to wrap models that only transform the
  data, but not the coordinates, so it should no be a major limitation.

  The implementation works as follows:

  1. Separate the `.data` contained in the data modalities from the rest of the
    modality structure (modality object, coordinates, etc.). This is also
    necessary because `remat` cannot take arbitrary python objects as inputs
    (unless they are labelled as static arguments or registered with jax_tree).
  2. Pass only the `.data` fields as explicit inputs to the remat function.
  3. Once inside the remat function, inject the data structures back in via a
    closure, and merge it back with the input data.
  4. Call the input function with the merged inputs from step 3.
  5. Take the outputs from step 4, and separate data from the structures
    similar to step 1.
  6. Return the output structures via a non-local variable. This is necessary
    because remat cannot return arbitrary python objects (unless they are
    labelled as static arguments or registered with jax_tree).
  7. Return the output data via a regular output of the rematted function.
  8. Merge the output data back into the structures and return.

  Args:
    fn: A function that takes data modalities as args and kwargs, and one
      or more outputs, some of which may be data modalities.
  Returns:
    `fn` wrapped in `hk.remat`, such that the data of the data modalities are
    passed through `hk.remat`, and the rest is injected via the closure.
  """

  # TODO(alvarosg,matthjw): Decide if we want to make this work with arbitrary
  # tree containing data modalities, rather than just first level args/kwargs.

  def wrapped_fn(*args, **kwargs):

    # Extract data out of data modalities in args and kwargs.
    data_only_args = []
    for arg in args:
      if isinstance(arg, data_modalities.Data):
        arg = arg.data
      data_only_args.append(arg)

    data_only_kwargs = {}
    for key, arg in kwargs.items():
      if isinstance(arg, data_modalities.Data):
        arg = arg.data
      data_only_kwargs[key] = arg

    # We will return here all information necessary to restore the output data
    # modalities back into their original structures, via a non-local variable.
    outputs_structures = None

    @hk.remat
    def fn_to_remat(*data_only_args, **data_only_kwargs):

      # Restore the data back into the data modalities injected via the closure.
      restored_args = []
      for data_only_arg, arg in zip(data_only_args, args):
        if isinstance(arg, data_modalities.Data):
          arg = arg.replace_data(data_only_arg)
        restored_args.append(arg)
      restored_kwargs = {}
      for key, arg in kwargs.items():
        if isinstance(arg, data_modalities.Data):
          arg = arg.replace_data(data_only_kwargs[key])
        restored_kwargs[key] = arg

      # Run the function with the restored data modalities.
      outputs = fn(*restored_args, **restored_kwargs)

      # Need to distinguish and keep track whether there is a single output and
      # an output that consists of a single element. If the former, we treat it
      # as if it was a tuple of length 1 and then remove the tuple later.
      is_single_output = isinstance(outputs, data_modalities.Data)
      if is_single_output:
        outputs = tuple([outputs])

      # Separate the data from each data_modality, and put the structures to
      # be returned as `_StaticOutputModalityDataRemat`.
      outputs_data = []
      for output in outputs:
        if isinstance(output, data_modalities.Data):
          output = output.data
        outputs_data.append(output)

      if is_single_output:
        outputs_data = outputs_data[0]
        outputs = outputs[0]

      nonlocal outputs_structures
      outputs_structures = outputs

      return outputs_data

    # Call the function with only data as explicit input, and return
    # the output data only (without outptu structures).
    outputs_data = fn_to_remat(*data_only_args, **data_only_kwargs)

    # Need to distinguish and keep track whether there is a single output and
    # an output that consists of a single element. If the former, we treat it
    # as if it was a tuple of length 1 and then remove the tuple later.
    assert outputs_structures is not None
    is_single_output = isinstance(outputs_structures, data_modalities.Data)
    if is_single_output:
      outputs_data = tuple([outputs_data])
      outputs_structures = tuple([outputs_structures])

    # Restore the output data back into the data modalities outputs.
    outputs = []
    for output_data, aux_output in zip(outputs_data, outputs_structures):
      if isinstance(aux_output, data_modalities.Data):
        output = aux_output.replace_data(output_data)
      else:
        output = output_data
      outputs.append(output)

    outputs = tuple(outputs)
    if is_single_output:
      outputs = outputs[0]
    return outputs

  return wrapped_fn  # pyrefly: ignore[bad-return]
