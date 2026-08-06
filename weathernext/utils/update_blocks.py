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

"""Base classes for neural network blocks that update data."""

import abc
from typing import Protocol, TypeVar
from weathernext.utils import data_modalities
import haiku as hk

# Some blocks are forced to preserve the same type of child, across one or more
# inputs and outputs.
Data = data_modalities.Data

SingleArray = data_modalities.SingleArray
CombinedArrays = data_modalities.CombinedArrays
TriangularMeshData = data_modalities.TriangularMeshData
LatLonGridData = data_modalities.LatLonGridData
LatLonPointsData = data_modalities.LatLonPointsData

SingleOrCombinedArraysVar = TypeVar(
    "SingleOrCombinedArraysVar", SingleArray, CombinedArrays)

AnotherSingleOrCombinedArraysVar = TypeVar(
    "AnotherSingleOrCombinedArraysVar", SingleArray, CombinedArrays)

DataSingleOrCombinedArraysVar = TypeVar(
    "DataSingleOrCombinedArraysVar",
    Data[SingleArray],
    Data[CombinedArrays])

AnotherDataSingleOrCombinedArraysVar = TypeVar(
    "AnotherDataSingleOrCombinedArraysVar",
    Data[SingleArray],
    Data[CombinedArrays])

GlobalDataSingleOrCombinedArrays = data_modalities.GlobalData[
    SingleArray | CombinedArrays]


# TODO(alvarosg): Re-consider if we actually need these base classes.
# The reason why these base classes may be useful is that some architectures may
# allow to parameterize the update block type, and this may be useful to
# define the signature, although it may require to make the types be
# a bit more specific, such as `Data[SingleArray]`.
class SingleBlockUpdate(hk.Module):
  """Base class to update the data in a `Data`."""

  @abc.abstractmethod
  def __call__(self,
               data: DataSingleOrCombinedArraysVar,
               global_data: GlobalDataSingleOrCombinedArrays | None = None,
               is_training: bool = False,
               ) -> DataSingleOrCombinedArraysVar:
    """Updates features in `Data`."""


class PairedBlockUpdate(hk.Module):
  """Base class to simultaneously update a pair of `Data`."""

  @abc.abstractmethod
  def __call__(
      self, data_a: DataSingleOrCombinedArraysVar,
      data_b: AnotherDataSingleOrCombinedArraysVar,
      global_data: GlobalDataSingleOrCombinedArrays | None = None,
      is_training: bool = False,
  ) -> tuple[DataSingleOrCombinedArraysVar,
             AnotherDataSingleOrCombinedArraysVar]:
    """Updates features in a pair of `Data`."""


class PointsMeshUpdate(Protocol):
  """Protocol for functions simultaneously updating points and mesh data."""

  def __call__(
      self,
      triangular_mesh_data: TriangularMeshData[
          SingleOrCombinedArraysVar],
      lat_lon_points_data: LatLonPointsData[
          AnotherSingleOrCombinedArraysVar],
      global_data: GlobalDataSingleOrCombinedArrays | None = None,
      is_training: bool = False,
  ) -> tuple[TriangularMeshData[SingleOrCombinedArraysVar],
             LatLonPointsData[AnotherSingleOrCombinedArraysVar]]:
    """Updates the triangular mesh data and lat lon points data."""


class PointsMeshUpdateConstructor(Protocol):
  """Protocol for annotating functions that construct a `PointsMeshUpdate`."""

  def __call__(
      self,
      points_name: str,
      stacked_points_inputs: bool = False,
      **kwargs,
  ) -> PointsMeshUpdate:
    """Builds the update block."""


class MeshUpdate(Protocol):
  """Protocol for annotating functions that update `TriangularMeshData`."""

  def __call__(
      self,
      triangular_mesh_data: TriangularMeshData[
          SingleOrCombinedArraysVar],
      global_data: GlobalDataSingleOrCombinedArrays | None = None,
      is_training: bool = False,
  ) -> TriangularMeshData[SingleOrCombinedArraysVar]:
    """Updates the triangular mesh data."""


class MeshUpdateConstructor(Protocol):
  """Protocol for annotating functions that construct a `MeshUpdate`."""

  # Note that 'dense_kwargs' are not passed here because the mesh transformer
  # block has a different API. We may unify them in the future.
  # Also, the names need to be specified based on the constructor to maintain
  # backward compatibility. We therefore do not mandate it in the protocol.
  def __call__(
      self,
      **kwargs,
  ) -> MeshUpdate:
    """Builds the update block."""


def separate_combined_arrays(
    data: DataSingleOrCombinedArraysVar,
    ) -> tuple[SingleArray | None, SingleArray | None, SingleArray | None]:
  """Helper for data that may be a single array or CombinedArray`."""
  # TODO(alvarosg): Consider instead of this, writing a helper to get each
  # field using different calls, so in the case of adding more fields in the
  # future, we don't need to touch all call-sites.
  data = data.data  # pyrefly: ignore[bad-assignment]
  if isinstance(data, CombinedArrays):
    return data.main, data.norm_conditioning, data.other_conditioning
  else:
    return data, None, None  # pyrefly: ignore[bad-return]


def update_main_data(
    data: DataSingleOrCombinedArraysVar,
    new_array: SingleArray,
    ) -> DataSingleOrCombinedArraysVar:
  """Helper for data that may be a single array or CombinedArray`."""
  if isinstance(data.data, CombinedArrays):
    return data.replace_data(
        data.data._replace(main=new_array))
  else:
    return data.replace_data(new_array)
