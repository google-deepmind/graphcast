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
"""Tests for xarray_tree."""

from absl.testing import absltest
from graphcast import xarray_tree
import numpy as np
import xarray


TEST_DATASET = xarray.Dataset(
    data_vars={
        "foo": (("x", "y"), np.zeros((2, 3))),
        "bar": (("x",), np.zeros((2,))),
    },
    coords={
        "x": [1, 2],
        "y": [10, 20, 30],
    }
)


class XarrayTreeTest(absltest.TestCase):

  def test_map_structure_maps_over_leaves_but_preserves_dataset_type(self):
    def fn(leaf):
      self.assertIsInstance(leaf, xarray.DataArray)
      result = leaf + 1
      # Removing the name from the returned DataArray to test that we don't rely
      # on it being present to restore the correct names in the result:
      result = result.rename(None)
      return result

    result = xarray_tree.map_structure(fn, TEST_DATASET)
    self.assertIsInstance(result, xarray.Dataset)
    self.assertSameElements({"foo", "bar"}, result.keys())

  def test_map_structure_on_data_arrays(self):
    data_arrays = dict(TEST_DATASET)
    result = xarray_tree.map_structure(lambda x: x+1, data_arrays)
    self.assertIsInstance(result, dict)
    self.assertSameElements({"foo", "bar"}, result.keys())

  def test_map_structure_on_dataset_plain_dict_when_coords_incompatible(self):
    def fn(leaf):
      # Returns DataArrays that can't be exactly merged back into a Dataset
      # due to the coordinates not matching:
      if leaf.name == "foo":
        return xarray.DataArray(
            data=np.zeros(2), dims=("x",), coords={"x": [1, 2]})
      else:
        return xarray.DataArray(
            data=np.zeros(2), dims=("x",), coords={"x": [3, 4]})

    result = xarray_tree.map_structure(fn, TEST_DATASET)
    self.assertIsInstance(result, dict)
    self.assertSameElements({"foo", "bar"}, result.keys())

  def test_map_structure_on_dataset_drops_vars_with_none_return_values(self):
    def fn(leaf):
      return leaf if leaf.name == "foo" else None

    result = xarray_tree.map_structure(fn, TEST_DATASET)
    self.assertIsInstance(result, xarray.Dataset)
    self.assertSameElements({"foo"}, result.keys())

  def test_map_structure_on_dataset_returns_plain_dict_other_return_types(self):
    def fn(leaf):
      self.assertIsInstance(leaf, xarray.DataArray)
      return "not a DataArray"

    result = xarray_tree.map_structure(fn, TEST_DATASET)
    self.assertEqual({"foo": "not a DataArray",
                      "bar": "not a DataArray"}, result)

  def test_map_structure_two_args_different_variable_orders(self):
    dataset_different_order = TEST_DATASET[["bar", "foo"]]
    def fn(arg1, arg2):
      self.assertEqual(arg1.name, arg2.name)
    xarray_tree.map_structure(fn, TEST_DATASET, dataset_different_order)


if __name__ == "__main__":
  absltest.main()
