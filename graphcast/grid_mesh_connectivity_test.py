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
"""Tests for graphcast.grid_mesh_connectivity."""

from absl.testing import absltest
from graphcast import grid_mesh_connectivity
from graphcast import icosahedral_mesh
import numpy as np


class GridMeshConnectivityTest(absltest.TestCase):

  def test_grid_lat_lon_to_coordinates(self):

    # Intervals of 30 degrees.
    grid_latitude = np.array([-45., 0., 45])
    grid_longitude = np.array([0., 90., 180., 270.])

    inv_sqrt2 = 1 / np.sqrt(2)
    expected_coordinates = np.array([
        [[inv_sqrt2, 0., -inv_sqrt2],
         [0., inv_sqrt2, -inv_sqrt2],
         [-inv_sqrt2, 0., -inv_sqrt2],
         [0., -inv_sqrt2, -inv_sqrt2]],
        [[1., 0., 0.],
         [0., 1., 0.],
         [-1., 0., 0.],
         [0., -1., 0.]],
        [[inv_sqrt2, 0., inv_sqrt2],
         [0., inv_sqrt2, inv_sqrt2],
         [-inv_sqrt2, 0., inv_sqrt2],
         [0., -inv_sqrt2, inv_sqrt2]],
    ])

    coordinates = grid_mesh_connectivity._grid_lat_lon_to_coordinates(
        grid_latitude, grid_longitude)
    np.testing.assert_allclose(expected_coordinates, coordinates, atol=1e-15)

  def test_radius_query_indices_smoke(self):
    # TODO(alvarosg): Add non-smoke test?
    grid_latitude = np.linspace(-75, 75, 6)
    grid_longitude = np.arange(12) * 30.
    mesh = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=3)[-1]
    grid_mesh_connectivity.radius_query_indices(
        grid_latitude=grid_latitude,
        grid_longitude=grid_longitude,
        mesh=mesh, radius=0.2)

  def test_in_mesh_triangle_indices_smoke(self):
    # TODO(alvarosg): Add non-smoke test?
    grid_latitude = np.linspace(-75, 75, 6)
    grid_longitude = np.arange(12) * 30.
    mesh = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=3)[-1]
    grid_mesh_connectivity.in_mesh_triangle_indices(
        grid_latitude=grid_latitude,
        grid_longitude=grid_longitude,
        mesh=mesh)


if __name__ == "__main__":
  absltest.main()
