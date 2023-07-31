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
"""Tools for converting from regular grids on a sphere, to triangular meshes."""

from graphcast import icosahedral_mesh
import numpy as np
import scipy
import trimesh


def _grid_lat_lon_to_coordinates(
    grid_latitude: np.ndarray, grid_longitude: np.ndarray) -> np.ndarray:
  """Lat [num_lat] lon [num_lon] to 3d coordinates [num_lat, num_lon, 3]."""
  # Convert to spherical coordinates phi and theta defined in the grid.
  # Each [num_latitude_points, num_longitude_points]
  phi_grid, theta_grid = np.meshgrid(
      np.deg2rad(grid_longitude),
      np.deg2rad(90 - grid_latitude))

  # [num_latitude_points, num_longitude_points, 3]
  # Note this assumes unit radius, since for now we model the earth as a
  # sphere of unit radius, and keep any vertical dimension as a regular grid.
  return np.stack(
      [np.cos(phi_grid)*np.sin(theta_grid),
       np.sin(phi_grid)*np.sin(theta_grid),
       np.cos(theta_grid)], axis=-1)


def radius_query_indices(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: icosahedral_mesh.TriangularMesh,
    radius: float) -> tuple[np.ndarray, np.ndarray]:
  """Returns mesh-grid edge indices for radius query.

  Args:
    grid_latitude: Latitude values for the grid [num_lat_points]
    grid_longitude: Longitude values for the grid [num_lon_points]
    mesh: Mesh object.
    radius: Radius of connectivity in R3. for a sphere of unit radius.

  Returns:
    tuple with `grid_indices` and `mesh_indices` indicating edges between the
    grid and the mesh such that the distances in a straight line (not geodesic)
    are smaller than or equal to `radius`.
    * grid_indices: Indices of shape [num_edges], that index into a
      [num_lat_points, num_lon_points] grid, after flattening the leading axes.
    * mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
  """

  # [num_grid_points=num_lat_points * num_lon_points, 3]
  grid_positions = _grid_lat_lon_to_coordinates(
      grid_latitude, grid_longitude).reshape([-1, 3])

  # [num_mesh_points, 3]
  mesh_positions = mesh.vertices
  kd_tree = scipy.spatial.cKDTree(mesh_positions)

  # [num_grid_points, num_mesh_points_per_grid_point]
  # Note `num_mesh_points_per_grid_point` is not constant, so this is a list
  # of arrays, rather than a 2d array.
  query_indices = kd_tree.query_ball_point(x=grid_positions, r=radius)

  grid_edge_indices = []
  mesh_edge_indices = []
  for grid_index, mesh_neighbors in enumerate(query_indices):
    grid_edge_indices.append(np.repeat(grid_index, len(mesh_neighbors)))
    mesh_edge_indices.append(mesh_neighbors)

  # [num_edges]
  grid_edge_indices = np.concatenate(grid_edge_indices, axis=0).astype(int)
  mesh_edge_indices = np.concatenate(mesh_edge_indices, axis=0).astype(int)

  return grid_edge_indices, mesh_edge_indices


def in_mesh_triangle_indices(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: icosahedral_mesh.TriangularMesh) -> tuple[np.ndarray, np.ndarray]:
  """Returns mesh-grid edge indices for grid points contained in mesh triangles.

  Args:
    grid_latitude: Latitude values for the grid [num_lat_points]
    grid_longitude: Longitude values for the grid [num_lon_points]
    mesh: Mesh object.

  Returns:
    tuple with `grid_indices` and `mesh_indices` indicating edges between the
    grid and the mesh vertices of the triangle that contain each grid point.
    The number of edges is always num_lat_points * num_lon_points * 3
    * grid_indices: Indices of shape [num_edges], that index into a
      [num_lat_points, num_lon_points] grid, after flattening the leading axes.
    * mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
  """

  # [num_grid_points=num_lat_points * num_lon_points, 3]
  grid_positions = _grid_lat_lon_to_coordinates(
      grid_latitude, grid_longitude).reshape([-1, 3])

  mesh_trimesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

  # [num_grid_points] with mesh face indices for each grid point.
  _, _, query_face_indices = trimesh.proximity.closest_point(
      mesh_trimesh, grid_positions)

  # [num_grid_points, 3] with mesh node indices for each grid point.
  mesh_edge_indices = mesh.faces[query_face_indices]

  # [num_grid_points, 3] with grid node indices, where every row simply contains
  # the row (grid_point) index.
  grid_indices = np.arange(grid_positions.shape[0])
  grid_edge_indices = np.tile(grid_indices.reshape([-1, 1]), [1, 3])

  # Flatten to get a regular list.
  # [num_edges=num_grid_points*3]
  mesh_edge_indices = mesh_edge_indices.reshape([-1])
  grid_edge_indices = grid_edge_indices.reshape([-1])

  return grid_edge_indices, mesh_edge_indices
