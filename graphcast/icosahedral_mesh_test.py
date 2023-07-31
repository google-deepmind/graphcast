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
"""Tests for icosahedral_mesh."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from graphcast import icosahedral_mesh
import numpy as np


def _get_mesh_spec(splits: int):
  """Returns size of the final icosahedral mesh resulting from the splitting."""
  num_vertices = 12
  num_faces = 20
  for _ in range(splits):
    # Each previous face adds three new vertices, but each vertex is shared
    # by two faces.
    num_vertices += num_faces * 3 // 2
    num_faces *= 4
  return num_vertices, num_faces


class IcosahedralMeshTest(parameterized.TestCase):

  def test_icosahedron(self):
    mesh = icosahedral_mesh.get_icosahedron()
    _assert_valid_mesh(
        mesh, num_expected_vertices=12, num_expected_faces=20)

  @parameterized.parameters(list(range(5)))
  def test_get_hierarchy_of_triangular_meshes_for_sphere(self, splits):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=splits)
    prev_vertices = None
    for mesh_i, mesh in enumerate(meshes):
      # Check that `mesh` is valid.
      num_expected_vertices, num_expected_faces = _get_mesh_spec(mesh_i)
      _assert_valid_mesh(mesh, num_expected_vertices, num_expected_faces)

      # Check that the first N vertices from this mesh match all of the
      # vertices from the previous mesh.
      if prev_vertices is not None:
        leading_mesh_vertices = mesh.vertices[:prev_vertices.shape[0]]
        np.testing.assert_array_equal(leading_mesh_vertices, prev_vertices)

      # Increase the expected/previous values for the next iteration.
      if mesh_i < len(meshes) - 1:
        prev_vertices = mesh.vertices

  @parameterized.parameters(list(range(4)))
  def test_merge_meshes(self, splits):
    mesh_hierarchy = (
        icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
            splits=splits))
    mesh = icosahedral_mesh.merge_meshes(mesh_hierarchy)

    expected_faces = np.concatenate([m.faces for m in mesh_hierarchy], axis=0)
    np.testing.assert_array_equal(mesh.vertices, mesh_hierarchy[-1].vertices)
    np.testing.assert_array_equal(mesh.faces, expected_faces)

  def test_faces_to_edges(self):

    faces = np.array([[0, 1, 2],
                      [3, 4, 5]])

    # This also documents the order of the edges returned by the method.
    expected_edges = np.array(
        [[0, 1],
         [3, 4],
         [1, 2],
         [4, 5],
         [2, 0],
         [5, 3]])
    expected_senders = expected_edges[:, 0]
    expected_receivers = expected_edges[:, 1]

    senders, receivers = icosahedral_mesh.faces_to_edges(faces)

    np.testing.assert_array_equal(senders, expected_senders)
    np.testing.assert_array_equal(receivers, expected_receivers)


def _assert_valid_mesh(mesh, num_expected_vertices, num_expected_faces):
  vertices = mesh.vertices
  faces = mesh.faces
  chex.assert_shape(vertices, [num_expected_vertices, 3])
  chex.assert_shape(faces, [num_expected_faces, 3])

  # Vertices norm should be 1.
  vertices_norm = np.linalg.norm(vertices, axis=-1)
  np.testing.assert_allclose(vertices_norm, 1., rtol=1e-6)

  _assert_positive_face_orientation(vertices, faces)


def _assert_positive_face_orientation(vertices, faces):

  # Obtain a unit vector that points, in the direction of the face.
  face_orientation = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                              vertices[faces[:, 2]] - vertices[faces[:, 1]])
  face_orientation /= np.linalg.norm(face_orientation, axis=-1, keepdims=True)

  # And a unit vector pointing from the origin to the center of the face.
  face_centers = vertices[faces].mean(1)
  face_centers /= np.linalg.norm(face_centers, axis=-1, keepdims=True)

  # Positive orientation means those two vectors should be parallel
  # (dot product, 1), and not anti-parallel (dot product, -1).
  dot_center_orientation = np.einsum("ik,ik->i", face_orientation, face_centers)

  # Check that the face normal is parallel to the vector that joins the center
  # of the face to the center of the sphere. Note we need a small tolerance
  # because some discretizations are not exactly uniform, so it will not be
  # exactly parallel.
  np.testing.assert_allclose(dot_center_orientation, 1., atol=6e-4)


if __name__ == "__main__":
  absltest.main()
