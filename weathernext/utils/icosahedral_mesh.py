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
"""Utils for creating icosahedral meshes."""

from collections.abc import Callable
import itertools
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import scipy as sp
from scipy.spatial import transform


_POLE_PARALLEL_FACES_DEFAULT = True

BASE_MESH_LABEL = "icosahedron"


class PolygonalMesh(NamedTuple):
  """Data structure for meshes for which faces may have different adjacencies.

  Attributes:
    vertices: spatial positions of the vertices of the mesh of shape
        [num_vertices, num_dims].
    faces: list of faces of length [num_faces]. Each face i may have shape [N_i]
        where N_i is the adjacency of the corresponding face (3 for a traingular
        face, 5 pentagonal, 6 hexagonal.). Contains integer indices into
        `vertices`.

  """
  vertices: np.ndarray
  faces: Sequence[np.ndarray]


class TriangularMesh(PolygonalMesh):
  """Data structure for triangular meshes.

  Attributes:
    vertices: spatial positions of the vertices of the mesh of shape
        [num_vertices, num_dims].
    faces: triangular faces of the mesh of shape [num_faces, 3]. Contains
        integer indices into `vertices`.

  """
  vertices: np.ndarray  # pyrefly: ignore[bad-override]
  faces: np.ndarray  # pyrefly: ignore[bad-override]


class LabeledTriangularMesh(NamedTuple):
  """Data structure for a TriangularMesh with a label.

  Attributes:
    label: The label str.
    mesh: The TriangularMesh.
  """

  label: str
  mesh: TriangularMesh


def assert_all_meshes_compatible(mesh_list: Sequence[TriangularMesh]) -> None:
  """Checks that each n+1 finer mesh contains the n-th's coarser vertices."""
  for mesh_i, mesh_ip1 in itertools.pairwise(mesh_list):
    num_nodes_mesh_i = mesh_i.vertices.shape[0]
    assert np.allclose(mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i])


def merge_meshes(
    mesh_list: Sequence[TriangularMesh]) -> TriangularMesh:
  """Merges all meshes into one. Assumes the last mesh is the finest.

  Args:
     mesh_list: Sequence of meshes, from coarse to fine refinement levels. The
       vertices and faces may contain those from preceding, coarser levels.

  Returns:
     `TriangularMesh` for which the vertices correspond to the highest
     resolution mesh in the hierarchy, and the faces are the join set of the
     faces at all levels of the hierarchy.
  """
  assert_all_meshes_compatible(mesh_list)
  return TriangularMesh(
      vertices=mesh_list[-1].vertices,
      faces=np.concatenate([mesh.faces for mesh in mesh_list], axis=0))


def get_hierarchy_of_triangular_meshes_for_sphere(
    splits: int,
    pole_parallel_faces: bool = _POLE_PARALLEL_FACES_DEFAULT,
    ) -> List[TriangularMesh]:
  """Returns a sequence of meshes, each with triangularization sphere.

  Starting with a regular icosahedron (12 vertices, 20 faces, 30 edges) with
  circumscribed unit sphere. Then, each triangular face is iteratively
  subdivided into 4 triangular faces `splits` times. The new vertices are then
  projected back onto the unit sphere. All resulting meshes are returned in a
  list, from lowest to highest resolution.

  The vertices in each face are specified in counter-clockwise order as
  observed from the outside the icosahedron.

  Args:
     splits: How many times to split each triangle.
     pole_parallel_faces: If True, for the initial icosahedron, the top and
         bottom face will be parallel to the X-Y plane, which avoids having a
         vertex exactly at the poles of the sphere.

  Returns:
     Sequence of `TriangularMesh`s of length `splits + 1` each with:

       vertices: [num_vertices, 3] vertex positions in 3D, all with unit norm.
       faces: [num_faces, 3] with triangular faces joining sets of 3 vertices.
           Each row contains three indices into the vertices array, indicating
           the vertices adjacent to the face. Always with positive orientation
           (counterclock-wise when looking from the outside).
  """
  current_mesh = get_icosahedron(pole_parallel_faces=pole_parallel_faces)
  output_meshes = [current_mesh]
  for _ in range(splits):
    current_mesh = _two_split_unit_sphere_triangle_faces(current_mesh)
    output_meshes.append(current_mesh)
  return output_meshes


def get_icosahedron(
    pole_parallel_faces: bool = _POLE_PARALLEL_FACES_DEFAULT) -> TriangularMesh:
  """Returns a regular icosahedral mesh with circumscribed unit sphere.

  See https://en.wikipedia.org/wiki/Regular_icosahedron#Cartesian_coordinates
  for details on the construction of the regular icosahedron.

  The vertices in each face are specified in counter-clockwise order as observed
  from the outside of the icosahedron.

  Args:
     pole_parallel_faces: If True, the top and bottom face will be parallel
         to the X-Y plane, which prevents having nodes exactly at the north/
         south pole. If False, the top will be an arist parallel to the y axis.

  Returns:
     TriangularMesh with:

     vertices: [num_vertices=12, 3] vertex positions in 3D, all with unit norm.
     faces: [num_faces=20, 3] with triangular faces joining sets of 3 vertices.
         Each row contains three indices into the vertices array, indicating
         the vertices adjacent to the face. Always with positive orientation (
         counterclock-wise when looking from the outside).

  """
  phi = (1 + np.sqrt(5)) / 2
  vertices = []
  for c1 in [1., -1.]:
    for c2 in [phi, -phi]:
      vertices.append((c1, c2, 0.))
      vertices.append((0., c1, c2))
      vertices.append((c2, 0., c1))

  vertices = np.array(vertices, dtype=np.float32)
  vertices /= np.linalg.norm([1., phi])

  # I did this manually, checking the orientation one by one.
  faces = [(0, 1, 2),
           (0, 6, 1),
           (8, 0, 2),
           (8, 4, 0),
           (3, 8, 2),
           (3, 2, 7),
           (7, 2, 1),
           (0, 4, 6),
           (4, 11, 6),
           (6, 11, 5),
           (1, 5, 7),
           (4, 10, 11),
           (4, 8, 10),
           (10, 8, 3),
           (10, 3, 9),
           (11, 10, 9),
           (11, 9, 5),
           (5, 9, 7),
           (9, 3, 7),
           (1, 6, 5),
           ]

  if pole_parallel_faces:
    # By default the top is an aris parallel to the Y axis.
    # Need to rotate around the y axis by half the supplementary to the
    # angle between faces divided by two to get the desired orientation.
    #                          /O\  (top arist)
    #                     /          \                           Z
    # (adjacent face)/                    \  (adjacent face)     ^
    #           /     angle_between_faces      \                 |
    #      /                                        \            |
    #  /                                                 \      YO-----> X
    # This results in:
    #  (adjacent faceis now top plane)
    #  ----------------------O\  (top arist)
    #                           \
    #                             \
    #                               \     (adjacent face)
    #                                 \
    #                                   \
    #                                     \

    angle_between_faces = 2 * np.arcsin(phi / np.sqrt(3))
    rotation_angle = (np.pi - angle_between_faces) / 2
    rotation = transform.Rotation.from_euler(seq="y", angles=rotation_angle)
    rotation_matrix = rotation.as_matrix()
    vertices = np.dot(vertices, rotation_matrix)

  return TriangularMesh(vertices=vertices.astype(np.float32),
                        faces=np.array(faces, dtype=np.int32))


def _two_split_unit_sphere_triangle_faces(
    triangular_mesh: TriangularMesh) -> TriangularMesh:
  """Splits each triangular face into 4 triangles keeping the orientation."""

  # Every time we split a triangle into 4 we will be adding 3 extra vertices,
  # located at the edge centres.
  # This class handles the positioning of the new vertices, and avoids creating
  # duplicates.
  new_vertices_builder = _ChildVerticesBuilder(triangular_mesh.vertices)

  new_faces = []
  for ind1, ind2, ind3 in triangular_mesh.faces:
    # Transform each triangular face into 4 triangles,
    # preserving the orientation.
    #                    ind3
    #                   /    \
    #                /          \
    #              /      #3       \
    #            /                  \
    #         ind31 -------------- ind23
    #         /   \                /   \
    #       /       \     #4     /      \
    #     /    #1     \        /    #2    \
    #   /               \    /              \
    # ind1 ------------ ind12 ------------ ind2
    ind12 = new_vertices_builder.get_new_child_vertex_index((ind1, ind2))
    ind23 = new_vertices_builder.get_new_child_vertex_index((ind2, ind3))
    ind31 = new_vertices_builder.get_new_child_vertex_index((ind3, ind1))
    # Note how each of the 4 triangular new faces specifies the order of the
    # vertices to preserve the orientation of the original face. As the input
    # face should always be counter-clockwise as specified in the diagram,
    # this means child faces should also be counter-clockwise.
    new_faces.extend([[ind1, ind12, ind31],  # 1
                      [ind12, ind2, ind23],  # 2
                      [ind31, ind23, ind3],  # 3
                      [ind12, ind23, ind31],  # 4
                      ])
  return TriangularMesh(vertices=new_vertices_builder.get_all_vertices(),
                        faces=np.array(new_faces, dtype=np.int32))


def _three_split_unit_sphere_triangle_faces(
    triangular_mesh: TriangularMesh) -> TriangularMesh:
  """Splits each triangular face into 9 triangles keeping the orientation."""
  # Every time we split a triangle into 9 we will be adding 4 extra vertices,
  # one in the center of the triangle, and two equally distant ones between each
  # pairs of existing vertices.
  # This class handles the positioning of the new vertices, and avoids creating
  # duplicates.
  new_vertices_builder = _ChildVerticesBuilder(triangular_mesh.vertices)

  new_faces = []
  for ind1, ind2, ind3 in triangular_mesh.faces:
    # Transform each triangular face into 8 triangles,
    # preserving the orientation.
    #                              ind3
    #                             /    \
    #                          /          \
    #                        /      #9      \
    #                      /                  \
    #                   ind331 ------------ ind233
    #                   /   \                /   \
    #                 /       \     #8     /      \
    #               /    #6     \        /    #7    \
    #             /               \    /              \
    #         ind311 ------------ ind123 ----------- ind223
    #         /   \                /   \             /   \
    #       /       \     #4     /      \    #5    /      \
    #     /    #1     \        /    #2    \      /    #3    \
    #   /               \    /              \  /              \
    # ind1 ----------- ind112 ------------ ind122 ----------- ind2
    ind112 = new_vertices_builder.get_new_child_vertex_index((ind1, ind1, ind2))
    ind122 = new_vertices_builder.get_new_child_vertex_index((ind1, ind2, ind2))
    ind223 = new_vertices_builder.get_new_child_vertex_index((ind2, ind2, ind3))
    ind233 = new_vertices_builder.get_new_child_vertex_index((ind2, ind3, ind3))
    ind331 = new_vertices_builder.get_new_child_vertex_index((ind3, ind3, ind1))
    ind311 = new_vertices_builder.get_new_child_vertex_index((ind3, ind1, ind1))
    ind123 = new_vertices_builder.get_new_child_vertex_index((ind1, ind2, ind3))
    # Note how each of the 9 triangular new faces specifies the order of the
    # vertices to preserve the orientation of the original face. As the input
    # face should always be counter-clockwise as specified in the diagram,
    # this means child faces should also be counter-clockwise.
    new_faces.extend([[ind1, ind112, ind311],  # 1
                      [ind112, ind122, ind123],  # 2
                      [ind122, ind2, ind223],  # 3
                      [ind112, ind123, ind311],  # 4
                      [ind122, ind223, ind123],  # 5
                      [ind311, ind123, ind331],  # 6
                      [ind123, ind223, ind233],  # 7
                      [ind123, ind233, ind331],  # 8
                      [ind331, ind233, ind3],  # 9
                      ])
  return TriangularMesh(vertices=new_vertices_builder.get_all_vertices(),
                        faces=np.array(new_faces, dtype=np.int32))


class _ChildVerticesBuilder(object):
  """Bookkeeping of new child vertices added to an existing set of vertices."""

  def __init__(self, parent_vertices):

    # Because the same new vertex will be required when splitting adjacent
    # triangles (which share an edge) we keep them in a hash table indexed by
    # sorted indices of the vertices adjacent to the edge, to avoid creating
    # duplicated child vertices.
    self._child_vertices_index_mapping = {}
    self._parent_vertices = parent_vertices
    # We start with all previous vertices.
    self._all_vertices_list = list(parent_vertices)

  def _get_child_vertex_key(self, parent_vertex_indices):
    return tuple(sorted(parent_vertex_indices))

  def _create_child_vertex(self, parent_vertex_indices):
    """Creates a new vertex."""
    # Position for new vertex is the middle point, between the parent points,
    # projected to unit sphere.
    child_vertex_position = self._parent_vertices[
        list(parent_vertex_indices)].mean(0)
    child_vertex_position /= np.linalg.norm(child_vertex_position)

    # Add the vertex to the output list. The index for this new vertex will
    # match the length of the list before adding it.
    child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
    self._child_vertices_index_mapping[child_vertex_key] = len(
        self._all_vertices_list)
    self._all_vertices_list.append(child_vertex_position)

  def get_new_child_vertex_index(self, parent_vertex_indices):
    """Returns index for a child vertex, creating it if necessary."""
    # Get the key to see if we already have a new vertex in the middle.
    child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
    if child_vertex_key not in self._child_vertices_index_mapping:
      self._create_child_vertex(parent_vertex_indices)
    return self._child_vertices_index_mapping[child_vertex_key]

  def get_all_vertices(self):
    """Returns an array with old vertices."""
    return np.array(self._all_vertices_list)


def faces_to_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Transforms polygonal faces to sender and receiver indices.

  It does so by transforming every face into N_i edges. Such if the triangular
  face has indices [0, 1, 2], three edges are added 0->1, 1->2, and 2->0.

  If all faces have consistent orientation, and the surface represented by the
  faces is closed, then every edge in a polygon with a certain orientation
  is also part of another polygon with the opposite orientation. In this
  situation, the edges returned by the method are always bidirectional.

  Args:
    faces: Integer array of shape [num_faces, 3]. Contains node indices
        adjacent to each face.
  Returns:
    Tuple with sender/receiver indices, each of shape [num_edges=num_faces*3].

  """
  assert faces.ndim == 2
  assert faces.shape[-1] == 3
  senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
  receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
  return senders, receivers


def get_last_triangular_mesh_for_sphere(splits: int) -> TriangularMesh:
  return get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)[-1]


def build_mesh_label(
    previous_mesh_label: Optional[str] = None, split_size: Optional[int] = None
) -> str:
  if previous_mesh_label is None:
    return BASE_MESH_LABEL
  return f"{previous_mesh_label}_{split_size}"


def get_sparse_adjacency_matrix(mesh: TriangularMesh) -> sp.sparse.csr_matrix:
  """Returns a sparse adjacency matrix for the mesh with basic connectivity."""
  num_mesh_nodes = mesh.vertices.shape[0]
  senders, receivers = faces_to_edges(mesh.faces)
  adj_mat = sp.sparse.csr_matrix((num_mesh_nodes, num_mesh_nodes))
  adj_mat[senders, receivers] = 1
  return adj_mat


def get_permutation_to_banded(
    mesh: TriangularMesh,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
  adj_mat = get_sparse_adjacency_matrix(mesh)
  # Permutation to banded (this algorithm is deterministic, a given sparse
  # adjacency matrix will yield the same permutation every time this is run).
  permutation = sp.sparse.csgraph.reverse_cuthill_mckee(
      adj_mat, symmetric_mode=True
  )
  vertex_permutation_map = {j: i for i, j in enumerate(permutation)}
  permute_func = np.vectorize(lambda x: vertex_permutation_map[x])
  return permutation, permute_func


def get_hierarchy_of_labeled_triangular_meshes_for_sphere(
    *,
    splits_list: Sequence[int],
    pole_parallel_faces: bool = _POLE_PARALLEL_FACES_DEFAULT,
    finest_mesh_first: bool = False,
    starting_refinement_level: int = 0,
) -> List[LabeledTriangularMesh]:
  """Returns a sequence of (label, mesh), each with triangularization sphere.

  Starting with a regular icosahedron (12 vertices, 20 faces, 30 edges) with
  circumscribed unit sphere. Then, for each value n of the `splits_list`, each
  triangular face is iteratively subdivided into n^2 triangular faces (only
  2-way splits are supported). The new vertices are then projected back onto the
  unit sphere. All resulting meshes are returned in a list, from lowest to
  highest resolution (default) or from highest to lowest resolution (if
  finest_mesh_first=True).

  The vertices in each face are specified in counter-clockwise order as
  observed from the outside the icosahedron.

  Args:
     splits_list: Sequence containing the split size for each level of the
       hierarchy with respect to the previous one. Allowed values are 2 or 3.
     pole_parallel_faces: If True, for the initial icosahedron, the top and
         bottom face will be parallel to the X-Y plane, which avoids having a
         vertex exactly at the poles of the sphere. Note that 3-way splits
         will still put nodes at the poles even if this is set to True.
     finest_mesh_first: if False, the first mesh is the coarsest one, if True
       the first mesh is the finest one.
     starting_refinement_level: Which level of the refinement to start from.
       Used to get a multiscale mesh without some of the coarser levels.

  Returns:
     Sequence of `LabeledTriangularMesh`s of length len(splits_list) + 1, where
     each mesh has:

       vertices: [num_vertices, 3] vertex positions in 3D, all with unit norm.
       faces: [num_faces, 3] with triangular faces joining sets of 3 vertices.
           Each row contains three indices into the vertices array, indicating
           the vertices adjacent to the face. Always with positive orientation
           (counterclock-wise when looking from the outside).

     If finest_mesh_first=False, mesh[i+1] corresponds to a refinement
     of mesh[i] by splitting each face edge into splits_list[i] segments. In
     practice edges (triangular faces) are always split into 2 or 3 edges (4 or
     9 faces).
  """
  current_mesh = get_icosahedron(pole_parallel_faces=pole_parallel_faces)
  label = build_mesh_label()
  output_meshes = [LabeledTriangularMesh(label=label, mesh=current_mesh)]
  for split_size in splits_list:
    if split_size == 2:
      current_mesh = _two_split_unit_sphere_triangle_faces(current_mesh)
    elif split_size == 3:
      current_mesh = _three_split_unit_sphere_triangle_faces(current_mesh)
    else:
      raise ValueError(f"Unexpected `splits_list` {splits_list}, "
                       "only 2 or 3 values are allowed")
    label = build_mesh_label(previous_mesh_label=label, split_size=split_size)
    output_meshes.append(LabeledTriangularMesh(label=label, mesh=current_mesh))
  if finest_mesh_first:
    output_meshes = list(reversed(output_meshes))
  return output_meshes[starting_refinement_level:]
