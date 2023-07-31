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
"""Serialize and deserialize trees."""

import dataclasses
import io
import types
from typing import Any, BinaryIO, Optional, TypeVar

import numpy as np

_T = TypeVar("_T")


def dump(dest: BinaryIO, value: Any) -> None:
  """Dump a tree of dicts/dataclasses to a file object.

  Args:
    dest: a file object to write to.
    value: A tree of dicts, lists, tuples and dataclasses of numpy arrays and
      other basic types. Unions are not supported, other than Optional/None
      which is only supported in dataclasses, not in dicts, lists or tuples.
      All leaves must be coercible to a numpy array, and recoverable as a single
      arg to a type.
  """
  buffer = io.BytesIO()  # In case the destination doesn't support seeking.
  np.savez(buffer, **_flatten(value))
  dest.write(buffer.getvalue())


def load(source: BinaryIO, typ: type[_T]) -> _T:
  """Load from a file object and convert it to the specified type.

  Args:
    source: a file object to read from.
    typ: a type object that acts as a schema for deserialization. It must match
      what was serialized. If a type is Any, it will be returned however numpy
      serialized it, which is what you want for a tree of numpy arrays.

  Returns:
    the deserialized value as the specified type.
  """
  return _convert_types(typ, _unflatten(np.load(source)))


_SEP = ":"


def _flatten(tree: Any) -> dict[str, Any]:
  """Flatten a tree of dicts/dataclasses/lists/tuples to a single dict."""
  if dataclasses.is_dataclass(tree):
    # Don't use dataclasses.asdict as it is recursive so skips dropping None.
    tree = {f.name: v for f in dataclasses.fields(tree)
            if (v := getattr(tree, f.name)) is not None}
  elif isinstance(tree, (list, tuple)):
    tree = dict(enumerate(tree))

  assert isinstance(tree, dict)

  flat = {}
  for k, v in tree.items():
    k = str(k)
    assert _SEP not in k
    if dataclasses.is_dataclass(v) or isinstance(v, (dict, list, tuple)):
      for a, b in _flatten(v).items():
        flat[f"{k}{_SEP}{a}"] = b
    else:
      assert v is not None
      flat[k] = v
  return flat


def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
  """Unflatten a dict to a tree of dicts."""
  tree = {}
  for flat_key, v in flat.items():
    node = tree
    keys = flat_key.split(_SEP)
    for k in keys[:-1]:
      if k not in node:
        node[k] = {}
      node = node[k]
    node[keys[-1]] = v
  return tree


def _convert_types(typ: type[_T], value: Any) -> _T:
  """Convert some structure into the given type. The structures must match."""
  if typ in (Any, ...):
    return value

  if typ in (int, float, str, bool):
    return typ(value)

  if typ is np.ndarray:
    assert isinstance(value, np.ndarray)
    return value

  if dataclasses.is_dataclass(typ):
    kwargs = {}
    for f in dataclasses.fields(typ):
      # Only support Optional for dataclasses, as numpy can't serialize it
      # directly (without pickle), and dataclasses are the only case where we
      # can know the full set of values and types and therefore know the
      # non-existence must mean None.
      if isinstance(f.type, (types.UnionType, type(Optional[int]))):
        constructors = [t for t in f.type.__args__ if t is not types.NoneType]
        if len(constructors) != 1:
          raise TypeError(
              "Optional works, Union with anything except None doesn't")
        if f.name not in value:
          kwargs[f.name] = None
          continue
        constructor = constructors[0]
      else:
        constructor = f.type

      if f.name in value:
        kwargs[f.name] = _convert_types(constructor, value[f.name])
      else:
        raise ValueError(f"Missing value: {f.name}")
    return typ(**kwargs)

  base_type = getattr(typ, "__origin__", None)

  if base_type is dict:
    assert len(typ.__args__) == 2
    key_type, value_type = typ.__args__
    return {_convert_types(key_type, k): _convert_types(value_type, v)
            for k, v in value.items()}

  if base_type is list:
    assert len(typ.__args__) == 1
    value_type = typ.__args__[0]
    return [_convert_types(value_type, v)
            for _, v in sorted(value.items(), key=lambda x: int(x[0]))]

  if base_type is tuple:
    if len(typ.__args__) == 2 and typ.__args__[1] == ...:
      # An arbitrary length tuple of a single type, eg: tuple[int, ...]
      value_type = typ.__args__[0]
      return tuple(_convert_types(value_type, v)
                   for _, v in sorted(value.items(), key=lambda x: int(x[0])))
    else:
      # A fixed length tuple of arbitrary types, eg: tuple[int, str, float]
      assert len(typ.__args__) == len(value)
      return tuple(
          _convert_types(t, v)
          for t, (_, v) in zip(
              typ.__args__, sorted(value.items(), key=lambda x: int(x[0]))))

  # This is probably unreachable with reasonable serializable inputs.
  try:
    return typ(value)
  except TypeError as e:
    raise TypeError(
        "_convert_types expects the type argument to be a dataclass defined "
        "with types that are valid constructors (eg tuple is fine, Tuple "
        "isn't), and accept a numpy array as the sole argument.") from e
