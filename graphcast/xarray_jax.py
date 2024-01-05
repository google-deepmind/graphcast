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
"""Helpers to use xarray.{Variable,DataArray,Dataset} with JAX.

Allows them to be based on JAX arrays without converting to numpy arrays under
the hood, so you can start with a JAX array, do some computation with it in
xarray-land, get a JAX array out the other end and (for example) jax.jit
through the whole thing. You can even jax.jit a function which accepts and
returns xarray.Dataset, DataArray and Variable.

## Creating xarray datatypes from jax arrays, and vice-versa.

You can use the xarray_jax.{Variable,DataArray,Dataset} constructors, which have
the same API as the standard xarray constructors but will accept JAX arrays
without converting them to numpy.

It does this by wrapping the JAX array in a wrapper before passing it to
xarray; you can also do this manually by calling xarray_jax.wrap on your JAX
arrays before passing them to the standard xarray constructors.

To get non-wrapped JAX arrays out the other end, you can use e.g.:

  xarray_jax.jax_vars(dataset)
  xarray_jax.jax_data(dataset.some_var)

which will complain if the data isn't actually a JAX array. Use this if you need
to make sure the computation has gone via JAX, e.g. if it's the output of code
that you want to JIT or compute gradients through. If this is not the case and
you want to support passing plain numpy arrays through as well as potentially
JAX arrays, you can use:

  xarray_jax.unwrap_vars(dataset)
  xarray_jax.unwrap_data(dataset.some_var)

which will unwrap the data if it is a wrapped JAX array, but otherwise pass
it through to you without complaint.

The wrapped JAX arrays aim to support all the core operations from the numpy
array API that xarray expects, however there may still be some gaps; if you run
into any problems around this, you may need to add a few more proxy methods onto
the wrapper class below.

In future once JAX and xarray support the new  Python array API standard
(https://data-apis.org/array-api/latest/index.html), we hope to avoid the need
for wrapping the JAX arrays like this.

## jax.jit and pmap of functions taking and returning xarray datatypes

We register xarray datatypes with jax.tree_util, which allows them to be treated
as generic containers of jax arrays by various parts of jax including jax.jit.

This allows for, e.g.:

  @jax.jit
  def foo(input: xarray.Dataset) -> xarray.Dataset:
    ...

It will not work out-of-the-box with shape-modifying transformations like
jax.pmap, or e.g. a jax.tree_util.tree_map with some transform that alters array
shapes or dimension order. That's because we won't know what dimension names
and/or coordinates to use when unflattening, if the results have a different
shape to the data that was originally flattened.

You can work around this using xarray_jax.dims_change_on_unflatten, however,
and in the case of jax.pmap we provide a wrapper xarray_jax.pmap which allows
it to be used with functions taking and returning xarrays.

## Treatment of coordinates

We don't support passing jax arrays as coordinates when constructing a
DataArray/Dataset. This is because xarray's advanced indexing and slicing is
unlikely to work with jax arrays (at least when a Tracer is used during
jax.jit), and also because some important datatypes used for coordinates, like
timedelta64 and datetime64, are not supported by jax.

For the purposes of tree_util and jax.jit, coordinates are not treated as leaves
of the tree (array data 'contained' by a Dataset/DataArray), they are just a
static part of the structure. That means that if a jit'ed function is called
twice with Dataset inputs that use different coordinates, it will compile a
separate version of the function for each. The coordinates are treated like
static_argnums by jax.jit.

If you want to use dynamic data for coordinates, we recommend making it a
data_var instead of a coord. You won't be able to do indexing and slicing using
the coordinate, but that wasn't going to work with a jax array anyway.
"""

import collections
import contextlib
import contextvars
from typing import Any, Callable, Hashable, Iterator, Mapping, Optional, Union, Tuple, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
import tree
import xarray


def Variable(dims, data, **kwargs) -> xarray.Variable:  # pylint:disable=invalid-name
  """Like xarray.Variable, but can wrap JAX arrays."""
  return xarray.Variable(dims, wrap(data), **kwargs)


_JAX_COORD_ATTR_NAME = '_jax_coord'


def DataArray(  # pylint:disable=invalid-name
    data,
    coords=None,
    dims=None,
    name=None,
    attrs=None,
    jax_coords=None,
    ) -> xarray.DataArray:
  """Like xarray.DataArray, but supports using JAX arrays.

  Args:
    data: As for xarray.DataArray, except jax arrays are also supported.
    coords: Coordinates for the array, see xarray.DataArray. These coordinates
      must be based on plain numpy arrays or something convertible to plain
      numpy arrays. Their values will form a static part of the data structure
      from the point of view of jax.tree_util. In particular this means these
      coordinates will be passed as plain numpy arrays even inside a JIT'd
      function, and the JIT'd function will be recompiled under the hood if the
      coordinates of DataArrays passed into it change.
      If this is not convenient for you, see also jax_coords below.
    dims: See xarray.DataArray.
    name: See xarray.DataArray.
    attrs: See xarray.DataArray.
    jax_coords: Additional coordinates, which *can* use JAX arrays. These
      coordinates will be treated as JAX data from the point of view of
      jax.tree_util, that means when JIT'ing they will be passed as tracers and
      computation involving them will be JIT'd.
      Unfortunately a side-effect of this is that they can't be used as index
      coordinates (because xarray's indexing logic is not JIT-able). If you
      specify a coordinate with the same name as a dimension here, it will not
      be set as an index coordinate; this behaviour is different to the default
      for `coords`, and it means that things like `.sel` based on the jax
      coordinate will not work.
      Note we require `jax_coords` to be explicitly specified via a different
      constructor argument to `coords`, rather than just looking for jax arrays
      within the `coords` and treating them differently. This is because it
      affects the way jax.tree_util treats them, which is somewhat orthogonal to
      whether the value is passed in as numpy or not, and generally needs to be
      handled consistently so is something we encourage explicit control over.

  Returns:
    An instance of xarray.DataArray. Where JAX arrays are used as data or
    coords, they will be wrapped with JaxArrayWrapper and can be unwrapped via
    `unwrap` and `unwrap_data`.
  """
  result = xarray.DataArray(
      wrap(data), dims=dims, name=name, attrs=attrs or {})
  return assign_coords(result, coords=coords, jax_coords=jax_coords)


def Dataset(  # pylint:disable=invalid-name
    data_vars,
    coords=None,
    attrs=None,
    jax_coords=None,
    ) -> xarray.Dataset:
  """Like xarray.Dataset, but can wrap JAX arrays.

  Args:
    data_vars: As for xarray.Dataset, except jax arrays are also supported.
    coords: Coordinates for the dataset, see xarray.Dataset. These coordinates
      must be based on plain numpy arrays or something convertible to plain
      numpy arrays. Their values will form a static part of the data structure
      from the point of view of jax.tree_util. In particular this means these
      coordinates will be passed as plain numpy arrays even inside a JIT'd
      function, and the JIT'd function will be recompiled under the hood if the
      coordinates of DataArrays passed into it change.
      If this is not convenient for you, see also jax_coords below.
    attrs: See xarray.Dataset.
    jax_coords: Additional coordinates, which *can* use JAX arrays. These
      coordinates will be treated as JAX data from the point of view of
      jax.tree_util, that means when JIT'ing they will be passed as tracers and
      computation involving them will be JIT'd.
      Unfortunately a side-effect of this is that they can't be used as index
      coordinates (because xarray's indexing logic is not JIT-able). If you
      specify a coordinate with the same name as a dimension here, it will not
      be set as an index coordinate; this behaviour is different to the default
      for `coords`, and it means that things like `.sel` based on the jax
      coordinate will not work.
      Note we require `jax_coords` to be explicitly specified via a different
      constructor argument to `coords`, rather than just looking for jax arrays
      within the `coords` and treating them differently. This is because it
      affects the way jax.tree_util treats them, which is somewhat orthogonal to
      whether the value is passed in as numpy or not, and generally needs to be
      handled consistently so is something we encourage explicit control over.

  Returns:
    An instance of xarray.Dataset. Where JAX arrays are used as data, they
    will be wrapped with JaxArrayWrapper.
  """
  wrapped_data_vars = {}
  for name, var_like in data_vars.items():
    # xarray.Dataset accepts a few different formats for data_vars:
    if isinstance(var_like, jax.Array):
      wrapped_data_vars[name] = wrap(var_like)
    elif isinstance(var_like, tuple):
      # Layout is (dims, data, ...). We wrap data.
      wrapped_data_vars[name] = (var_like[0], wrap(var_like[1])) + var_like[2:]
    else:
      # Could be a plain numpy array or scalar (we don't wrap), or an
      # xarray.Variable, DataArray etc, which we must assume is already wrapped
      # if necessary (e.g. if creating using xarray_jax.{Variable,DataArray}).
      wrapped_data_vars[name] = var_like

  result = xarray.Dataset(
      data_vars=wrapped_data_vars,
      attrs=attrs)

  return assign_coords(result, coords=coords, jax_coords=jax_coords)


DatasetOrDataArray = TypeVar(
    'DatasetOrDataArray', xarray.Dataset, xarray.DataArray)


def assign_coords(
    x: DatasetOrDataArray,
    *,
    coords: Optional[Mapping[Hashable, Any]] = None,
    jax_coords: Optional[Mapping[Hashable, Any]] = None,
    ) -> DatasetOrDataArray:
  """Replacement for assign_coords which works in presence of jax_coords.

  `jax_coords` allow certain specified coordinates to have their data passed as
  JAX arrays (including through jax.jit boundaries). The compromise in return is
  that they are not created as index coordinates and cannot be used for .sel
  and other coordinate-based indexing operations. See docs for `jax_coords` on
  xarray_jax.Dataset and xarray_jax.DataArray for more information.

  This function can be used to set jax_coords on an existing DataArray or
  Dataset, and also to set a mix of jax and non-jax coordinates. It implements
  some workarounds to prevent xarray trying and failing to create IndexVariables
  from jax arrays under the hood.

  If you have any jax_coords with the same name as a dimension, you'll need to
  use this function instead of data_array.assign_coords or dataset.assign_coords
  in general, to avoid an xarray bug where it tries (and in our case fails) to
  create indexes for existing jax coords. See
  https://github.com/pydata/xarray/issues/7885.

  Args:
    x: An xarray Dataset or DataArray.
    coords: Dict of (non-JAX) coords, or None if not assigning any.
    jax_coords: Dict of JAX coords, or None if not assigning any. See docs for
      xarray_jax.Dataset / DataArray for more information on jax_coords.

  Returns:
    The Dataset or DataArray with coordinates assigned, similarly to
    Dataset.assign_coords / DataArray.assign_coords.
  """
  coords = {} if coords is None else dict(coords)  # Copy before mutating.
  jax_coords = {} if jax_coords is None else dict(jax_coords)

  # Any existing JAX coords must be dropped and re-added via the workaround
  # below, since otherwise .assign_coords will trigger an xarray bug where
  # it tries to recreate the indexes again for the existing coordinates.
  # Can remove if/when https://github.com/pydata/xarray/issues/7885 fixed.
  existing_jax_coords = get_jax_coords(x)
  jax_coords = existing_jax_coords | jax_coords
  x = x.drop_vars(existing_jax_coords.keys())

  # We need to ensure that xarray doesn't try to create an index for
  # coordinates with the same name as a dimension, since this will fail if
  # given a wrapped JAX tracer.
  # It appears the only way to avoid this is to name them differently to any
  # dimension name, then rename them back afterwards.
  renamed_jax_coords = {}
  for name, coord in jax_coords.items():
    if isinstance(coord, xarray.DataArray):
      coord = coord.variable
    if isinstance(coord, xarray.Variable):
      coord = coord.copy(deep=False)  # Copy before mutating attrs.
    else:
      # Must wrap as Variable with the correct dims first if this has not
      # already been done, otherwise xarray.Dataset will assume the dimension
      # name is also __NONINDEX_{n}.
      coord = Variable((name,), coord)

    # We set an attr on each jax_coord identifying it as such. These attrs on
    # the coord Variable gets reflected on the coord DataArray exposed too, and
    # when set on coordinates they generally get preserved under the default
    # keep_attrs setting.
    # These attrs are used by jax.tree_util registered flatten/unflatten to
    # determine which coords need to be treated as leaves of the flattened
    # structure vs static data.
    coord.attrs[_JAX_COORD_ATTR_NAME] = True
    renamed_jax_coords[f'__NONINDEX_{name}'] = coord

  x = x.assign_coords(coords=coords | renamed_jax_coords)

  rename_back_mapping = {f'__NONINDEX_{name}': name for name in jax_coords}
  if isinstance(x, xarray.Dataset):
    # Using 'rename' doesn't work if renaming to the same name as a dimension.
    return x.rename_vars(rename_back_mapping)
  else:  # DataArray
    return x.rename(rename_back_mapping)


def get_jax_coords(x: DatasetOrDataArray) -> Mapping[Hashable, Any]:
  return {
      name: coord_var
      for name, coord_var in x.coords.variables.items()
      if coord_var.attrs.get(_JAX_COORD_ATTR_NAME, False)}


def assign_jax_coords(
    x: DatasetOrDataArray,
    jax_coords: Optional[Mapping[Hashable, Any]] = None,
    **jax_coords_kwargs
    ) -> DatasetOrDataArray:
  """Assigns only jax_coords, with same API as xarray's assign_coords."""
  return assign_coords(x, jax_coords=jax_coords or jax_coords_kwargs)


def wrap(value):
  """Wraps JAX arrays for use in xarray, passing through other values."""
  if isinstance(value, jax.Array):
    return JaxArrayWrapper(value)
  else:
    return value


def unwrap(value, require_jax=False):
  """Unwraps wrapped JAX arrays used in xarray, passing through other values."""
  if isinstance(value, JaxArrayWrapper):
    return value.jax_array
  elif isinstance(value, jax.Array):
    return value
  elif require_jax:
    raise TypeError(f'Expected JAX array, found {type(value)}.')
  else:
    return value


def _wrapped(func):
  """Surrounds a function with JAX array unwrapping/wrapping."""
  def wrapped_func(*args, **kwargs):
    args, kwargs = tree.map_structure(unwrap, (args, kwargs))
    result = func(*args, **kwargs)
    return tree.map_structure(wrap, result)
  return wrapped_func


def unwrap_data(
    value: Union[xarray.Variable, xarray.DataArray],
    require_jax: bool = False
    ) -> Union[jax.Array, np.ndarray]:
  """The unwrapped (see unwrap) data of a an xarray.Variable or DataArray."""
  return unwrap(value.data, require_jax=require_jax)


def unwrap_vars(
    dataset: Mapping[Hashable, xarray.DataArray],
    require_jax: bool = False
    ) -> Mapping[str, Union[jax.Array, np.ndarray]]:
  """The unwrapped data (see unwrap) of the variables in a dataset."""
  # xarray types variable names as Hashable, but in practice they're invariably
  # strings and we convert to str to allow for a more useful return type.
  return {str(name): unwrap_data(var, require_jax=require_jax)
          for name, var in dataset.items()}


def unwrap_coords(
    dataset: Union[xarray.Dataset, xarray.DataArray],
    require_jax: bool = False
    ) -> Mapping[str, Union[jax.Array, np.ndarray]]:
  """The unwrapped data (see unwrap) of the coords in a Dataset or DataArray."""
  return {str(name): unwrap_data(var, require_jax=require_jax)
          for name, var in dataset.coords.items()}


def jax_data(value: Union[xarray.Variable, xarray.DataArray]) -> jax.Array:
  """Like unwrap_data, but will complain if not a jax array."""
  # Implementing this separately so we can give a more specific return type
  # for it.
  return cast(jax.Array, unwrap_data(value, require_jax=True))


def jax_vars(
    dataset: Mapping[Hashable, xarray.DataArray]) -> Mapping[str, jax.Array]:
  """Like unwrap_vars, but will complain if vars are not all jax arrays."""
  return cast(Mapping[str, jax.Array], unwrap_vars(dataset, require_jax=True))


class JaxArrayWrapper(np.lib.mixins.NDArrayOperatorsMixin):
  """Wraps a JAX array into a duck-typed array suitable for use with xarray.

  This uses an older duck-typed array protocol based on __array_ufunc__ and
  __array_function__ which works with numpy and xarray. (In newer versions
  of xarray it implements xarray.namedarray._typing._array_function.)

  This is in the process of being superseded by the Python array API standard
  (https://data-apis.org/array-api/latest/index.html), but JAX hasn't
  implemented it yet. Once they have, we should be able to get rid of
  this wrapper and use JAX arrays directly with xarray.

  """

  def __init__(self, jax_array):
    self.jax_array = jax_array

  def __array_ufunc__(self, ufunc, method, *args, **kwargs):
    for x in args:
      if not isinstance(x, (jax.typing.ArrayLike, type(self))):
        return NotImplemented
    if method != '__call__':
      return NotImplemented
    try:
      # Get the corresponding jax.numpy function to the NumPy ufunc:
      func = getattr(jnp, ufunc.__name__)
    except AttributeError:
      return NotImplemented
    # There may be an 'out' kwarg requesting an in-place operation, e.g. when
    # this is called via __iadd__ (+=), __imul__ (*=) etc. JAX doesn't support
    # in-place operations so we just remove this argument and have the ufunc
    # return a fresh JAX array instead.
    kwargs.pop('out', None)
    return _wrapped(func)(*args, **kwargs)

  def __array_function__(self, func, types, args, kwargs):
    try:
      # Get the corresponding jax.np function to the NumPy function:
      func = getattr(jnp, func.__name__)
    except AttributeError:
      return NotImplemented
    return _wrapped(func)(*args, **kwargs)

  def __repr__(self):
    return f'xarray_jax.JaxArrayWrapper({repr(self.jax_array)})'

  # NDArrayOperatorsMixin already proxies most __dunder__ operator methods.
  # We need to proxy through a few more methods in a similar way:

  # Essential array properties:

  @property
  def shape(self):
    return self.jax_array.shape

  @property
  def dtype(self):
    return self.jax_array.dtype

  @property
  def ndim(self):
    return self.jax_array.ndim

  @property
  def size(self):
    return self.jax_array.size

  @property
  def real(self):
    return self.jax_array.real

  @property
  def imag(self):
    return self.jax_array.imag

  # Array methods not covered by NDArrayOperatorsMixin:

  # Allows conversion to numpy array using np.asarray etc. Warning: doing this
  # will fail in a jax.jit-ed function.
  def __array__(self, dtype=None, context=None):
    return np.asarray(self.jax_array, dtype=dtype)

  __getitem__ = _wrapped(lambda array, *args: array.__getitem__(*args))
  # We drop the kwargs on this as they are not supported by JAX, but xarray
  # uses at least one of them (the copy arg).
  astype = _wrapped(lambda array, *args, **kwargs: array.astype(*args))

  # There are many more methods which are more canonically available via (j)np
  # functions, e.g. .sum() available via jnp.sum, and also mean, max, min,
  # argmax, argmin etc. We don't attempt to proxy through all of these as
  # methods, since this doesn't appear to be expected from a duck-typed array
  # implementation. But there are a few which xarray calls as methods, so we
  # proxy those:
  transpose = _wrapped(jnp.transpose)
  reshape = _wrapped(jnp.reshape)
  all = _wrapped(jnp.all)


def apply_ufunc(func, *args, require_jax=False, **apply_ufunc_kwargs):
  """Like xarray.apply_ufunc but for jax-specific ufuncs.

  Many numpy ufuncs will work fine out of the box with xarray_jax and
  JaxArrayWrapper, since JaxArrayWrapper quacks (mostly) like a numpy array and
  will convert many numpy operations to jax ops under the hood. For these
  situations, xarray.apply_ufunc should work fine.

  But sometimes you need a jax-specific ufunc which needs to be given a
  jax array as input or return a jax array as output. In that case you should
  use this helper as it will remove any JaxArrayWrapper before calling the func,
  and wrap the result afterwards before handing it back to xarray.

  Args:
    func: A function that works with jax arrays (e.g. using functions from
      jax.numpy) but otherwise meets the spec for the func argument to
      xarray.apply_ufunc.
    *args: xarray arguments to be mapped to arguments for func
      (see xarray.apply_ufunc).
    require_jax: Whether to require that inputs are based on jax arrays or allow
      those based on plain numpy arrays too.
    **apply_ufunc_kwargs: See xarray.apply_ufunc.

  Returns:
    Corresponding xarray results (see xarray.apply_ufunc).
  """
  def wrapped_func(*maybe_wrapped_args):
    unwrapped_args = [unwrap(a, require_jax) for a in maybe_wrapped_args]
    result = func(*unwrapped_args)
    # Result can be an array or a tuple of arrays, this handles both:
    return jax.tree_util.tree_map(wrap, result)
  return xarray.apply_ufunc(wrapped_func, *args, **apply_ufunc_kwargs)


def pmap(fn: Callable[..., Any],
         dim: str,
         axis_name: Optional[str] = None,
         devices: ... = None,
         backend: ... = None) -> Callable[..., Any]:
  """Wraps a subset of jax.pmap functionality to handle xarray input/output.

  Constraints:
    * Any Dataset or DataArray passed to the function must have `dim` as the
      first dimension. This will be checked. You can ensure this if necessary
      by calling `.transpose(dim, ...)` beforehand.
    * All args and return values will be mapped over the first dimension,
      it will use in_axes=0, out_axes=0.
    * No support for static_broadcasted_argnums, donate_argnums etc.

  Args:
    fn: Function to be pmap'd which takes and returns trees which may contain
      xarray Dataset/DataArray. Any Dataset/DataArrays passed as input must use
      `dim` as the first dimension on all arrays.
    dim: The xarray dimension name corresponding to the first dimension that is
      pmapped over (pmap is called with in_axes=0, out_axes=0).
    axis_name: Used by jax to identify the mapped axis so that parallel
      collectives can be applied. Defaults to same as `dim`.
    devices:
    backend:
      See jax.pmap.

  Returns:
    A pmap'd version of `fn`, which takes and returns Dataset/DataArray with an
    extra leading dimension `dim` relative to what the original `fn` sees.
  """
  input_treedef = None
  output_treedef = None

  def fn_passed_to_pmap(*flat_args):
    assert input_treedef is not None
    # Inside the pmap the original first dimension will no longer be present:
    def check_and_remove_leading_dim(dims):
      try:
        index = dims.index(dim)
      except ValueError:
        index = None
      if index != 0:
        raise ValueError(f'Expected dim {dim} at index 0, found at {index}.')
      return dims[1:]
    with dims_change_on_unflatten(check_and_remove_leading_dim):
      args = jax.tree_util.tree_unflatten(input_treedef, flat_args)
    result = fn(*args)
    nonlocal output_treedef
    flat_result, output_treedef = jax.tree_util.tree_flatten(result)
    return flat_result

  pmapped_fn = jax.pmap(
      fn_passed_to_pmap,
      axis_name=axis_name or dim,
      in_axes=0,
      out_axes=0,
      devices=devices,
      backend=backend)

  def result_fn(*args):
    nonlocal input_treedef
    flat_args, input_treedef = jax.tree_util.tree_flatten(args)
    flat_result = pmapped_fn(*flat_args)
    assert output_treedef is not None
    # After the pmap an extra leading axis will be present, we need to add an
    # xarray dimension for this when unflattening the result:
    with dims_change_on_unflatten(lambda dims: (dim,) + dims):
      return jax.tree_util.tree_unflatten(output_treedef, flat_result)

  return result_fn


# Register xarray datatypes with jax.tree_util.


DimsChangeFn = Callable[[Tuple[Hashable, ...]], Tuple[Hashable, ...]]
_DIMS_CHANGE_ON_UNFLATTEN_FN: contextvars.ContextVar[DimsChangeFn] = (
    contextvars.ContextVar('dims_change_on_unflatten_fn'))


@contextlib.contextmanager
def dims_change_on_unflatten(dims_change_fn: DimsChangeFn):
  """Can be used to change the dims used when unflattening arrays into xarrays.

  This is useful when some axes were added to / removed from the underlying jax
  arrays after they were flattened using jax.tree_util.tree_flatten, and you
  want to unflatten them again afterwards using the original treedef but
  adjusted for the added/removed dimensions.

  It can also be used with jax.tree_util.tree_map, when it's called with a
  function that adds/removes axes or otherwise changes the axis order.

  When dimensions are removed, any coordinates using those removed dimensions
  will also be removed on unflatten.

  This is implemented as a context manager that sets some thread-local state
  affecting the behaviour of our unflatten functions, because it's not possible
  to directly modify the treedef to change the dims/coords in it (and with
  tree_map, the treedef isn't exposed to you anyway).

  Args:
    dims_change_fn: Maps a tuple of dimension names for the original
      Variable/DataArray/Dataset that was flattened, to an updated tuple of
      dimensions which should be used when unflattening.

  Yields:
    To a context manager in whose scope jax.tree_util.tree_unflatten and
    jax.tree_util.tree_map will apply the dims_change_fn before reconstructing
    xarrays from jax arrays.
  """
  token = _DIMS_CHANGE_ON_UNFLATTEN_FN.set(dims_change_fn)
  try:
    yield
  finally:
    _DIMS_CHANGE_ON_UNFLATTEN_FN.reset(token)


def _flatten_variable(v: xarray.Variable) -> Tuple[
    Tuple[jax.typing.ArrayLike], Tuple[Hashable, ...]]:
  """Flattens a Variable for jax.tree_util."""
  children = (unwrap_data(v),)
  aux = v.dims
  return children, aux


def _unflatten_variable(
    aux: Tuple[Hashable, ...],
    children: Tuple[jax.typing.ArrayLike]) -> xarray.Variable:
  """Unflattens a Variable for jax.tree_util."""
  dims = aux
  dims_change_fn = _DIMS_CHANGE_ON_UNFLATTEN_FN.get(None)
  if dims_change_fn: dims = dims_change_fn(dims)
  return Variable(dims=dims, data=children[0])


def _split_static_and_jax_coords(
    coords: xarray.core.coordinates.Coordinates) -> Tuple[
        Mapping[Hashable, xarray.Variable], Mapping[Hashable, xarray.Variable]]:
  static_coord_vars = {}
  jax_coord_vars = {}
  for name, coord in coords.items():
    if coord.attrs.get(_JAX_COORD_ATTR_NAME, False):
      jax_coord_vars[name] = coord.variable
    else:
      assert not isinstance(coord, (jax.Array, JaxArrayWrapper))
      static_coord_vars[name] = coord.variable
  return static_coord_vars, jax_coord_vars


def _drop_with_none_of_dims(
    coord_vars: Mapping[Hashable, xarray.Variable],
    dims: Tuple[Hashable]) -> Mapping[Hashable, xarray.Variable]:
  return {name: var for name, var in coord_vars.items()
          if set(var.dims) <= set(dims)}


class _HashableCoords(collections.abc.Mapping):
  """Wraps a dict of xarray Variables as hashable, used for static coordinates.

  This needs to be hashable so that when an xarray.Dataset is passed to a
  jax.jit'ed function, jax can check whether it's seen an array with the
  same static coordinates(*) before or whether it needs to recompile the
  function for the new values of the static coordinates.

  (*) note jax_coords are not included in this; their value can be different
  on different calls without triggering a recompile.
  """

  def __init__(self, coord_vars: Mapping[Hashable, xarray.Variable]):
    self._variables = coord_vars

  def __repr__(self) -> str:
    return f'_HashableCoords({repr(self._variables)})'

  def __getitem__(self, key: Hashable) -> xarray.Variable:
    return self._variables[key]

  def __len__(self) -> int:
    return len(self._variables)

  def __iter__(self) -> Iterator[Hashable]:
    return iter(self._variables)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash(frozenset((name, var.data.tobytes())
                                  for name, var in self._variables.items()))
    return self._hash

  def __eq__(self, other):
    if self is other:
      return True
    elif not isinstance(other, type(self)):
      return NotImplemented
    elif self._variables is other._variables:
      return True
    else:
      return self._variables.keys() == other._variables.keys() and all(
          variable.equals(other._variables[name])
          for name, variable in self._variables.items())


def _flatten_data_array(v: xarray.DataArray) -> Tuple[
    # Children (data variable, jax_coord_vars):
    Tuple[xarray.Variable, Mapping[Hashable, xarray.Variable]],
    # Static auxiliary data (name, static_coord_vars):
    Tuple[Optional[Hashable], _HashableCoords]]:
  """Flattens a DataArray for jax.tree_util."""
  static_coord_vars, jax_coord_vars = _split_static_and_jax_coords(v.coords)
  children = (v.variable, jax_coord_vars)
  aux = (v.name, _HashableCoords(static_coord_vars))
  return children, aux


def _unflatten_data_array(
    aux: Tuple[Optional[Hashable], _HashableCoords],
    children: Tuple[xarray.Variable, Mapping[Hashable, xarray.Variable]],
) -> xarray.DataArray:
  """Unflattens a DataArray for jax.tree_util."""
  variable, jax_coord_vars = children
  name, static_coord_vars = aux
  # Drop static coords which have dims not present in any of the data_vars.
  # These would generally be dims that were dropped by a dims_change_fn, but
  # because static coordinates don't go through dims_change_fn on unflatten, we
  # just drop them where this causes a problem.
  # Since jax_coords go through the dims_change_fn on unflatten we don't need
  # to do this for jax_coords.
  static_coord_vars = _drop_with_none_of_dims(static_coord_vars, variable.dims)
  return DataArray(
      variable, name=name, coords=static_coord_vars, jax_coords=jax_coord_vars)


def _flatten_dataset(dataset: xarray.Dataset) -> Tuple[
    # Children (data variables, jax_coord_vars):
    Tuple[Mapping[Hashable, xarray.Variable],
          Mapping[Hashable, xarray.Variable]],
    # Static auxiliary data (static_coord_vars):
    _HashableCoords]:
  """Flattens a Dataset for jax.tree_util."""
  variables = {name: data_array.variable
               for name, data_array in dataset.data_vars.items()}
  static_coord_vars, jax_coord_vars = _split_static_and_jax_coords(
      dataset.coords)
  children = (variables, jax_coord_vars)
  aux = _HashableCoords(static_coord_vars)
  return children, aux


def _unflatten_dataset(
    aux: _HashableCoords,
    children: Tuple[Mapping[Hashable, xarray.Variable],
                    Mapping[Hashable, xarray.Variable]],
    ) -> xarray.Dataset:
  """Unflattens a Dataset for jax.tree_util."""
  data_vars, jax_coord_vars = children
  static_coord_vars = aux
  dataset = xarray.Dataset(data_vars)
  # Drop static coords which have dims not present in any of the data_vars.
  # See corresponding comment in _unflatten_data_array.
  static_coord_vars = _drop_with_none_of_dims(static_coord_vars, dataset.dims)  # pytype: disable=wrong-arg-types
  return assign_coords(
      dataset, coords=static_coord_vars, jax_coords=jax_coord_vars)


jax.tree_util.register_pytree_node(
    xarray.Variable, _flatten_variable, _unflatten_variable)
# This is a subclass of Variable but still needs registering separately.
# Flatten/unflatten for IndexVariable is a bit of a corner case but we do
# need to support it.
jax.tree_util.register_pytree_node(
    xarray.IndexVariable, _flatten_variable, _unflatten_variable)
jax.tree_util.register_pytree_node(
    xarray.DataArray, _flatten_data_array, _unflatten_data_array)
jax.tree_util.register_pytree_node(
    xarray.Dataset, _flatten_dataset, _unflatten_dataset)
