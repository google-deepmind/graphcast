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
"""Tests for xarray_jax."""

from absl.testing import absltest
import chex
from graphcast import xarray_jax
import jax
import jax.numpy as jnp
import numpy as np
import xarray


class XarrayJaxTest(absltest.TestCase):

  def test_jax_array_wrapper_with_numpy_api(self):
    # This is just a side benefit of making things work with xarray, but the
    # JaxArrayWrapper does allow you to manipulate JAX arrays using the
    # standard numpy API, without converting them to numpy in the process:
    ones = jnp.ones((3, 4), dtype=np.float32)
    x = xarray_jax.JaxArrayWrapper(ones)
    x = np.abs((x + 2) * (x - 3))
    x = x[:-1, 1:3]
    x = np.concatenate([x, x + 1], axis=0)
    x = np.transpose(x, (1, 0))
    x = np.reshape(x, (-1,))
    x = x.astype(np.int32)
    self.assertIsInstance(x, xarray_jax.JaxArrayWrapper)
    # An explicit conversion gets us out of JAX-land however:
    self.assertIsInstance(np.asarray(x), np.ndarray)

  def test_jax_xarray_variable(self):
    def ops_via_xarray(inputs):
      x = xarray_jax.Variable(('lat', 'lon'), inputs)
      # We'll apply a sequence of operations just to test that the end result is
      # still a JAX array, i.e. we haven't converted to numpy at any point.
      x = np.abs((x + 2) * (x - 3))
      x = x.isel({'lat': slice(0, -1), 'lon': slice(1, 3)})
      x = xarray.Variable.concat([x, x + 1], dim='lat')
      x = x.transpose('lon', 'lat')
      x = x.stack(channels=('lon', 'lat'))
      x = x.sum()
      return xarray_jax.jax_data(x)

    # Check it doesn't leave jax-land when passed concrete values:
    ones = jnp.ones((3, 4), dtype=np.float32)
    result = ops_via_xarray(ones)
    self.assertIsInstance(result, jax.Array)

    # And that you can JIT it and compute gradients through it. These will
    # involve passing jax tracers through the xarray computation:
    jax.jit(ops_via_xarray)(ones)
    jax.grad(ops_via_xarray)(ones)

  def test_jax_xarray_data_array(self):
    def ops_via_xarray(inputs):
      x = xarray_jax.DataArray(dims=('lat', 'lon'),
                               data=inputs,
                               coords={'lat': np.arange(3) * 10,
                                       'lon': np.arange(4) * 10})
      x = np.abs((x + 2) * (x - 3))
      x = x.sel({'lat': slice(0, 20)})
      y = xarray_jax.DataArray(dims=('lat', 'lon'),
                               data=ones,
                               coords={'lat': np.arange(3, 6) * 10,
                                       'lon': np.arange(4) * 10})
      x = xarray.concat([x, y], dim='lat')
      x = x.transpose('lon', 'lat')
      x = x.stack(channels=('lon', 'lat'))
      x = x.unstack()
      x = x.sum()
      return xarray_jax.jax_data(x)

    ones = jnp.ones((3, 4), dtype=np.float32)
    result = ops_via_xarray(ones)
    self.assertIsInstance(result, jax.Array)

    jax.jit(ops_via_xarray)(ones)
    jax.grad(ops_via_xarray)(ones)

  def test_jax_xarray_dataset(self):
    def ops_via_xarray(foo, bar):
      x = xarray_jax.Dataset(
          data_vars={'foo': (('lat', 'lon'), foo),
                     'bar': (('time', 'lat', 'lon'), bar)},
          coords={
              'time': np.arange(2),
              'lat': np.arange(3) * 10,
              'lon': np.arange(4) * 10})
      x = np.abs((x + 2) * (x - 3))
      x = x.sel({'lat': slice(0, 20)})
      y = xarray_jax.Dataset(
          data_vars={'foo': (('lat', 'lon'), foo),
                     'bar': (('time', 'lat', 'lon'), bar)},
          coords={
              'time': np.arange(2),
              'lat': np.arange(3, 6) * 10,
              'lon': np.arange(4) * 10})
      x = xarray.concat([x, y], dim='lat')
      x = x.transpose('lon', 'lat', 'time')
      x = x.stack(channels=('lon', 'lat'))
      x = (x.foo + x.bar).sum()
      return xarray_jax.jax_data(x)

    foo = jnp.ones((3, 4), dtype=np.float32)
    bar = jnp.ones((2, 3, 4), dtype=np.float32)
    result = ops_via_xarray(foo, bar)
    self.assertIsInstance(result, jax.Array)

    jax.jit(ops_via_xarray)(foo, bar)
    jax.grad(ops_via_xarray)(foo, bar)

  def test_jit_function_with_xarray_variable_arguments_and_return(self):
    function = jax.jit(lambda v: v + 1)
    with self.subTest('jax input'):
      inputs = xarray_jax.Variable(
          ('lat', 'lon'), jnp.ones((3, 4), dtype=np.float32))
      _ = function(inputs)
      # We test running the jitted function a second time, to exercise logic in
      # jax which checks if the structure of the inputs (including dimension
      # names and coordinates) is the same as it was for the previous call and
      # so whether it needs to re-trace-and-compile a new version of the
      # function or not. This can run into problems if the 'aux' structure
      # returned by the registered flatten function is not hashable/comparable.
      outputs = function(inputs)
      self.assertEqual(outputs.dims, inputs.dims)
    with self.subTest('numpy input'):
      inputs = xarray.Variable(
          ('lat', 'lon'), np.ones((3, 4), dtype=np.float32))
      _ = function(inputs)
      outputs = function(inputs)
      self.assertEqual(outputs.dims, inputs.dims)

  def test_jit_problem_if_convert_to_plain_numpy_array(self):
    inputs = xarray_jax.DataArray(
        data=jnp.ones((2,), dtype=np.float32), dims=('foo',))
    with self.assertRaises(jax.errors.TracerArrayConversionError):
      # Calling .values on a DataArray converts its values to numpy:
      jax.jit(lambda data_array: data_array.values)(inputs)

  def test_grad_function_with_xarray_variable_arguments(self):
    x = xarray_jax.Variable(('lat', 'lon'), jnp.ones((3, 4), dtype=np.float32))
    # For grad we still need a JAX scalar as the output:
    jax.grad(lambda v: xarray_jax.jax_data(v.sum()))(x)

  def test_jit_function_with_xarray_data_array_arguments_and_return(self):
    inputs = xarray_jax.DataArray(
        data=jnp.ones((3, 4), dtype=np.float32),
        dims=('lat', 'lon'),
        coords={'lat': np.arange(3),
                'lon': np.arange(4) * 10})
    fn = jax.jit(lambda v: v + 1)
    _ = fn(inputs)
    outputs = fn(inputs)
    self.assertEqual(outputs.dims, inputs.dims)
    chex.assert_trees_all_equal(outputs.coords, inputs.coords)

  def test_jit_function_with_data_array_and_jax_coords(self):
    inputs = xarray_jax.DataArray(
        data=jnp.ones((3, 4), dtype=np.float32),
        dims=('lat', 'lon'),
        coords={'lat': np.arange(3)},
        jax_coords={'lon': jnp.arange(4) * 10})
    # Verify the jax_coord 'lon' retains jax data, and has not been created
    # as an index coordinate:
    self.assertIsInstance(inputs.coords['lon'].data, xarray_jax.JaxArrayWrapper)
    self.assertNotIn('lon', inputs.indexes)

    @jax.jit
    def fn(v):
      # The non-JAX coord is passed with numpy array data and an index:
      self.assertIsInstance(v.coords['lat'].data, np.ndarray)
      self.assertIn('lat', v.indexes)

      # The jax_coord is passed with JAX array data:
      self.assertIsInstance(v.coords['lon'].data, xarray_jax.JaxArrayWrapper)
      self.assertNotIn('lon', v.indexes)

      # Use the jax coord in the computation:
      v = v + v.coords['lon']

      # Return with an updated jax coord:
      return xarray_jax.assign_jax_coords(v, lon=v.coords['lon'] + 1)

    _ = fn(inputs)
    outputs = fn(inputs)

    # Verify the jax_coord 'lon' has jax data in the output too:
    self.assertIsInstance(
        outputs.coords['lon'].data, xarray_jax.JaxArrayWrapper)
    self.assertNotIn('lon', outputs.indexes)

    self.assertEqual(outputs.dims, inputs.dims)
    chex.assert_trees_all_equal(outputs.coords['lat'], inputs.coords['lat'])
    # Check our computations with the coordinate values worked:
    chex.assert_trees_all_equal(
        outputs.coords['lon'].data, (inputs.coords['lon']+1).data)
    chex.assert_trees_all_equal(
        outputs.data, (inputs + inputs.coords['lon']).data)

  def test_jit_function_with_xarray_dataset_arguments_and_return(self):
    foo = jnp.ones((3, 4), dtype=np.float32)
    bar = jnp.ones((2, 3, 4), dtype=np.float32)
    inputs = xarray_jax.Dataset(
        data_vars={'foo': (('lat', 'lon'), foo),
                   'bar': (('time', 'lat', 'lon'), bar)},
        coords={
            'time': np.arange(2),
            'lat': np.arange(3) * 10,
            'lon': np.arange(4) * 10})
    fn = jax.jit(lambda v: v + 1)
    _ = fn(inputs)
    outputs = fn(inputs)
    self.assertEqual({'foo', 'bar'}, outputs.data_vars.keys())
    self.assertEqual(inputs.foo.dims, outputs.foo.dims)
    self.assertEqual(inputs.bar.dims, outputs.bar.dims)
    chex.assert_trees_all_equal(outputs.coords, inputs.coords)

  def test_jit_function_with_dataset_and_jax_coords(self):
    foo = jnp.ones((3, 4), dtype=np.float32)
    bar = jnp.ones((2, 3, 4), dtype=np.float32)
    inputs = xarray_jax.Dataset(
        data_vars={'foo': (('lat', 'lon'), foo),
                   'bar': (('time', 'lat', 'lon'), bar)},
        coords={
            'time': np.arange(2),
            'lat': np.arange(3) * 10,
        },
        jax_coords={'lon': jnp.arange(4) * 10}
    )
    # Verify the jax_coord 'lon' retains jax data, and has not been created
    # as an index coordinate:
    self.assertIsInstance(inputs.coords['lon'].data, xarray_jax.JaxArrayWrapper)
    self.assertNotIn('lon', inputs.indexes)

    @jax.jit
    def fn(v):
      # The non-JAX coords are passed with numpy array data and an index:
      self.assertIsInstance(v.coords['lat'].data, np.ndarray)
      self.assertIn('lat', v.indexes)

      # The jax_coord is passed with JAX array data:
      self.assertIsInstance(v.coords['lon'].data, xarray_jax.JaxArrayWrapper)
      self.assertNotIn('lon', v.indexes)

      # Use the jax coord in the computation:
      v = v + v.coords['lon']

      # Return with an updated jax coord:
      return xarray_jax.assign_jax_coords(v, lon=v.coords['lon'] + 1)

    _ = fn(inputs)
    outputs = fn(inputs)

    # Verify the jax_coord 'lon' has jax data in the output too:
    self.assertIsInstance(
        outputs.coords['lon'].data, xarray_jax.JaxArrayWrapper)
    self.assertNotIn('lon', outputs.indexes)

    self.assertEqual(outputs.dims, inputs.dims)
    chex.assert_trees_all_equal(outputs.coords['lat'], inputs.coords['lat'])
    # Check our computations with the coordinate values worked:
    chex.assert_trees_all_equal(
        (outputs.coords['lon']).data,
        (inputs.coords['lon']+1).data,
    )
    outputs_dict = {key: outputs[key].data for key in outputs}
    inputs_and_inputs_coords_dict = {
        key: (inputs + inputs.coords['lon'])[key].data
        for key in inputs + inputs.coords['lon']
    }
    chex.assert_trees_all_equal(outputs_dict, inputs_and_inputs_coords_dict)

  def test_flatten_unflatten_variable(self):
    variable = xarray_jax.Variable(
        ('lat', 'lon'), jnp.ones((3, 4), dtype=np.float32))
    children, aux = xarray_jax._flatten_variable(variable)
    # Check auxiliary info is hashable/comparable (important for jax.jit):
    hash(aux)
    self.assertEqual(aux, aux)
    roundtrip = xarray_jax._unflatten_variable(aux, children)
    self.assertTrue(variable.equals(roundtrip))

  def test_flatten_unflatten_data_array(self):
    data_array = xarray_jax.DataArray(
        data=jnp.ones((3, 4), dtype=np.float32),
        dims=('lat', 'lon'),
        coords={'lat': np.arange(3)},
        jax_coords={'lon': np.arange(4) * 10},
    )
    children, aux = xarray_jax._flatten_data_array(data_array)
    # Check auxiliary info is hashable/comparable (important for jax.jit):
    hash(aux)
    self.assertEqual(aux, aux)
    roundtrip = xarray_jax._unflatten_data_array(aux, children)
    self.assertTrue(data_array.equals(roundtrip))

  def test_flatten_unflatten_dataset(self):
    foo = jnp.ones((3, 4), dtype=np.float32)
    bar = jnp.ones((2, 3, 4), dtype=np.float32)
    dataset = xarray_jax.Dataset(
        data_vars={'foo': (('lat', 'lon'), foo),
                   'bar': (('time', 'lat', 'lon'), bar)},
        coords={
            'time': np.arange(2),
            'lat': np.arange(3) * 10},
        jax_coords={
            'lon': np.arange(4) * 10})
    children, aux = xarray_jax._flatten_dataset(dataset)
    # Check auxiliary info is hashable/comparable (important for jax.jit):
    hash(aux)
    self.assertEqual(aux, aux)
    roundtrip = xarray_jax._unflatten_dataset(aux, children)
    self.assertTrue(dataset.equals(roundtrip))

  def test_flatten_unflatten_added_dim(self):
    data_array = xarray_jax.DataArray(
        data=jnp.ones((3, 4), dtype=np.float32),
        dims=('lat', 'lon'),
        coords={'lat': np.arange(3),
                'lon': np.arange(4) * 10})
    leaves, treedef = jax.tree_util.tree_flatten(data_array)
    leaves = [jnp.expand_dims(x, 0) for x in leaves]
    with xarray_jax.dims_change_on_unflatten(lambda dims: ('new',) + dims):
      with_new_dim = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertEqual(('new', 'lat', 'lon'), with_new_dim.dims)
    xarray.testing.assert_identical(
        jax.device_get(data_array),
        jax.device_get(with_new_dim.isel(new=0)))

  def test_map_added_dim(self):
    data_array = xarray_jax.DataArray(
        data=jnp.ones((3, 4), dtype=np.float32),
        dims=('lat', 'lon'),
        coords={'lat': np.arange(3),
                'lon': np.arange(4) * 10})
    with xarray_jax.dims_change_on_unflatten(lambda dims: ('new',) + dims):
      with_new_dim = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0),
                                            data_array)
    self.assertEqual(('new', 'lat', 'lon'), with_new_dim.dims)
    xarray.testing.assert_identical(
        jax.device_get(data_array),
        jax.device_get(with_new_dim.isel(new=0)))

  def test_map_remove_dim(self):
    foo = jnp.ones((1, 3, 4), dtype=np.float32)
    bar = jnp.ones((1, 2, 3, 4), dtype=np.float32)
    dataset = xarray_jax.Dataset(
        data_vars={'foo': (('batch', 'lat', 'lon'), foo),
                   'bar': (('batch', 'time', 'lat', 'lon'), bar)},
        coords={
            'batch': np.array([123]),
            'time': np.arange(2),
            'lat': np.arange(3) * 10,
            'lon': np.arange(4) * 10})
    with xarray_jax.dims_change_on_unflatten(lambda dims: dims[1:]):
      with_removed_dim = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, 0),
                                                dataset)
    self.assertEqual(('lat', 'lon'), with_removed_dim['foo'].dims)
    self.assertEqual(('time', 'lat', 'lon'), with_removed_dim['bar'].dims)
    self.assertNotIn('batch', with_removed_dim.dims)
    self.assertNotIn('batch', with_removed_dim.coords)
    xarray.testing.assert_identical(
        jax.device_get(dataset.isel(batch=0, drop=True)),
        jax.device_get(with_removed_dim))

  def test_pmap(self):
    devices = jax.local_device_count()
    foo = jnp.zeros((devices, 3, 4), dtype=np.float32)
    bar = jnp.zeros((devices, 2, 3, 4), dtype=np.float32)
    dataset = xarray_jax.Dataset({
        'foo': (('device', 'lat', 'lon'), foo),
        'bar': (('device', 'time', 'lat', 'lon'), bar)})

    def func(d):
      self.assertNotIn('device', d.dims)
      return d + 1
    func = xarray_jax.pmap(func, dim='device')

    result = func(dataset)
    xarray.testing.assert_identical(
        jax.device_get(dataset + 1),
        jax.device_get(result))

    # Can call it again with a different argument structure (it will recompile
    # under the hood but should work):
    dataset = dataset.drop_vars('foo')
    result = func(dataset)
    xarray.testing.assert_identical(
        jax.device_get(dataset + 1),
        jax.device_get(result))

  def test_pmap_with_jax_coords(self):
    devices = jax.local_device_count()
    foo = jnp.zeros((devices, 3, 4), dtype=np.float32)
    bar = jnp.zeros((devices, 2, 3, 4), dtype=np.float32)
    time = jnp.zeros((devices, 2), dtype=np.float32)
    dataset = xarray_jax.Dataset(
        {'foo': (('device', 'lat', 'lon'), foo),
         'bar': (('device', 'time', 'lat', 'lon'), bar)},
        coords={
            'lat': np.arange(3),
            'lon': np.arange(4),
        },
        jax_coords={
            # Currently any jax_coords need a leading device dimension to use
            # with pmap, same as for data_vars.
            # TODO(matthjw): have pmap automatically broadcast to all devices
            # where the device dimension not present.
            'time': xarray_jax.Variable(('device', 'time'), time),
        }
    )

    def func(d):
      self.assertNotIn('device', d.dims)
      self.assertNotIn('device', d.coords['time'].dims)

      # The jax_coord 'time' should be passed in backed by a JAX array, but
      # not as an index coordinate.
      self.assertIsInstance(d.coords['time'].data, xarray_jax.JaxArrayWrapper)
      self.assertNotIn('time', d.indexes)

      return d + 1
    func = xarray_jax.pmap(func, dim='device')

    result = func(dataset)
    xarray.testing.assert_identical(
        jax.device_get(dataset + 1),
        jax.device_get(result))

    # Can call it again with a different argument structure (it will recompile
    # under the hood but should work):
    dataset = dataset.drop_vars('foo')
    result = func(dataset)
    xarray.testing.assert_identical(
        jax.device_get(dataset + 1),
        jax.device_get(result))

  def test_pmap_with_tree_mix_of_xarray_and_jax_array(self):
    devices = jax.local_device_count()
    data_array = xarray_jax.DataArray(
        data=jnp.ones((devices, 3, 4), dtype=np.float32),
        dims=('device', 'lat', 'lon'))
    plain_array = jnp.ones((devices, 2), dtype=np.float32)
    inputs = {'foo': data_array,
              'bar': plain_array}

    def func(x):
      return x['foo'] + 1, x['bar'] + 1

    func = xarray_jax.pmap(func, dim='device')
    result_foo, result_bar = func(inputs)
    xarray.testing.assert_identical(
        jax.device_get(inputs['foo'] + 1),
        jax.device_get(result_foo))
    np.testing.assert_array_equal(
        jax.device_get(inputs['bar'] + 1),
        jax.device_get(result_bar))

  def test_pmap_complains_when_dim_not_first(self):
    devices = jax.local_device_count()
    data_array = xarray_jax.DataArray(
        data=jnp.ones((3, devices, 4), dtype=np.float32),
        dims=('lat', 'device', 'lon'))

    func = xarray_jax.pmap(lambda x: x+1, dim='device')
    with self.assertRaisesRegex(
        ValueError, 'Expected dim device at index 0, found at 1'):
      func(data_array)

  def test_apply_ufunc(self):
    inputs = xarray_jax.DataArray(
        data=jnp.asarray([[1, 2], [3, 4]]),
        dims=('x', 'y'),
        coords={'x': [0, 1],
                'y': [2, 3]})
    result = xarray_jax.apply_ufunc(
        lambda x: jnp.sum(x, axis=-1),
        inputs,
        input_core_dims=[['x']])
    expected_result = xarray_jax.DataArray(
        data=[4, 6],
        dims=('y',),
        coords={'y': [2, 3]})
    xarray.testing.assert_identical(expected_result, jax.device_get(result))

  def test_apply_ufunc_multiple_return_values(self):
    def ufunc(array):
      return jnp.min(array, axis=-1), jnp.max(array, axis=-1)
    inputs = xarray_jax.DataArray(
        data=jnp.asarray([[1, 4], [3, 2]]),
        dims=('x', 'y'),
        coords={'x': [0, 1],
                'y': [2, 3]})
    result = xarray_jax.apply_ufunc(
        ufunc, inputs, input_core_dims=[['x']], output_core_dims=[[], []])
    expected = (
        # Mins:
        xarray_jax.DataArray(
            data=[1, 2],
            dims=('y',),
            coords={'y': [2, 3]}
        ),
        # Maxes:
        xarray_jax.DataArray(
            data=[3, 4],
            dims=('y',),
            coords={'y': [2, 3]}
        )
    )
    xarray.testing.assert_identical(expected[0], jax.device_get(result[0]))
    xarray.testing.assert_identical(expected[1], jax.device_get(result[1]))

if __name__ == '__main__':
  absltest.main()
