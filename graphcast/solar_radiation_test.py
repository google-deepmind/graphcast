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
import timeit
from typing import Sequence

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from graphcast import solar_radiation
import numpy as np
import pandas as pd
import xarray as xa


def _get_grid_lat_lon_coords(
    num_lat: int, num_lon: int
) -> tuple[np.ndarray, np.ndarray]:
  """Generates a linear latitude-longitude grid of the given size.

  Args:
    num_lat: Size of the latitude dimension of the grid.
    num_lon: Size of the longitude dimension of the grid.

  Returns:
    A tuple `(lat, lon)` containing 1D arrays with the latitude and longitude
    coordinates in degrees of the generated grid.
  """
  lat = np.linspace(-90.0, 90.0, num=num_lat, endpoint=True)
  lon = np.linspace(0.0, 360.0, num=num_lon, endpoint=False)
  return lat, lon


class SolarRadiationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)

  def test_missing_dim_raises_value_error(self):
    data = xa.DataArray(
        np.random.randn(2, 2),
        coords=[np.array([0.1, 0.2]), np.array([0.0, 0.5])],
        dims=["lon", "x"],
    )
    with self.assertRaisesRegex(
        ValueError, r".* dimensions are missing in `data_array_like`."
    ):
      solar_radiation.get_toa_incident_solar_radiation_for_xarray(
          data, integration_period="1h", num_integration_bins=360
      )

  def test_missing_coordinate_raises_value_error(self):
    data = xa.Dataset(
        data_vars={"var1": (["x", "lat", "lon"], np.random.randn(2, 3, 2))},
        coords={
            "lat": np.array([0.0, 0.1, 0.2]),
            "lon": np.array([0.0, 0.5]),
        },
    )
    with self.assertRaisesRegex(
        ValueError, r".* coordinates are missing in `data_array_like`."
    ):
      solar_radiation.get_toa_incident_solar_radiation_for_xarray(
          data, integration_period="1h", num_integration_bins=360
      )

  def test_shape_multiple_timestamps(self):
    data = xa.Dataset(
        data_vars={"var1": (["time", "lat", "lon"], np.random.randn(2, 4, 2))},
        coords={
            "lat": np.array([0.0, 0.1, 0.2, 0.3]),
            "lon": np.array([0.0, 0.5]),
            "time": np.array([100, 200], dtype="timedelta64[s]"),
            "datetime": xa.Variable(
                "time", np.array([10, 20], dtype="datetime64[D]")
            ),
        },
    )

    actual = solar_radiation.get_toa_incident_solar_radiation_for_xarray(
        data, integration_period="1h", num_integration_bins=2
    )

    self.assertEqual(("time", "lat", "lon"), actual.dims)
    self.assertEqual((2, 4, 2), actual.shape)

  def test_shape_single_timestamp(self):
    data = xa.Dataset(
        data_vars={"var1": (["lat", "lon"], np.random.randn(4, 2))},
        coords={
            "lat": np.array([0.0, 0.1, 0.2, 0.3]),
            "lon": np.array([0.0, 0.5]),
            "datetime": np.datetime64(10, "D"),
        },
    )

    actual = solar_radiation.get_toa_incident_solar_radiation_for_xarray(
        data, integration_period="1h", num_integration_bins=2
    )

    self.assertEqual(("lat", "lon"), actual.dims)
    self.assertEqual((4, 2), actual.shape)

  @parameterized.named_parameters(
      dict(
          testcase_name="one_timestamp_jitted",
          periods=1,
          repeats=3,
          use_jit=True,
      ),
      dict(
          testcase_name="one_timestamp_non_jitted",
          periods=1,
          repeats=3,
          use_jit=False,
      ),
      dict(
          testcase_name="ten_timestamps_non_jitted",
          periods=10,
          repeats=1,
          use_jit=False,
      ),
  )
  def test_full_spatial_resolution(
      self, periods: int, repeats: int, use_jit: bool
  ):
    timestamps = pd.date_range(start="2023-09-25", periods=periods, freq="6h")
    # Generate a linear grid with 0.25 degrees resolution similar to ERA5.
    lat, lon = _get_grid_lat_lon_coords(num_lat=721, num_lon=1440)

    def benchmark() -> None:
      solar_radiation.get_toa_incident_solar_radiation(
          timestamps,
          lat,
          lon,
          integration_period="1h",
          num_integration_bins=360,
          use_jit=use_jit,
      ).block_until_ready()

    results = timeit.repeat(benchmark, repeat=repeats, number=1)

    logging.info(
        "Times to compute `tisr` for input of shape `%d, %d, %d` (seconds): %s",
        len(timestamps),
        len(lat),
        len(lon),
        np.array2string(np.array(results), precision=1),
    )


class GetTsiTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="reference_tsi_data",
          loader=solar_radiation.reference_tsi_data,
          expected_tsi=np.array([1361.0]),
      ),
      dict(
          testcase_name="era5_tsi_data",
          loader=solar_radiation.era5_tsi_data,
          expected_tsi=np.array([1360.9440]),  # 0.9965 * 1365.7240
      ),
  )
  def test_mid_2020_lookup(
      self, loader: solar_radiation.TsiDataLoader, expected_tsi: np.ndarray
  ):
    tsi_data = loader()

    tsi = solar_radiation.get_tsi(
        [np.datetime64("2020-07-02T00:00:00")], tsi_data
    )

    np.testing.assert_allclose(expected_tsi, tsi)

  @parameterized.named_parameters(
      dict(
          testcase_name="beginning_2020_left_boundary",
          timestamps=[np.datetime64("2020-01-01T00:00:00")],
          expected_tsi=np.array([1000.0]),
      ),
      dict(
          testcase_name="mid_2020_exact",
          timestamps=[np.datetime64("2020-07-02T00:00:00")],
          expected_tsi=np.array([1000.0]),
      ),
      dict(
          testcase_name="beginning_2021_interpolated",
          timestamps=[np.datetime64("2021-01-01T00:00:00")],
          expected_tsi=np.array([1150.0]),
      ),
      dict(
          testcase_name="mid_2021_lookup",
          timestamps=[np.datetime64("2021-07-02T12:00:00")],
          expected_tsi=np.array([1300.0]),
      ),
      dict(
          testcase_name="beginning_2022_interpolated",
          timestamps=[np.datetime64("2022-01-01T00:00:00")],
          expected_tsi=np.array([1250.0]),
      ),
      dict(
          testcase_name="mid_2022_lookup",
          timestamps=[np.datetime64("2022-07-02T12:00:00")],
          expected_tsi=np.array([1200.0]),
      ),
      dict(
          testcase_name="beginning_2023_right_boundary",
          timestamps=[np.datetime64("2023-01-01T00:00:00")],
          expected_tsi=np.array([1200.0]),
      ),
  )
  def test_interpolation(
      self, timestamps: Sequence[np.datetime64], expected_tsi: np.ndarray
  ):
    tsi_data = xa.DataArray(
        np.array([1000.0, 1300.0, 1200.0]),
        dims=["time"],
        coords={"time": np.array([2020.5, 2021.5, 2022.5])},
    )

    tsi = solar_radiation.get_tsi(timestamps, tsi_data)

    np.testing.assert_allclose(expected_tsi, tsi)


if __name__ == "__main__":
  absltest.main()
