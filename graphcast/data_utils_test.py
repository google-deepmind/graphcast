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
"""Tests for `data_utils.py`."""

import datetime
from absl.testing import absltest
from absl.testing import parameterized
from graphcast import data_utils
import numpy as np
import xarray as xa


class DataUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Fix the seed for reproducibility.
    np.random.seed(0)

  def test_year_progress_is_zero_at_year_start_or_end(self):
    year_progress = data_utils.get_year_progress(
        np.array([
            0,
            data_utils.AVG_SEC_PER_YEAR,
            data_utils.AVG_SEC_PER_YEAR * 42,  # 42 years.
        ])
    )
    np.testing.assert_array_equal(year_progress, np.zeros(year_progress.shape))

  def test_year_progress_is_almost_one_before_year_ends(self):
    year_progress = data_utils.get_year_progress(
        np.array([
            data_utils.AVG_SEC_PER_YEAR - 1,
            (data_utils.AVG_SEC_PER_YEAR - 1) * 42,  # ~42 years
        ])
    )
    with self.subTest("Year progress values are close to 1"):
      self.assertTrue(np.all(year_progress > 0.999))
    with self.subTest("Year progress values != 1"):
      self.assertTrue(np.all(year_progress < 1.0))

  def test_day_progress_computes_for_all_times_and_longitudes(self):
    times = np.random.randint(low=0, high=1e10, size=10)
    longitudes = np.arange(0, 360.0, 1.0)
    day_progress = data_utils.get_day_progress(times, longitudes)
    with self.subTest("Day progress is computed for all times and longinutes"):
      self.assertSequenceEqual(
          day_progress.shape, (len(times), len(longitudes))
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="random_date_1",
          year=1988,
          month=11,
          day=7,
          hour=2,
          minute=45,
          second=34,
      ),
      dict(
          testcase_name="random_date_2",
          year=2022,
          month=3,
          day=12,
          hour=7,
          minute=1,
          second=0,
      ),
  )
  def test_day_progress_is_in_between_zero_and_one(
      self, year, month, day, hour, minute, second
  ):
    # Datetime from a timestamp.
    dt = datetime.datetime(year, month, day, hour, minute, second)
    # Epoch time.
    epoch_time = datetime.datetime(1970, 1, 1)
    # Seconds since epoch.
    seconds_since_epoch = np.array([(dt - epoch_time).total_seconds()])

    # Longitudes with 1 degree resolution.
    longitudes = np.arange(0, 360.0, 1.0)

    day_progress = data_utils.get_day_progress(seconds_since_epoch, longitudes)
    with self.subTest("Day progress >= 0"):
      self.assertTrue(np.all(day_progress >= 0.0))
    with self.subTest("Day progress < 1"):
      self.assertTrue(np.all(day_progress < 1.0))

  def test_day_progress_is_zero_at_day_start_or_end(self):
    day_progress = data_utils.get_day_progress(
        seconds_since_epoch=np.array([
            0,
            data_utils.SEC_PER_DAY,
            data_utils.SEC_PER_DAY * 42,  # 42 days.
        ]),
        longitude=np.array([0.0]),
    )
    np.testing.assert_array_equal(day_progress, np.zeros(day_progress.shape))

  def test_day_progress_specific_value(self):
    day_progress = data_utils.get_day_progress(
        seconds_since_epoch=np.array([123]),
        longitude=np.array([0.0]),
    )
    np.testing.assert_array_almost_equal(
        day_progress, np.array([[0.00142361]]), decimal=6
    )

  def test_featurize_progress_valid_values_and_dimensions(self):
    day_progress = np.array([0.0, 0.45, 0.213])
    feature_dimensions = ("time",)
    progress_features = data_utils.featurize_progress(
        name="day_progress", dims=feature_dimensions, progress=day_progress
    )
    for feature in progress_features.values():
      with self.subTest(f"Valid dimensions for {feature}"):
        self.assertSequenceEqual(feature.dims, feature_dimensions)

    with self.subTest("Valid values for day_progress"):
      np.testing.assert_array_equal(
          day_progress, progress_features["day_progress"].values
      )

    with self.subTest("Valid values for day_progress_sin"):
      np.testing.assert_array_almost_equal(
          np.array([0.0, 0.30901699, 0.97309851]),
          progress_features["day_progress_sin"].values,
          decimal=6,
      )

    with self.subTest("Valid values for day_progress_cos"):
      np.testing.assert_array_almost_equal(
          np.array([1.0, -0.95105652, 0.23038943]),
          progress_features["day_progress_cos"].values,
          decimal=6,
      )

  def test_featurize_progress_invalid_dimensions(self):
    year_progress = np.array([0.0, 0.45, 0.213])
    feature_dimensions = ("time", "longitude")
    with self.assertRaises(ValueError):
      data_utils.featurize_progress(
          name="year_progress", dims=feature_dimensions, progress=year_progress
      )

  def test_add_derived_vars_variables_added(self):
    data = xa.Dataset(
        data_vars={
            "var1": (["x", "lon", "datetime"], 8 * np.random.randn(2, 2, 3))
        },
        coords={
            "lon": np.array([0.0, 0.5]),
            "datetime": np.array([
                datetime.datetime(2021, 1, 1),
                datetime.datetime(2023, 1, 1),
                datetime.datetime(2023, 1, 3),
            ]),
        },
    )
    data_utils.add_derived_vars(data)
    all_variables = set(data.variables)

    with self.subTest("Original value was not removed"):
      self.assertIn("var1", all_variables)
    with self.subTest("Year progress feature was added"):
      self.assertIn(data_utils.YEAR_PROGRESS, all_variables)
    with self.subTest("Day progress feature was added"):
      self.assertIn(data_utils.DAY_PROGRESS, all_variables)

  def test_add_derived_vars_existing_vars_not_overridden(self):
    dims = ["x", "lon", "datetime"]
    data = xa.Dataset(
        data_vars={
            "var1": (dims, 8 * np.random.randn(2, 2, 3)),
            data_utils.YEAR_PROGRESS: (dims, np.full((2, 2, 3), 0.111)),
            data_utils.DAY_PROGRESS: (dims, np.full((2, 2, 3), 0.222)),
        },
        coords={
            "lon": np.array([0.0, 0.5]),
            "datetime": np.array([
                datetime.datetime(2021, 1, 1),
                datetime.datetime(2023, 1, 1),
                datetime.datetime(2023, 1, 3),
            ]),
        },
    )

    data_utils.add_derived_vars(data)

    with self.subTest("Year progress feature was not overridden"):
      np.testing.assert_allclose(data[data_utils.YEAR_PROGRESS], 0.111)
    with self.subTest("Day progress feature was not overridden"):
      np.testing.assert_allclose(data[data_utils.DAY_PROGRESS], 0.222)

  @parameterized.named_parameters(
      dict(testcase_name="missing_datetime", coord_name="lon"),
      dict(testcase_name="missing_lon", coord_name="datetime"),
  )
  def test_add_derived_vars_missing_coordinate_raises_value_error(
      self, coord_name
  ):
    with self.subTest(f"Missing {coord_name} coordinate"):
      data = xa.Dataset(
          data_vars={"var1": (["x", coord_name], 8 * np.random.randn(2, 2))},
          coords={
              coord_name: np.array([0.0, 0.5]),
          },
      )
      with self.assertRaises(ValueError):
        data_utils.add_derived_vars(data)

  def test_add_tisr_var_variable_added(self):
    data = xa.Dataset(
        data_vars={
            "var1": (["time", "lat", "lon"], np.full((2, 2, 2), 8.0))
        },
        coords={
            "lat": np.array([2.0, 1.0]),
            "lon": np.array([0.0, 0.5]),
            "time": np.array([100, 200], dtype="timedelta64[s]"),
            "datetime": xa.Variable(
                "time", np.array([10, 20], dtype="datetime64[D]")
            ),
        },
    )

    data_utils.add_tisr_var(data)

    self.assertIn(data_utils.TISR, set(data.variables))

  def test_add_tisr_var_existing_var_not_overridden(self):
    dims = ["time", "lat", "lon"]
    data = xa.Dataset(
        data_vars={
            "var1": (dims, np.full((2, 2, 2), 8.0)),
            data_utils.TISR: (dims, np.full((2, 2, 2), 1200.0)),
        },
        coords={
            "lat": np.array([2.0, 1.0]),
            "lon": np.array([0.0, 0.5]),
            "time": np.array([100, 200], dtype="timedelta64[s]"),
            "datetime": xa.Variable(
                "time", np.array([10, 20], dtype="datetime64[D]")
            ),
        },
    )

    data_utils.add_derived_vars(data)

    np.testing.assert_allclose(data[data_utils.TISR], 1200.0)

  def test_add_tisr_var_works_with_batch_dim_size_one(self):
    data = xa.Dataset(
        data_vars={
            "var1": (
                ["batch", "time", "lat", "lon"],
                np.full((1, 2, 2, 2), 8.0),
            )
        },
        coords={
            "lat": np.array([2.0, 1.0]),
            "lon": np.array([0.0, 0.5]),
            "time": np.array([100, 200], dtype="timedelta64[s]"),
            "datetime": xa.Variable(
                ("batch", "time"), np.array([[10, 20]], dtype="datetime64[D]")
            ),
        },
    )

    data_utils.add_tisr_var(data)

    self.assertIn(data_utils.TISR, set(data.variables))

  def test_add_tisr_var_fails_with_batch_dim_size_greater_than_one(self):
    data = xa.Dataset(
        data_vars={
            "var1": (
                ["batch", "time", "lat", "lon"],
                np.full((2, 2, 2, 2), 8.0),
            )
        },
        coords={
            "lat": np.array([2.0, 1.0]),
            "lon": np.array([0.0, 0.5]),
            "time": np.array([100, 200], dtype="timedelta64[s]"),
            "datetime": xa.Variable(
                ("batch", "time"),
                np.array([[10, 20], [100, 200]], dtype="datetime64[D]"),
            ),
        },
    )

    with self.assertRaisesRegex(ValueError, r"cannot select a dimension"):
      data_utils.add_tisr_var(data)


if __name__ == "__main__":
  absltest.main()
