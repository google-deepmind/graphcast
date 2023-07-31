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
"""Check that the checkpoint serialization is reversable."""

import dataclasses
import io
from typing import Any, Optional, Union

from absl.testing import absltest
from graphcast import checkpoint
import numpy as np


@dataclasses.dataclass
class SubConfig:
  a: int
  b: str


@dataclasses.dataclass
class Config:
  bt: bool
  bf: bool
  i: int
  f: float
  o1: Optional[int]
  o2: Optional[int]
  o3: Union[int, None]
  o4: Union[int, None]
  o5: int | None
  o6: int | None
  li: list[int]
  ls: list[str]
  ldc: list[SubConfig]
  tf: tuple[float, ...]
  ts: tuple[str, ...]
  t: tuple[str, int, SubConfig]
  tdc: tuple[SubConfig, ...]
  dsi: dict[str, int]
  dss: dict[str, str]
  dis: dict[int, str]
  dsdis: dict[str, dict[int, str]]
  dc: SubConfig
  dco: Optional[SubConfig]
  ddc: dict[str, SubConfig]


@dataclasses.dataclass
class Checkpoint:
  params: dict[str, Any]
  config: Config


class DataclassTest(absltest.TestCase):

  def test_serialize_dataclass(self):
    ckpt = Checkpoint(
        params={
            "layer1": {
                "w": np.arange(10).reshape(2, 5),
                "b": np.array([2, 6]),
            },
            "layer2": {
                "w": np.arange(8).reshape(2, 4),
                "b": np.array([2, 6]),
            },
            "blah": np.array([3, 9]),
        },
        config=Config(
            bt=True,
            bf=False,
            i=42,
            f=3.14,
            o1=1,
            o2=None,
            o3=2,
            o4=None,
            o5=3,
            o6=None,
            li=[12, 9, 7, 15, 16, 14, 1, 6, 11, 4, 10, 5, 13, 3, 8, 2],
            ls=list("qhjfdxtpzgemryoikwvblcaus"),
            ldc=[SubConfig(1, "hello"), SubConfig(2, "world")],
            tf=(1, 4, 2, 10, 5, 9, 13, 16, 15, 8, 12, 7, 11, 14, 3, 6),
            ts=("hello", "world"),
            t=("foo", 42, SubConfig(1, "bar")),
            tdc=(SubConfig(1, "hello"), SubConfig(2, "world")),
            dsi={"a": 1, "b": 2, "c": 3},
            dss={"d": "e", "f": "g"},
            dis={1: "a", 2: "b", 3: "c"},
            dsdis={"a": {1: "hello", 2: "world"}, "b": {1: "world"}},
            dc=SubConfig(1, "hello"),
            dco=None,
            ddc={"a": SubConfig(1, "hello"), "b": SubConfig(2, "world")},
        ))

    buffer = io.BytesIO()
    checkpoint.dump(buffer, ckpt)
    buffer.seek(0)
    ckpt2 = checkpoint.load(buffer, Checkpoint)
    np.testing.assert_array_equal(ckpt.params["layer1"]["w"],
                                  ckpt2.params["layer1"]["w"])
    np.testing.assert_array_equal(ckpt.params["layer1"]["b"],
                                  ckpt2.params["layer1"]["b"])
    np.testing.assert_array_equal(ckpt.params["layer2"]["w"],
                                  ckpt2.params["layer2"]["w"])
    np.testing.assert_array_equal(ckpt.params["layer2"]["b"],
                                  ckpt2.params["layer2"]["b"])
    np.testing.assert_array_equal(ckpt.params["blah"], ckpt2.params["blah"])
    self.assertEqual(ckpt.config, ckpt2.config)


if __name__ == "__main__":
  absltest.main()
