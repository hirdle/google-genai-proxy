# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Tests for get_function_map."""

import pytest

from ..._extra_utils import get_function_map
from ...errors import UnsupportedFunctionError
from ...types import GenerateContentConfig


def test_coroutine_function():
  async def func_under_test():
    pass

  config = GenerateContentConfig(tools=[func_under_test])

  with pytest.raises(UnsupportedFunctionError):
    get_function_map(config)


def test_empty_config():
  config = {}

  assert get_function_map(config) == {}


def test_empty_tools():
  config = GenerateContentConfig(top_p=0.5)

  assert get_function_map(config) == {}


def test_valid_function():
  def func_under_test():
    pass

  config = GenerateContentConfig(tools=[func_under_test])

  assert get_function_map(config) == {'func_under_test': func_under_test}
