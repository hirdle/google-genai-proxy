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


"""Tests for get_function_response_parts."""

import pytest

from ..._extra_utils import get_function_response_parts
from ...errors import UnsupportedFunctionError
from ...types import Candidate
from ...types import Content
from ...types import FunctionCall
from ...types import FunctionResponse
from ...types import GenerateContentResponse
from ...types import Part


def test_integer_value():
  def func_under_test(a: int) -> int:
    return a + 1

  response = GenerateContentResponse(
      candidates=[
          Candidate(
              content=Content(
                  parts=[
                      Part(
                          function_call=FunctionCall(
                              name='func_under_test',
                              args={'a': 1},
                          )
                      )
                  ]
              )
          )
      ]
  )
  function_map = {'func_under_test': func_under_test}
  expected_parts = [
      Part(
          function_response=FunctionResponse(
              name='func_under_test',
              response={'result': 2},
          )
      )
  ]
  actual_parts = get_function_response_parts(response, function_map)

  for actual_part, expected_part in zip(actual_parts, expected_parts):
    assert actual_part.model_dump_json(
        exclude_none=True
    ) == expected_part.model_dump_json(exclude_none=True)


def test_float_value():
  def func_under_test(a: float) -> float:
    return a + 1.0

  response = GenerateContentResponse(
      candidates=[
          Candidate(
              content=Content(
                  parts=[
                      Part(
                          function_call=FunctionCall(
                              name='func_under_test',
                              args={'a': 1.0},
                          )
                      )
                  ]
              )
          )
      ]
  )
  function_map = {'func_under_test': func_under_test}
  expected_parts = [
      Part(
          function_response=FunctionResponse(
              name='func_under_test',
              response={'result': 2.0},
          )
      )
  ]
  actual_parts = get_function_response_parts(response, function_map)

  for actual_part, expected_part in zip(actual_parts, expected_parts):
    assert actual_part.model_dump_json(
        exclude_none=True
    ) == expected_part.model_dump_json(exclude_none=True)


def test_string_value():
  def func_under_test(a: str) -> str:
    return a + '1'

  response = GenerateContentResponse(
      candidates=[
          Candidate(
              content=Content(
                  parts=[
                      Part(
                          function_call=FunctionCall(
                              name='func_under_test',
                              args={'a': '1.0'},
                          )
                      )
                  ]
              )
          )
      ]
  )
  function_map = {'func_under_test': func_under_test}
  expected_parts = [
      Part(
          function_response=FunctionResponse(
              name='func_under_test',
              response={'result': '1.01'},
          )
      )
  ]
  actual_parts = get_function_response_parts(response, function_map)

  for actual_part, expected_part in zip(actual_parts, expected_parts):
    assert actual_part.model_dump_json(
        exclude_none=True
    ) == expected_part.model_dump_json(exclude_none=True)
