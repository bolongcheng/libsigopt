# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import math
import sys

import numpy as np
import pytest

from libsigopt.aux.utils import is_integer, is_number


@pytest.mark.parametrize(
    "num",
    [
        -1,
        0,
        1,
        0x213,
        0o2321,
        2**31 - 1,
        sys.maxsize,
        -sys.maxsize - 1,
    ],
)
def test_int_is_integer_and_number(num):
    assert is_integer(num)
    assert is_number(num)


@pytest.mark.parametrize(
    "num",
    [
        "a",
        True,
        [],
        np.nan,
        np.inf,
        float("inf"),
        float("nan"),
        math.inf,
        -math.inf,
    ],
)
def test_not_int_or_numbers(num):
    assert is_integer(num) is False
    assert is_number(num) is False


@pytest.mark.parametrize(
    "num",
    [
        -1e-32,
        -1.023,
        0.0,
        1.2,
        np.pi,
        1e32,
        1e123,
    ],
)
def test_double_is_not_int(num):
    assert is_integer(num) is False
    assert is_number(num)
