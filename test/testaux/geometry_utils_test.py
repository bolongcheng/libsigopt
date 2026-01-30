# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from libsigopt.aux.geometry_utils import compute_distance_matrix_squared

from testaux.numerical_test_case import assert_vector_within_relative_norm


@pytest.mark.parametrize("dim", [5, 10, 15])
@pytest.mark.parametrize("num_x", [10, 100, 1000])
@pytest.mark.parametrize("num_z", [100, 200, 300])
def test_distance_is_correct(dim: int, num_x: int, num_z: int) -> None:
    x = np.random.random((num_x, dim))
    z = np.random.random((num_z, dim))
    dm_sq = compute_distance_matrix_squared(x, z)
    dm_sq_cdist = cdist(x, z) ** 2
    assert_vector_within_relative_norm(dm_sq, dm_sq_cdist, tol=float(1e-15 * dim), norm=np.inf)
