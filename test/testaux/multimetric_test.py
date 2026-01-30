# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import pytest

from libsigopt.aux.multimetric import find_pareto_frontier_observations_for_maximization


def test_find_pareto_frontier_observations_for_maximization() -> None:
    values = [[0.0, 1.0], [1.0, 2.0], [3.0, 4.0], [4.0, 3.0]]
    observations = [0, 1, 2, 3]
    pf, npf = find_pareto_frontier_observations_for_maximization(values, observations)
    assert set(pf) == {2, 3} and set(npf) == {0, 1}


def test_exclude_repeated_values_from_pareto_frontier() -> None:
    values = [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [4.0, 2.0], [5.0, 2.0], [6.0, 1.0]]
    observations = [0, 1, 2, 3, 4, 5]
    pf, npf = find_pareto_frontier_observations_for_maximization(values, observations)
    assert set(pf) == {2, 4, 5} and set(npf) == {0, 1, 3}
