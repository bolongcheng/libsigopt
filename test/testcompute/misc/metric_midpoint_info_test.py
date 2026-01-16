# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import numpy as np
import pytest

from libsigopt.compute.misc.constant import (
    DEFAULT_CONSTANT_LIAR_VALUE,
    ConstantLiarType,
)
from libsigopt.compute.misc.data_containers import MultiMetricMidpointInfo, SingleMetricMidpointInfo


class TestSingleMetricMidpointInfo:
    def test_creation_skip(self):
        mmi = SingleMetricMidpointInfo(np.array([]), np.array([], dtype=bool))
        assert mmi.skip
        mmi = SingleMetricMidpointInfo(np.random.random(5), np.full(5, True, dtype=bool))
        assert mmi.skip

        assert mmi.relative_objective_value(2.3) == -2.3
        assert mmi.relative_objective_variance(2.3) == 2.3
        assert mmi.relative_objective_variance(0) > 0
        assert mmi.undo_scaling(2.3) == -2.3
        assert mmi.undo_scaling_variances(2.3) == 2.3

        unscaled_values = mmi.undo_scaling(np.array([1.3, 5.3]))
        assert isinstance(unscaled_values, np.ndarray) and all(unscaled_values == -np.array([1.3, 5.3]))

    # NOTE: This test is hard-coded to pass for scaling to [-.1, .1]
    #       This is intentional so that we think about what impact that has if we ever change it
    def test_scaling(self):
        count = 10
        mmi = SingleMetricMidpointInfo(np.arange(count), np.full(count, False, dtype=bool))
        assert not mmi.skip
        assert mmi.min == 0.0
        assert mmi.max == 9.0
        assert mmi.midpoint == 4.5
        assert mmi.scale == 0.2 / 9.0

        assert mmi.relative_objective_value(0.0) == 0.1
        assert mmi.relative_objective_value(9.0) == -0.1
        assert mmi.relative_objective_variance(0.3) == (0.2 / 9.0) ** 2 * 0.3
        assert mmi.undo_scaling(0.1) == 0.0
        assert mmi.undo_scaling(-0.1) == 9.0
        assert mmi.undo_scaling_variances((0.2 / 9.0) ** 2 * 0.3) == 0.3

        unscaled_values = mmi.undo_scaling(np.array([0.1, 0.0]))
        assert isinstance(unscaled_values, np.ndarray) and all(unscaled_values == np.array([0.0, 4.5]))

    def test_same_value_scaling(self):
        count = 10
        mmi = SingleMetricMidpointInfo(10 * np.ones(count), np.full(count, False, dtype=bool))
        assert not mmi.skip
        assert mmi.min == mmi.max == 10
        assert mmi.midpoint == 10
        assert mmi.scale == 1 / 10

        assert mmi.relative_objective_value(2.0) == 0.8
        assert mmi.relative_objective_value(12.0) == -0.2
        assert mmi.relative_objective_variance(0.5) == 0.5 * 0.1**2
        assert mmi.undo_scaling(0.1) == 9.0
        assert mmi.undo_scaling(-3.0) == 40.0
        assert np.allclose(mmi.undo_scaling_variances(3.6e-3), 0.36)

        unscaled_values = mmi.undo_scaling(np.array([0.1, 0.0]))
        assert isinstance(unscaled_values, np.ndarray) and all(unscaled_values == np.array([9.0, 10.0]))

        mmi = SingleMetricMidpointInfo(0.01 * np.ones(count), np.full(count, False, dtype=bool))
        assert not mmi.skip
        assert mmi.min == mmi.max == 0.01
        assert mmi.midpoint == 0
        assert mmi.scale == 1

        assert mmi.relative_objective_value(0.01) == -0.01
        assert mmi.relative_objective_value(-1.0) == 1.0
        assert mmi.relative_objective_variance(0.5) == 0.5
        assert mmi.undo_scaling(0.1) == -0.1
        assert mmi.undo_scaling(-3.0) == 3.0
        assert mmi.undo_scaling_variances(0.6) == 0.6

        unscaled_values = mmi.undo_scaling(np.array([0.01, 0.0]))
        assert isinstance(unscaled_values, np.ndarray) and all(unscaled_values == np.array([-0.01, 0.0]))

    def test_single_metric_objective_minimize(self):
        metric_objective = "minimize"
        mmi = SingleMetricMidpointInfo(np.array([]), np.array([], dtype=bool), metric_objective)
        assert mmi.negate == 1
        assert mmi.relative_objective_value(1.0) == 1.0
        assert mmi.relative_objective_value(-10) == -10

        unscaled_values = mmi.undo_scaling(np.array([0.1, 0.0]))
        assert isinstance(unscaled_values, np.ndarray) and all(unscaled_values == np.array([0.1, 0.0]))

        count = 10
        mmi = SingleMetricMidpointInfo(np.arange(count), np.full(count, False, dtype=bool), metric_objective)
        assert mmi.relative_objective_value(0.0) == -0.1
        assert mmi.relative_objective_value(9.0) == 0.1
        assert mmi.undo_scaling(0.1) == 9.0
        assert mmi.undo_scaling(-0.1) == 0.0

        unscaled_values = mmi.undo_scaling(np.array([0.1, 0.0]))
        assert isinstance(unscaled_values, np.ndarray) and all(unscaled_values == np.array([9.0, 4.5]))

    def test_single_metric_objective_maximize(self):
        metric_objective = "maximize"
        mmi = SingleMetricMidpointInfo(np.array([]), np.array([], dtype=bool), metric_objective)
        assert mmi.negate == -1
        assert mmi.relative_objective_value(1.0) == -1.0
        assert mmi.relative_objective_value(-10) == 10

        unscaled_values = mmi.undo_scaling(np.array([0.1, 0.0]))
        assert isinstance(unscaled_values, np.ndarray) and all(unscaled_values == np.array([-0.1, 0.0]))

        count = 10
        mmi = SingleMetricMidpointInfo(np.arange(count), np.full(count, False, dtype=bool), metric_objective)
        assert mmi.relative_objective_value(0.0) == 0.1
        assert mmi.relative_objective_value(9.0) == -0.1
        assert mmi.undo_scaling(0.1) == 0.0
        assert mmi.undo_scaling(-0.1) == 9.0

        unscaled_values = mmi.undo_scaling(np.array([0.1, 0.0]))
        assert isinstance(unscaled_values, np.ndarray) and all(unscaled_values == np.array([0.0, 4.5]))

    def test_lie_management(self):
        count = 10
        values = np.arange(count)
        failures = np.full(count, False, dtype=bool)
        mmi = SingleMetricMidpointInfo(values=values, failures=failures)
        with pytest.raises(AssertionError):
            mmi.compute_lie_value("fake_lie_method")

        mmi = SingleMetricMidpointInfo(values=np.array([]), failures=np.array([]))
        assert mmi.compute_lie_value(ConstantLiarType.MEAN) == DEFAULT_CONSTANT_LIAR_VALUE

        mmi = SingleMetricMidpointInfo(values=np.array([1]), failures=np.array([True]))
        assert mmi.compute_lie_value(ConstantLiarType.MAX) == DEFAULT_CONSTANT_LIAR_VALUE

        values = np.array([0.1, 0.3, -0.8, -0.2, 0.25])
        failures = np.array([False, True, False, True, False], dtype=bool)
        outputs = {
            ConstantLiarType.MIN: -0.8,
            ConstantLiarType.MAX: 0.25,
            ConstantLiarType.MEAN: (0.1 - 0.8 + 0.25) / 3.0,
        }

        mmi = SingleMetricMidpointInfo(values=values, failures=failures)
        for lie_method, lie_value in outputs.items():
            assert mmi.compute_lie_value(lie_method) == lie_value

        failures = np.full_like(values, False, dtype=bool)
        outputs = {
            ConstantLiarType.MIN: -0.8,
            ConstantLiarType.MAX: 0.3,
            ConstantLiarType.MEAN: (0.1 + 0.3 - 0.8 - 0.2 + 0.25) / 5.0,
        }

        mmi = SingleMetricMidpointInfo(values=values, failures=failures)
        for lie_method, lie_value in outputs.items():
            assert mmi.compute_lie_value(lie_method) == lie_value

        outputs = {
            ConstantLiarType.MIN: 0.3,
            ConstantLiarType.MAX: -0.8,
            ConstantLiarType.MEAN: (0.1 + 0.3 - 0.8 - 0.2 + 0.25) / 5.0,
        }
        mmi = SingleMetricMidpointInfo(values=values, failures=failures, objective="minimize")
        for lie_method, lie_value in outputs.items():
            assert mmi.compute_lie_value(lie_method) == lie_value


class TestMultiMetricMidpointInfo:
    def test_creation_skip(self):
        with pytest.raises(AssertionError):
            MultiMetricMidpointInfo(np.array([]), np.array([], dtype=bool))
        with pytest.raises(AssertionError):
            MultiMetricMidpointInfo(np.random.random(5), np.full(5, True, dtype=bool))

        mmi = MultiMetricMidpointInfo(np.random.random((5, 1)), np.full(5, True, dtype=bool))
        assert mmi.skip
        assert mmi.relative_objective_value(2.3) == -2.3
        assert mmi.relative_objective_variance(2.3) == 2.3
        assert mmi.relative_objective_variance(0) > 0
        assert mmi.undo_scaling(2.3) == -2.3
        assert mmi.undo_scaling_variances(2.3) == 2.3

        unscaled_values = mmi.undo_scaling(np.array([1.3, 5.3]))
        assert isinstance(unscaled_values, np.ndarray) and all(unscaled_values == -np.array([1.3, 5.3]))

        mmi = MultiMetricMidpointInfo(np.random.random((5, 2)), np.full(5, True, dtype=bool))
        assert mmi.skip
        assert (mmi.relative_objective_value(np.array([2.3, -1])) == np.array([-2.3, 1])).all()
        assert (mmi.relative_objective_variance(np.array([2.3, 1])) == np.array([2.3, 1])).all()
        assert (mmi.relative_objective_variance(np.zeros((1, 2))) > np.zeros((1, 2))).all()
        assert (mmi.undo_scaling(np.array([2.3, 1])) == np.array([-2.3, -1])).all()
        assert (mmi.undo_scaling_variances(np.array([2.3, 1])) == np.array([2.3, 1])).all()

        unscaled_values = mmi.undo_scaling(np.array([[1.3, 5.3], [2.1, 2.2]]))
        assert (
            isinstance(unscaled_values, np.ndarray) and (unscaled_values == -np.array([[1.3, 5.3], [2.1, 2.2]])).all()
        )

    def test_one_metric_all_same(self):
        mmi = MultiMetricMidpointInfo(
            np.concatenate((np.zeros((5, 1)), np.arange(5)[:, None]), axis=1),
            np.full(5, False, dtype=bool),
        )
        assert not mmi.skip
        assert (mmi.midpoint == np.array([0, 2.0])).all()
        assert (mmi.scale == np.array([1, 0.05])).all()

        assert (mmi.relative_objective_value(np.array([0.0, 0.0])) == np.array([0.0, 0.1])).all()
        assert (mmi.relative_objective_value(np.array([9.0, 4.0])) == np.array([-9.0, -0.1])).all()
        assert (mmi.relative_objective_variance(0.3) == np.array([0.3, (0.2 / 4.0) ** 2 * 0.3])).all()
        assert (mmi.undo_scaling(np.array([0.1, 0.1])) == np.array([-0.1, 0.0])).all()
        assert (mmi.undo_scaling(np.array([-0.1, -0.1])) == np.array([0.1, 4.0])).all()
        assert np.allclose(mmi.undo_scaling_variances(0.02), [0.02, 8.0], atol=1e-10)

        unscaled_values = mmi.undo_scaling(np.array([[0.1, 0.0], [-1.0, 0.1]]))
        assert isinstance(unscaled_values, np.ndarray)
        assert (unscaled_values == np.array([[-0.1, 2.0], [1.0, 0.0]])).all()

    def test_scaling(self):
        mmi = MultiMetricMidpointInfo(
            np.concatenate((np.arange(10)[:, None], np.arange(0, -10, -1)[:, None]), axis=1),
            np.full(10, False, dtype=bool),
        )
        assert not mmi.skip
        assert (mmi.midpoint == np.array([4.5, -4.5])).all()
        assert mmi.scale is not None
        assert (mmi.scale == 0.2 / 9.0).all()

        assert (mmi.relative_objective_value(np.array([0.0, 0.0])) == np.array([0.1, -0.1])).all()
        assert (mmi.relative_objective_value(np.array([9.0, -9.0])) == np.array([-0.1, 0.1])).all()
        assert (mmi.relative_objective_variance(0.3) == (0.2 / 9.0) ** 2 * 0.3).all()
        assert (mmi.undo_scaling(np.array([0.1, 0.1])) == np.array([0.0, -9.0])).all()
        assert (mmi.undo_scaling(np.array([-0.1, -0.1])) == np.array([9.0, 0.0])).all()
        assert (mmi.undo_scaling_variances((0.2 / 9.0) ** 2 * 0.3) == 0.3).all()

        unscaled_values = mmi.undo_scaling(np.array([[0.1, 0.0], [0.0, 0.1]]))
        assert isinstance(unscaled_values, np.ndarray)
        assert (unscaled_values == np.array([[0.0, -4.5], [4.5, -9.0]])).all()

    def test_multimetric_objective(self):
        metric_objectives = ["minimize", "maximize"]
        values = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
        mmi = MultiMetricMidpointInfo(values, np.full(3, False, dtype=bool), metric_objectives)

        assert mmi.negate is not None
        assert mmi.negate.shape[0] == 2
        assert mmi.negate[0] == 1
        assert mmi.negate[1] == -1

        scaled_values = mmi.relative_objective_value(values)
        assert (scaled_values == np.array([[-0.1, 0.1], [0.0, 0.0], [0.1, -0.1]])).all()

        unscaled_values = mmi.undo_scaling(scaled_values)
        assert (unscaled_values == np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])).all()

        assert isinstance(unscaled_values, np.ndarray)

    def test_lie_management(self):
        mmi = MultiMetricMidpointInfo(np.empty((10, 2)), np.full(10, False, dtype=bool))
        with pytest.raises(AssertionError):
            mmi.compute_lie_value("fake_lie_method")

        mmi = MultiMetricMidpointInfo(values=np.array([[]]), failures=np.array([]))
        assert (mmi.compute_lie_value(ConstantLiarType.MEAN) == DEFAULT_CONSTANT_LIAR_VALUE).all()

        mmi = MultiMetricMidpointInfo(values=np.array([[]]), failures=np.array([]))
        assert (mmi.compute_lie_value(ConstantLiarType.MIN) == DEFAULT_CONSTANT_LIAR_VALUE).all()

        values = np.array([[0.2, 0.5], [0.1, -0.2], [0.7, 0.4], [-0.5, 0.9], [0.8, -0.3]])
        failures = np.array([False, True, False, True, False], dtype=bool)
        outputs = {
            ConstantLiarType.MIN: np.array([0.2, -0.3]),
            ConstantLiarType.MAX: np.array([0.8, 0.5]),
            ConstantLiarType.MEAN: np.array([(0.2 + 0.7 + 0.8) / 3, (0.5 + 0.4 - 0.3) / 3]),
        }

        mmi = MultiMetricMidpointInfo(values=values, failures=failures)
        for lie_method, lie_value in outputs.items():
            assert (mmi.compute_lie_value(lie_method) == lie_value).all()
