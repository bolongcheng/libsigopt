# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0

import numpy as np
import pytest

from libsigopt.compute.covariance import SquareExponential
from libsigopt.compute.domain import CategoricalDomain
from libsigopt.compute.expected_improvement import (
    AugmentedExpectedImprovement,
    ExpectedImprovement,
    ExpectedImprovementWithFailures,
    ExpectedParallelImprovement,
)
from libsigopt.compute.gaussian_process import GaussianProcess
from libsigopt.compute.misc.data_containers import HistoricalData
from libsigopt.compute.multitask_acquisition_function import MultitaskAcquisitionFunction
from libsigopt.compute.probabilistic_failures import ProbabilisticFailuresCDF

from testaux.numerical_test_case import assert_scalar_within_relative, assert_vector_within_relative


@pytest.fixture(scope="session")
def multitask_af_setup():
    domain = CategoricalDomain(
        [
            {"var_type": "double", "elements": (-2, 3)},
            {"var_type": "double", "elements": (-1, 1)},
            {"var_type": "double", "elements": (0.1, 1.0)},
        ]
    ).one_hot_domain
    cov = SquareExponential([1.0, 0.3, 0.3, 0.4])

    x = domain.generate_quasi_random_points_in_domain(14)
    y = np.random.random(len(x))
    v = np.full_like(y, 1e-1)
    data = HistoricalData(3)
    data.append_historical_data(x, y, v)
    points_being_sampled = domain.generate_quasi_random_points_in_domain(2)

    gp = GaussianProcess(cov, data, [[0, 0, 0]])
    threshold = np.random.uniform(0.2, 0.6)
    pf = ProbabilisticFailuresCDF(gp, threshold)

    ei = ExpectedImprovement(gp)
    eif = ExpectedImprovementWithFailures(gp, pf)
    aei = AugmentedExpectedImprovement(gp)
    qei = ExpectedParallelImprovement(gp, 1, points_being_sampled=points_being_sampled)
    return {
        "domain": domain,
        "cov": cov,
        "data": data,
        "gp": gp,
        "ei": ei,
        "eif": eif,
        "aei": aei,
        "qei": qei,
    }


@pytest.mark.parametrize("af_name", ["ei", "eif", "aei", "qei"])
def test_creation(multitask_af_setup, af_name):
    af = multitask_af_setup[af_name]
    domain = multitask_af_setup["domain"]
    multitask_af = MultitaskAcquisitionFunction(af)
    new_point = domain.generate_quasi_random_points_in_domain(1)

    af_val = multitask_af.evaluate_at_point_list(new_point)[0]
    assert not (np.isnan(af_val) or np.isinf(af_val))

    if multitask_af.differentiable:
        af_grad = multitask_af.evaluate_grad_at_point_list(new_point)[0]
        assert af_grad.shape == (multitask_af.dim,)
        assert not np.any(np.isnan(af_grad)) or np.any(np.isinf(af_grad))


def test_lower_cost_preferred(multitask_af_setup):
    task_options = [0.1, 0.3, 1.0]
    data_orig = multitask_af_setup["data"]
    domain = multitask_af_setup["domain"]
    cov = multitask_af_setup["cov"]

    x = np.tile(data_orig.points_sampled[:, :-1], (len(task_options), 1))
    task_costs = np.tile(task_options, (len(data_orig.points_sampled), 1)).T.reshape(-1, 1)
    x = np.concatenate((x, task_costs), axis=1)
    y = np.tile(data_orig.points_sampled_value, (len(task_options),))
    v = np.full_like(y, 1e-2)
    data = HistoricalData(data_orig.dim)
    data.append_historical_data(x, y, v)

    gp = GaussianProcess(cov, data, [[0] * data.dim])
    ei = ExpectedImprovement(gp)
    multitask_ei = MultitaskAcquisitionFunction(ei)

    new_points_with_costs = domain.generate_quasi_random_points_in_domain(5)
    new_points = np.tile(new_points_with_costs[:, :-1], (len(task_options), 1))
    task_costs = np.tile(task_options, (len(new_points_with_costs), 1)).T.reshape(-1, 1)
    new_points = np.concatenate((new_points, task_costs), axis=1)
    mtei_vals = multitask_ei.evaluate_at_point_list(new_points)
    ei_vals = ei.evaluate_at_point_list(new_points)
    ei_vals_points_per_task = np.reshape(ei_vals, (3, 5)).T
    largest_ei_diff = np.max(abs(np.diff(ei_vals_points_per_task, axis=1)))
    mtei_vals_points_per_task = np.reshape(mtei_vals, (3, 5)).T
    assert np.allclose(ei_vals_points_per_task / task_options, mtei_vals_points_per_task)
    # MTEI prefers lower costs up to the largest difference of ei * task
    upper_bound_on_mtei_diff = largest_ei_diff * np.min(np.diff(task_options))
    assert np.all(np.diff(mtei_vals_points_per_task, axis=1) < upper_bound_on_mtei_diff)

    # NOTE: This test implicitly tests the joint_function_gradient_eval in all the other AFs


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize("af_name", ["ei", "eif", "aei", "qei"])
def test_joint_function_gradient_eval(multitask_af_setup, af_name):
    af = multitask_af_setup[af_name]
    domain = multitask_af_setup["domain"]
    test_points = domain.generate_quasi_random_points_in_domain(4)
    multitask_af = MultitaskAcquisitionFunction(af)
    if multitask_af.differentiable:
        vals, grad_vals = multitask_af.joint_function_gradient_eval(test_points)
        for test_point, val, grad_val in zip(test_points, vals, grad_vals):
            val_compare = multitask_af.evaluate_at_point_list(np.atleast_2d(test_point))[0]
            grad_val_compare = multitask_af.evaluate_grad_at_point_list(np.atleast_2d(test_point))[0]
            assert_scalar_within_relative(val, val_compare, 1e-13)
            assert_vector_within_relative(grad_val, grad_val_compare, 1e-12)
    else:
        with pytest.raises(NotImplementedError):
            multitask_af.joint_function_gradient_eval(test_points)
        vals = multitask_af.evaluate_at_point_list(test_points)
        # This test incorporates the stochastic nature of AF that have no gradient
        for test_point, val in zip(test_points, vals):
            val_compare_list = [multitask_af.evaluate_at_point_list(np.atleast_2d(test_point))[0] for _ in range(50)]
            assert abs(val - np.mean(val_compare_list)) < 3 * np.std(val_compare_list)
