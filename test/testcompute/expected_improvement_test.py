# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
"""Test cases for the expected improvement acquisition function and any potential gradients."""

import numpy as np
import pytest

from libsigopt.compute.covariance import SquareExponential
from libsigopt.compute.domain import ContinuousDomain
from libsigopt.compute.expected_improvement import (
    ExpectedImprovement,
    ExpectedImprovementWithFailures,
    ExpectedParallelImprovement,
)
from libsigopt.compute.gaussian_process import GaussianProcess
from libsigopt.compute.misc.data_containers import HistoricalData
from libsigopt.compute.vectorized_optimizers import AdamOptimizer

from testaux.numerical_test_case import (
    assert_scalar_within_relative,
    assert_vector_within_relative,
    check_gradient_with_finite_difference,
)
from testcompute.gp_test_utils import form_continous_and_uniform_domain, form_gaussian_process_and_data


def _check_ei_symmetry(ei_eval, point_to_sample, shifts):
    """Compute ei at each ``[point_to_sample +/- shift for shift in shifts]`` and check for equality."""
    for shift in shifts:
        left_ei = ei_eval.evaluate_at_point_list(np.atleast_2d(point_to_sample - shift))[0]
        left_grad_ei = ei_eval.evaluate_grad_at_point_list(np.atleast_2d(point_to_sample - shift))[0]

        right_ei = ei_eval.evaluate_at_point_list(np.atleast_2d(point_to_sample + shift))[0]
        right_grad_ei = ei_eval.evaluate_grad_at_point_list(np.atleast_2d(point_to_sample + shift))[0]

        assert_scalar_within_relative(left_ei, right_ei, 5.0e-15)
        assert_vector_within_relative(left_grad_ei, -right_grad_ei, 5.0e-15)


def test_1d_analytic_ei_edge_cases():
    """Test cases where analytic EI would attempt to compute 0/0 without variance lower bounds."""
    base_coord = np.array([[0.5]])
    points = np.array([base_coord, base_coord * 2.0])
    values = np.array([-1.809342, -1.09342])
    value_vars = np.array([0, 0])

    # First a symmetric case: only one historical point
    data = HistoricalData(1)
    data.append_historical_data(points[0], values[0, None], value_vars[0, None])

    hyperparameters = np.array([0.2, 0.3])
    covariance = SquareExponential(hyperparameters)
    gaussian_process = GaussianProcess(covariance, data)

    ei_eval = ExpectedImprovement(gaussian_process)

    ei = ei_eval.evaluate_at_point_list(base_coord)[0]
    grad_ei = ei_eval.evaluate_grad_at_point_list(base_coord)[0]
    assert_scalar_within_relative(ei, 0.0, 1.0e-14)
    assert_vector_within_relative(grad_ei, np.zeros(grad_ei.shape), 1.0e-15)

    shifts = (2.0e-15, 4.0e-11, 3.14e-6, 8.89e-1, 2.71)
    _check_ei_symmetry(ei_eval, base_coord[0], shifts)

    # Now introduce some asymmetry with a second point
    # Right side has a larger objetive value, so the EI minimum
    # is shifted *slightly* to the left of best_value.
    data.append_historical_data(points[1], values[1, None], value_vars[1, None])
    gaussian_process = GaussianProcess(covariance, data)
    shift = 3.0e-12
    ei_eval = ExpectedImprovement(gaussian_process)
    ei = ei_eval.evaluate_at_point_list(base_coord - shift)[0]
    grad_ei = ei_eval.evaluate_grad_at_point_list(base_coord - shift)[0]
    assert_scalar_within_relative(ei, 0.0, 1.0e-14)
    assert_vector_within_relative(grad_ei, np.zeros(grad_ei.shape), 1.0e-15)


@pytest.mark.parametrize("idx", range(4))
def test_best_value_and_location(gaussian_process_list, idx):
    gaussian_process = gaussian_process_list[idx]
    ei = ExpectedImprovement(gaussian_process)
    assert_scalar_within_relative(ei.best_value, gaussian_process.best_observed_value, 1.0e-14)
    assert_vector_within_relative(ei.best_location, gaussian_process.best_observed_location, 1.0e-14)


@pytest.mark.parametrize("idx", range(4))
def test_evaluate_ei_at_points_for_base_ei(one_hot_domain_list, gaussian_process_list, idx):
    domain, gaussian_process = one_hot_domain_list[idx], gaussian_process_list[idx]

    ei_eval = ExpectedImprovement(gaussian_process)

    num_to_eval = 10
    points_to_evaluate = domain.generate_quasi_random_points_in_domain(num_to_eval)

    test_values = ei_eval.evaluate_at_point_list(points_to_evaluate)

    # NOTE: Because of the vectorization in distance matrix computation these can be off near unit roundoff
    for i, value in enumerate(test_values):
        truth = ei_eval.evaluate_at_point_list(points_to_evaluate[i, None, :])[0]
        assert_scalar_within_relative(value, truth, 1.0e-8)


@pytest.mark.parametrize("idx", range(4))
def test_qei_working(one_hot_domain_list, gaussian_process_list, idx):
    domain, gaussian_process = one_hot_domain_list[idx], gaussian_process_list[idx]
    all_points = domain.generate_quasi_random_points_in_domain(9)

    for i in range(2, len(all_points)):
        points_to_sample = all_points[:i]
        ei_eval = ExpectedParallelImprovement(
            gaussian_process,
            len(points_to_sample),
            num_mc_iterations=10000,
        )
        assert (
            ei_eval.evaluate_at_point_list(
                np.reshape(points_to_sample, (1, ei_eval.num_points_to_sample, ei_eval.dim))
            )[0]
            >= 0
        )

    num_points_to_sample = 3
    num_points_to_evaluate = 6
    points_being_sampled = all_points[:4]
    points_to_sample = domain.generate_quasi_random_points_in_domain(num_points_to_sample * num_points_to_evaluate)
    points_to_sample = points_to_sample.reshape(num_points_to_evaluate, num_points_to_sample, domain.dim)
    ei_eval = ExpectedParallelImprovement(
        gaussian_process,
        num_points_to_sample,
        points_being_sampled=points_being_sampled,
        num_mc_iterations=10000,
    )
    ei_vals = ei_eval.evaluate_at_point_list(points_to_sample)
    assert all(ei_vals >= 0)


@pytest.mark.flaky(reruns=1)
@pytest.mark.parametrize("idx", [0])
def test_qei_accuracy(one_hot_domain_list, gaussian_process_list, idx):
    num_points_to_sample = 3
    num_points_being_sampled = 4
    num_random_tests = 30
    mc_iterations_values = [100, 1000, 10000]

    domain, gaussian_process = one_hot_domain_list[idx], gaussian_process_list[idx]
    points_being_sampled = domain.generate_quasi_random_points_in_domain(num_points_being_sampled)
    points_to_sample = domain.generate_quasi_random_points_in_domain(num_points_to_sample)

    ei_eval = ExpectedParallelImprovement(
        gaussian_process,
        num_points_to_sample,
        points_being_sampled=points_being_sampled,
    )
    true_result = ei_eval._compute_expected_improvement_qd_analytic(points_to_sample)

    std_results = []
    for num_mc_iterations in mc_iterations_values:
        ei_eval.num_mc_iterations = num_mc_iterations

        mc_results = [
            ei_eval.evaluate_at_point_list(
                np.reshape(points_to_sample, (1, ei_eval.num_points_to_sample, ei_eval.dim))
            )[0]
            for _ in range(num_random_tests)
        ]
        ei_mean_eapl, ei_std_eapl = np.mean(mc_results), np.std(mc_results)

        mc_results = [
            ei_eval.evaluate_at_point_list(
                np.reshape(points_to_sample, (1, ei_eval.num_points_to_sample, ei_eval.dim))
            )[0]
            for _ in range(num_random_tests)
        ]
        ei_mean_caf, ei_std_caf = np.mean(mc_results), np.std(mc_results)

        assert abs(ei_mean_eapl - true_result) < 2 * ei_std_eapl
        assert abs(ei_mean_caf - true_result) < 2 * ei_std_caf
        std_results.append(ei_std_eapl)
    assert all(np.diff(std_results) < 0) or any(np.array(std_results) == 0)


def test_multistart_analytic_expected_improvement_optimization():
    """
    Check that multistart optimization (gradient descent) can find the optimum point
    to sample (using 1D analytic EI).
    """
    np.random.seed(3148)
    dim = 3
    domain = form_continous_and_uniform_domain(dim=dim, lower_element=-2, higher_element=2)
    gaussian_process = form_gaussian_process_and_data(domain=domain, num_sampled=50, noise_per_point=0.002)

    tolerance = 1.0e-6

    ei_eval = ExpectedImprovement(gaussian_process)

    # expand the domain so that we are definitely not doing constrained optimization
    expanded_domain = ContinuousDomain([[-4.0, 2.0]] * domain.dim)
    ei_optimizer = AdamOptimizer(
        acquisition_function=ei_eval,
        domain=expanded_domain,
        num_multistarts=100,
        maxiter=1000,
    )
    best_point, _ = ei_optimizer.optimize()

    # Check that gradients are small or that the answer is on a boundary
    gradient = ei_eval.evaluate_grad_at_point_list(np.atleast_2d(best_point))
    if not expanded_domain.check_point_on_boundary(best_point, tol=1e-3):
        assert_vector_within_relative(gradient, np.zeros(gradient.shape), tolerance)

    # Check that output is in the domain
    assert expanded_domain.check_point_acceptable(best_point) is True


@pytest.mark.parametrize("idx", range(4))
def test_evaluation_probabilistic_failures(
    one_hot_domain_list,
    gaussian_process_list,
    probabilistic_failures_list,
    idx,
):
    domain, gp, pf = one_hot_domain_list[idx], gaussian_process_list[idx], probabilistic_failures_list[idx]
    ei = ExpectedImprovement(gp)
    eif = ExpectedImprovementWithFailures(gp, pf)
    pts = domain.generate_quasi_random_points_in_domain(50)
    ei_vals = ei.evaluate_at_point_list(pts)
    pf_vals = pf.compute_probability_of_success(pts)
    eif_vals = eif.evaluate_at_point_list(pts)
    assert_vector_within_relative(ei_vals * pf_vals, eif_vals, 1e-13)


@pytest.mark.parametrize("idx", range(4))
def test_grad_against_finite_difference(
    one_hot_domain_list,
    gaussian_process_list,
    probabilistic_failures_list,
    idx,
):
    h = 1e-6
    n_test = 50
    domain, gp, pf = one_hot_domain_list[idx], gaussian_process_list[idx], probabilistic_failures_list[idx]
    eif = ExpectedImprovementWithFailures(gp, pf)
    pts = domain.generate_quasi_random_points_in_domain(n_test)
    check_gradient_with_finite_difference(
        pts,
        eif.evaluate_at_point_list,
        eif.evaluate_grad_at_point_list,
        tol=domain.dim * 1e-6,
        fd_step=h * np.ones(domain.dim),
    )


@pytest.mark.parametrize("idx", range(4))
def test_evaluation_product_probabilistic_failures(
    one_hot_domain_list,
    gaussian_process_list,
    product_of_list_probabilistic_failures_list,
    idx,
):
    domain, gp, ppf = (
        one_hot_domain_list[idx],
        gaussian_process_list[idx],
        product_of_list_probabilistic_failures_list[idx],
    )
    ei = ExpectedImprovement(gp)
    eif = ExpectedImprovementWithFailures(gp, ppf)
    pts = domain.generate_quasi_random_points_in_domain(50)
    ei_vals = ei.evaluate_at_point_list(pts)
    pf_vals = ppf.compute_probability_of_success(pts)
    eif_vals = eif.evaluate_at_point_list(pts)
    assert_vector_within_relative(ei_vals * pf_vals, eif_vals, 1e-13)


@pytest.mark.parametrize("idx", range(4))
def test_grad_product_against_finite_difference(
    one_hot_domain_list,
    gaussian_process_list,
    product_of_list_probabilistic_failures_list,
    idx,
):
    h = 1e-6
    n_test = 50
    domain, gp, ppf = (
        one_hot_domain_list[idx],
        gaussian_process_list[idx],
        product_of_list_probabilistic_failures_list[idx],
    )
    eif = ExpectedImprovementWithFailures(gp, ppf)
    pts = domain.generate_quasi_random_points_in_domain(n_test)
    check_gradient_with_finite_difference(
        pts,
        eif.evaluate_at_point_list,
        eif.evaluate_grad_at_point_list,
        tol=domain.dim * 1e-6,
        fd_step=h * np.ones(domain.dim),
    )
