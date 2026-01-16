# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import copy
from collections import namedtuple

import numpy as np
import pytest
from scipy.stats import norm

from libsigopt.compute.covariance import C2RadialMatern
from libsigopt.compute.domain import CategoricalDomain, ContinuousDomain
from libsigopt.compute.expected_improvement import (
    AugmentedExpectedImprovement,
    ExpectedImprovement,
    ExpectedImprovementWithFailures,
    ExpectedParallelImprovement,
    ExpectedParallelImprovementWithFailures,
)
from libsigopt.compute.gaussian_process import GaussianProcess
from libsigopt.compute.gaussian_process_sum import GaussianProcessSum
from libsigopt.compute.misc.data_containers import HistoricalData
from libsigopt.compute.probabilistic_failures import ProbabilisticFailures, ProductOfListOfProbabilisticFailures

from testaux.numerical_test_case import (
    assert_scalar_within_relative,
    assert_vector_within_relative,
    assert_vector_within_relative_norm,
    check_gradient_with_finite_difference,
)


ExpectedResults = namedtuple("ExpectedResults", "mean var grad_mean grad_var covariance")


FIXED_DIMS = False


@pytest.fixture(scope="module")
def gp_sum_setup():
    dims: list[int] | np.ndarray = [7] * 10 if FIXED_DIMS else np.random.randint(2, 25, size=(10,))
    domains = [CategoricalDomain([{"var_type": "double", "elements": (0, 1)}] * d).one_hot_domain for d in dims]
    gaussian_process_lists = []
    weights_lists = []
    list_size = 3
    for domain in domains:
        num_sampled = np.random.randint(50, 150)
        x = domain.generate_quasi_random_points_in_domain(num_sampled)
        y = np.sum((x - 0.5) ** 2, axis=1)
        v = np.full_like(y, 10 ** -np.random.uniform(-4, -1))
        gaussian_process_list = []
        for _ in range(list_size):
            data = HistoricalData(domain.dim)
            data.append_historical_data(x, y, v)
            cov = C2RadialMatern(np.random.uniform(0.2, 0.8, size=(domain.dim + 1,)))
            gp = GaussianProcess(cov, data, [[0] * domain.dim])
            gaussian_process_list.append(gp)
        gaussian_process_lists.append(gaussian_process_list)
        weights_lists.append(np.random.uniform(0.2, 0.8, size=list_size))
    return {
        "domains": domains,
        "gaussian_process_lists": gaussian_process_lists,
        "weights_lists": weights_lists,
    }


def compute_mixture_of_individual_gps_core_functionality(gp_list, weights, xt, option):
    num_points_xt, dim = xt.shape
    num_of_gps = len(gp_list)
    all_mean = np.zeros((num_points_xt, num_of_gps))
    all_var = np.zeros((num_points_xt, num_of_gps))
    all_grad_mean = np.zeros((num_points_xt, dim, num_of_gps))
    all_grad_var = np.zeros((num_points_xt, dim, num_of_gps))
    all_covariance = np.zeros((num_points_xt, num_points_xt, num_of_gps))

    for i, gp in enumerate(gp_list):
        if option in ("func",):
            all_mean[:, i], all_var[:, i] = gp.compute_mean_and_variance_of_points(xt)
        if option in (
            "grad",
            "both",
            "all",
        ):
            mean_variance_grad_of_points = gp.compute_mean_variance_grad_of_points(xt)
            (
                all_mean[:, i],
                all_var[:, i],
                all_grad_mean[:, :, i],
                all_grad_var[:, :, i],
            ) = mean_variance_grad_of_points
        if option in ("all",):
            all_covariance[:, :, i] = gp.compute_covariance_of_points(xt)

    mean = np.dot(all_mean, weights)
    var = np.dot(all_var, weights**2)
    grad_mean = np.dot(all_grad_mean, weights)
    grad_var = np.dot(all_grad_var, weights**2)
    covariance = np.dot(all_covariance, weights**2)
    return ExpectedResults(mean, var, grad_mean, grad_var, covariance)


def gp_samples_credible_interval_test(samples, mean, stds, k=3):
    num_samples, num_sampled_points = samples.shape
    deviation_from_mean = k * stds
    upper_bound = mean + deviation_from_mean
    lower_bound = mean - deviation_from_mean

    counter = np.zeros((1, num_sampled_points))
    for sample in samples:
        counter += np.logical_and(sample < upper_bound, sample > lower_bound).astype(int)

    # estimated probability of lying in the given interval
    probability_close_to_mean = counter / num_samples

    # true probability of lying in the interval
    expected_probability_close_to_mean = norm.cdf(k) - norm.cdf(-k)

    # probability_close_to_mean is an estimation so let's make an assert with some wiggle room
    p = probability_close_to_mean  # probability of success of each counter
    counter_expected_std = np.sqrt(p * (1 - p)).max()  # consider the worse case
    estimation_confidence = 10  # increase the imprecision by this amount
    counter_imprecision = counter_expected_std / np.sqrt(num_samples)
    counter_imprecision = estimation_confidence * counter_imprecision

    # next line is basically testing if probability_close_to_mean > 0.9 if k = 3
    assert (probability_close_to_mean > expected_probability_close_to_mean - counter_imprecision).all()


def test_gp_sum_no_gp_list(gp_sum_setup):
    with pytest.raises(AssertionError):
        GaussianProcessSum([], gp_sum_setup["weights_lists"][0])


def test_gp_sum_not_match(gp_sum_setup):
    gp_list, weights = gp_sum_setup["gaussian_process_lists"][0], gp_sum_setup["weights_lists"][0]
    with pytest.raises(AssertionError):
        GaussianProcessSum([gp_list[0]], weights)
    with pytest.raises(AssertionError):
        GaussianProcessSum(gp_list, weights[:1])


def test_gp_sum_initialization(gp_sum_setup):
    gp_list = copy.deepcopy(gp_sum_setup["gaussian_process_lists"][0])
    weights = copy.deepcopy(gp_sum_setup["weights_lists"][0])
    gp_sum = GaussianProcessSum(gp_list, weights)
    assert len(gp_sum.gaussian_process_list) == len(gp_list)
    assert len(gp_sum.weights) == len(weights)


def test_gp_sum_reference_passing(gp_sum_setup):
    domain = gp_sum_setup["domains"][0]
    gp_list = copy.deepcopy(gp_sum_setup["gaussian_process_lists"][0])
    weights = copy.deepcopy(gp_sum_setup["weights_lists"][0])
    gp_sum = GaussianProcessSum(gp_list, weights)

    first_gp_num_sampled = gp_list[0].num_sampled
    gp_list[0].append_lie_data(domain.generate_quasi_random_points_in_domain(3))
    assert gp_list[0].num_sampled == first_gp_num_sampled + 3
    assert gp_sum.num_sampled == first_gp_num_sampled + 3
    assert gp_sum.gaussian_process_list[0].num_sampled == first_gp_num_sampled + 3

    weights[0] = weights[0] + 1
    assert not weights[0] == gp_sum_setup["weights_lists"][0][0]
    assert all(np.isclose(w, z) for w, z in zip(gp_sum.weights, weights))


def test_gp_sum_properties(gp_sum_setup):
    for gp_list, weights in zip(gp_sum_setup["gaussian_process_lists"], gp_sum_setup["weights_lists"]):
        gp_sum = GaussianProcessSum(gp_list, weights)
        for gp in gp_list:
            assert gp_sum.dim == gp.dim
            assert gp_sum.num_sampled == gp.num_sampled
        for gp_from_sum, gp_from_list in zip(gp_sum.gaussian_process_list, gp_list):
            assert gp_from_sum.dim == gp_from_list.dim
            assert gp_from_sum.num_sampled == gp_from_list.num_sampled
            assert np.allclose(gp_from_sum.points_sampled, gp_from_list.points_sampled)
            assert np.allclose(gp_from_sum.points_sampled_value, gp_from_list.points_sampled_value)
            assert np.allclose(
                gp_from_sum.points_sampled_noise_variance,
                gp_from_list.points_sampled_noise_variance,
            )
        assert len(gp_sum.points_sampled) == gp_sum.num_sampled
        assert len(gp_sum.points_sampled_value) == gp_sum.num_sampled
        assert len(gp_sum.points_sampled_noise_variance) == gp_sum.num_sampled
        assert all(np.isclose(w_sum, w) for w_sum, w in zip(gp_sum.weights, weights))


def test_gp_sum_append_lie_data(gp_sum_setup):
    for domain, gp_list, weights in zip(
        gp_sum_setup["domains"], gp_sum_setup["gaussian_process_lists"], gp_sum_setup["weights_lists"]
    ):
        gp_sum = GaussianProcessSum(copy.deepcopy(gp_list), weights)
        num_sampled = gp_sum.num_sampled

        num_new_points = 9
        x_append = domain.generate_quasi_random_points_in_domain(num_new_points)
        gp_sum.append_lie_data(x_append)

        assert gp_sum.num_sampled == num_sampled + num_new_points
        assert len(gp_sum.points_sampled) == num_sampled + num_new_points
        assert len(gp_sum.points_sampled_value) == num_sampled + num_new_points
        assert len(gp_sum.points_sampled_noise_variance) == num_sampled + num_new_points
        assert np.all(gp_sum.points_sampled_value[-num_new_points:] == max(gp_sum.points_sampled_value))


def test_gp_sum_best_value_and_location(gp_sum_setup):
    for gp_list, weights in zip(gp_sum_setup["gaussian_process_lists"], gp_sum_setup["weights_lists"]):
        best_values = np.zeros((gp_list[0].num_sampled, len(gp_list)))
        for i, gp in enumerate(gp_list):
            best_values[:, i] = gp.points_sampled_value
        combined_values = np.dot(best_values, weights)
        best_index = np.argmin(combined_values)
        expected_best_value = combined_values[best_index]
        expected_best_location = gp_list[0].points_sampled[best_index, :]

        gp_sum = GaussianProcessSum(gp_list, weights)
        best_value = gp_sum.best_observed_value
        best_location = gp_sum.best_observed_location
        assert_scalar_within_relative(best_value, expected_best_value, 1e-12)
        assert_vector_within_relative(best_location, expected_best_location, 1e-12)


@pytest.mark.flaky(reruns=2)
def test_gp_sum_by_sampling(gp_sum_setup):
    n_samples = 2111
    num_points = 55
    for domain, gp_list, weights in zip(
        gp_sum_setup["domains"], gp_sum_setup["gaussian_process_lists"], gp_sum_setup["weights_lists"]
    ):
        num_of_gps = len(gp_list)
        xt = domain.generate_quasi_random_points_in_domain(num_points)

        gp_sum = GaussianProcessSum(gp_list, weights)

        mean = gp_sum.compute_mean_of_points(xt)
        var = gp_sum.compute_variance_of_points(xt)
        stds = np.sqrt(var)

        all_samples = np.zeros((n_samples, num_points, num_of_gps))
        for i, gp in enumerate(gp_list):
            all_samples[:, :, i] = gp.draw_posterior_samples_of_points(n_samples, xt)
        samples = np.dot(all_samples, weights)

        samples_mean = samples.mean(axis=0)
        samples_var = samples.var(axis=0)
        gp_samples_credible_interval_test(samples, mean, stds)
        assert_vector_within_relative_norm(mean, samples_mean, 0.1)
        assert_vector_within_relative_norm(var, samples_var, 0.1)


@pytest.mark.flaky(reruns=2)
def test_gp_sum_draw_posterior_samples_of_points(gp_sum_setup):
    n_samples = 2212
    num_points = 65

    for domain, gp_list, weights in zip(
        gp_sum_setup["domains"], gp_sum_setup["gaussian_process_lists"], gp_sum_setup["weights_lists"]
    ):
        xt = domain.generate_quasi_random_points_in_domain(num_points)

        gp_sum = GaussianProcessSum(gp_list, weights)
        mean = gp_sum.compute_mean_of_points(xt)
        var = gp_sum.compute_variance_of_points(xt)
        stds = np.sqrt(var)

        samples = gp_sum.draw_posterior_samples_of_points(n_samples, xt)

        samples_mean = samples.mean(axis=0)
        samples_var = samples.var(axis=0)
        gp_samples_credible_interval_test(samples, mean, stds)
        assert_vector_within_relative_norm(mean, samples_mean, 0.1)
        assert_vector_within_relative_norm(var, samples_var, 0.1)


def test_gp_sum_core_functionality(gp_sum_setup):
    num_points = 51
    for domain, gp_list, weights in zip(
        gp_sum_setup["domains"], gp_sum_setup["gaussian_process_lists"], gp_sum_setup["weights_lists"]
    ):
        xt = domain.generate_quasi_random_points_in_domain(num_points)
        expected_results = compute_mixture_of_individual_gps_core_functionality(
            gp_list,
            weights,
            xt,
            option="all",
        )

        gp_sum = GaussianProcessSum(gp_list, weights)
        mean = gp_sum.compute_mean_of_points(xt)
        var = gp_sum.compute_variance_of_points(xt)
        grad_mean = gp_sum.compute_grad_mean_of_points(xt)
        grad_var = gp_sum.compute_grad_variance_of_points(xt)
        covariance = gp_sum.compute_covariance_of_points(xt)

        h = 1e-3
        check_gradient_with_finite_difference(
            xt,
            gp_sum.compute_mean_of_points,
            gp_sum.compute_grad_mean_of_points,
            tol=h,
            fd_step=h * np.ones(domain.dim),
        )
        check_gradient_with_finite_difference(
            xt,
            gp_sum.compute_variance_of_points,
            gp_sum.compute_grad_variance_of_points,
            tol=h,
            fd_step=h * np.ones(domain.dim),
        )

        assert_vector_within_relative_norm(expected_results.mean, mean, 1e-6)
        assert_vector_within_relative_norm(expected_results.var, var, 1e-6)
        assert_vector_within_relative_norm(expected_results.grad_mean, grad_mean, 1e-6)
        assert_vector_within_relative_norm(expected_results.grad_var, grad_var, 1e-6)
        assert_vector_within_relative_norm(expected_results.covariance, covariance, 1e-6)

        mean2, var2 = gp_sum.compute_mean_and_variance_of_points(xt)
        assert_vector_within_relative_norm(expected_results.mean, mean2, 1e-6)
        assert_vector_within_relative_norm(expected_results.var, var2, 1e-6)

        mean3, var3, grad_mean2, grad_var2 = gp_sum.compute_mean_variance_grad_of_points(xt)
        assert_vector_within_relative_norm(expected_results.mean, mean3, 1e-6)
        assert_vector_within_relative_norm(expected_results.var, var3, 1e-6)
        assert_vector_within_relative_norm(expected_results.grad_mean, grad_mean2, 1e-6)
        assert_vector_within_relative_norm(expected_results.grad_var, grad_var2, 1e-6)


@pytest.fixture(scope="module")
def build_acquisition_function():
    def _acquisition_function(acquisition_function_name, domain, predictor, points_being_sampled):
        if acquisition_function_name == "ei":
            return ExpectedImprovement(predictor)
        elif acquisition_function_name == "aei":
            return AugmentedExpectedImprovement(predictor)
        elif acquisition_function_name == "eiwf":
            threshold = np.random.random()
            return ExpectedImprovementWithFailures(predictor, ProbabilisticFailures(predictor, threshold))
        elif acquisition_function_name == "epi":
            return ExpectedParallelImprovement(predictor, 1, points_being_sampled=points_being_sampled)
        elif acquisition_function_name == "epiwf":
            threshold = np.random.random()
            list_of_pfs = []
            for _ in range(2):
                threshold = np.random.random()
                list_of_pfs.append(ProbabilisticFailures(predictor, threshold))
            ppf = ProductOfListOfProbabilisticFailures(list_of_pfs)
            return ExpectedParallelImprovementWithFailures(
                predictor,
                num_points_to_sample=1,
                failure_model=ppf,
                points_being_sampled=points_being_sampled,
            )
        else:
            return None

    return _acquisition_function


@pytest.mark.parametrize("acquisition_function_name", ["ei", "aei", "eiwf", "epi", "epiwf"])
def test_acquisition_functions(gp_sum_setup, build_acquisition_function, acquisition_function_name):
    h = 1e-4
    num_points = 22
    for domain, gp_list, weights in zip(
        gp_sum_setup["domains"], gp_sum_setup["gaussian_process_lists"], gp_sum_setup["weights_lists"]
    ):
        gp_sum = GaussianProcessSum(gp_list, weights)
        xt = domain.generate_quasi_random_points_in_domain(num_points)
        acquisition_function = build_acquisition_function(
            acquisition_function_name,
            domain,
            gp_sum,
            gp_sum.points_sampled,
        )
        acquisition_function_values = acquisition_function.evaluate_at_point_list(xt)
        assert len(xt) == len(acquisition_function_values)
        if acquisition_function.differentiable:
            acq_values, _ = acquisition_function.joint_function_gradient_eval(xt)
            assert_vector_within_relative_norm(acquisition_function_values, acq_values, 1e-6)
            check_gradient_with_finite_difference(
                xt,
                acquisition_function.evaluate_at_point_list,
                acquisition_function.evaluate_grad_at_point_list,
                tol=domain.dim * h,
                fd_step=h * np.ones(domain.dim),
            )
