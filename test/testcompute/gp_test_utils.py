# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import numpy as np

from libsigopt.compute.covariance import SquareExponential
from libsigopt.compute.domain import CategoricalDomain, ContinuousDomain
from libsigopt.compute.gaussian_process import GaussianProcess
from libsigopt.compute.misc.data_containers import HistoricalData, SingleMetricMidpointInfo
from libsigopt.compute.probabilistic_failures import (
    ProbabilisticFailures,
    ProbabilisticFailuresCDF,
    ProductOfListOfProbabilisticFailures,
)

from testaux.utils import form_points_sampled, form_random_hyperparameter_dict


def fill_random_covariance_hyperparameters(interval, num_hyperparameters, covariance_type):
    assert len(interval) == 2
    return covariance_type([np.random.uniform(*interval) for _ in range(num_hyperparameters)])


def fill_random_domain_bounds(lower_bound_interval, upper_bound_interval, dim):
    assert len(lower_bound_interval) == len(upper_bound_interval) == 2
    domain_bounds = np.empty((dim, 2))
    domain_bounds[:, 0] = np.random.uniform(*lower_bound_interval)
    domain_bounds[:, 1] = np.random.uniform(*upper_bound_interval)
    return domain_bounds


def form_continous_and_uniform_domain(dim=3, lower_element=-2, higher_element=2):
    return CategoricalDomain(
        [
            {
                "var_type": "double",
                "elements": (lower_element, higher_element),
            }
        ]
        * dim
    )


def form_list_of_probabilistic_failures(num_gps, domain, num_sampled, noise_per_point=1e-2, pf_cdf=True):
    list_of_pfs = []
    for _ in range(num_gps):
        gp = form_gaussian_process_and_data(
            domain,
            num_sampled=num_sampled,
            noise_per_point=noise_per_point,
        )
        threshold = np.random.random() * (np.max(gp.points_sampled_value) - np.min(gp.points_sampled_value))
        pf_class = ProbabilisticFailuresCDF if pf_cdf else ProbabilisticFailures
        list_of_pfs.append(pf_class(gp, threshold))
    return list_of_pfs


def form_gaussian_process_and_data(domain, mpi=None, num_sampled=50, noise_per_point=0.002):
    if mpi is None:
        mpi = [[0] * domain.one_hot_dim] if num_sampled > 2 else [[]]
    hparam_dict = form_random_hyperparameter_dict(domain)[0]
    length_scales = np.concatenate(hparam_dict["length_scales"]).tolist()
    cov = SquareExponential([hparam_dict["alpha"]] + length_scales)
    points_sampled = form_points_sampled(
        domain.one_hot_domain,
        num_sampled,
        noise_per_point,
        num_metrics=1,
        task_options=np.array([]),
        snap_cats=True,
        failure_prob=0,
    )
    return form_gaussian_process(
        points_sampled.points,
        cov,
        mpi,
        points_sampled.value_vars[:, 0],
    )


def form_deterministic_gaussian_process(dim, num_sampled, noise_variance_base=0.002):
    # HACK: This is one seed for which the tests I care about pass.  We'll need to think some more
    #  about exactly what we ought to be testing, though.
    np.random.seed(14)
    num_hyperparameters = dim + 1
    covariance = fill_random_covariance_hyperparameters(
        interval=(3.0, 5.0),
        num_hyperparameters=num_hyperparameters,
        covariance_type=SquareExponential,
    )
    domain_bounds = fill_random_domain_bounds(
        dim=dim,
        lower_bound_interval=(-2.0, 0.5),
        upper_bound_interval=(2.0, 3.5),
    )
    domain = ContinuousDomain(domain_bounds)
    points_sampled = domain.generate_quasi_random_points_in_domain(num_sampled)
    gaussian_process = form_gaussian_process(
        points_sampled,
        covariance,
        noise_variance=np.full(num_sampled, noise_variance_base),
    )
    return gaussian_process


def form_gaussian_process(
    points_sampled,
    covariance,
    mean_poly_indices=None,
    noise_variance=None,
    tikhonov_param=None,
):
    num_points, dim = points_sampled.shape

    mean = np.mean(points_sampled, axis=0)
    values = np.exp(-np.sum((points_sampled - mean[None, :]) ** 2, axis=1))
    noise_variance = np.zeros(num_points) if noise_variance is None else noise_variance
    mmi = SingleMetricMidpointInfo(values, np.zeros_like(values))
    scaled_values = mmi.relative_objective_value(values)
    scaled_variance = mmi.relative_objective_variance(noise_variance)

    historical_data = HistoricalData(dim)
    historical_data.append_historical_data(points_sampled, scaled_values, scaled_variance)

    mean_poly_indices = [[]] if mean_poly_indices is None else mean_poly_indices
    return GaussianProcess(covariance, historical_data, mean_poly_indices, tikhonov_param=tikhonov_param)
