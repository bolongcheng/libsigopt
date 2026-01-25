# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import numpy as np
import pytest
from testviews.zigopt_input_utils import form_points_sampled, form_random_hyperparameter_dict

from libsigopt.compute.covariance import SquareExponential
from libsigopt.compute.domain import CategoricalDomain
from libsigopt.compute.gaussian_process import GaussianProcess
from libsigopt.compute.misc.data_containers import HistoricalData, SingleMetricMidpointInfo
from libsigopt.compute.probabilistic_failures import (
    ProbabilisticFailures,
    ProbabilisticFailuresCDF,
    ProductOfListOfProbabilisticFailures,
)

from testcompute.gp_test_utils import (
    form_continous_and_uniform_domain,
    form_deterministic_gaussian_process,
    form_gaussian_process_and_data,
    form_list_of_probabilistic_failures,
)


@pytest.fixture(scope="session")
def domain_dim():
    return 3


@pytest.fixture(scope="session")
def continuous_domain(domain_dim):
    return CategoricalDomain(
        [
            {
                "var_type": "double",
                "elements": (-2.0, 2.0),
            }
        ]
        * domain_dim
    )


@pytest.fixture(scope="session")
def one_hot_domain(continuous_domain):
    return continuous_domain.one_hot_domain


@pytest.fixture(scope="session")
def historical_data(one_hot_domain):
    dim = one_hot_domain.dim
    num_sampled = 14
    x = one_hot_domain.generate_quasi_random_points_in_domain(num_sampled)
    y = np.sum(x**2, axis=1)
    v = np.full_like(y, 1e-3)
    data = HistoricalData(dim)
    data.append_historical_data(x, y, v)
    return data


@pytest.fixture(scope="session")
def gaussian_process(historical_data, one_hot_domain):
    hparam_dict = form_random_hyperparameter_dict(one_hot_domain)[0]
    length_scales = np.concatenate(hparam_dict["length_scales"]).tolist()
    cov = SquareExponential([hparam_dict["alpha"]] + length_scales)
    mpi = [[0] * one_hot_domain.dim]
    return GaussianProcess(cov, historical_data, mpi)


@pytest.fixture(scope="session")
def domain_list():
    dim_list = [1, 3, 5, 9]
    return [form_continous_and_uniform_domain(dim=dim, lower_element=-1, higher_element=1) for dim in dim_list]


@pytest.fixture(scope="session")
def one_hot_domain_list(domain_list):
    return [domain.one_hot_domain for domain in domain_list]


@pytest.fixture(scope="session")
def gaussian_process_list(domain_list):
    num_sampled_list = [4, 8, 15, 16]
    return [
        form_gaussian_process_and_data(domain, num_sampled=num_sampled)
        for domain, num_sampled in zip(domain_list, num_sampled_list)
    ]


@pytest.fixture(scope="session")
def probabilistic_failures_list(gaussian_process_list):
    pf_list = []
    for gp in gaussian_process_list:
        cov, data, mpi, tikhonov = gp.get_core_data_copy()
        data.points_sampled_values = np.sum((data.points_sampled - 0.2) ** 2, axis=1)
        threshold = np.random.uniform(0.2, 0.8)
        pf = ProbabilisticFailures(GaussianProcess(cov, data, mpi, tikhonov), threshold)
        pf_list.append(pf)
    return pf_list


@pytest.fixture(scope="session", params=[True, False])
def list_probabilistic_failures_list(request, domain_list):
    num_gps_list = [1, 2, 4, 8]
    num_sampled_list = [8, 14, 26, 48]
    return [
        form_list_of_probabilistic_failures(num_gp, domain, num_sampled, pf_cdf=request.param)
        for domain, num_sampled, num_gp in zip(domain_list, num_sampled_list, num_gps_list)
    ]


@pytest.fixture(scope="session")
def product_of_list_probabilistic_failures_list(list_probabilistic_failures_list):
    return [ProductOfListOfProbabilisticFailures(list_of_pfs) for list_of_pfs in list_probabilistic_failures_list]


@pytest.fixture(scope="session")
def any_domain_list():
    random_dim_list = sorted(np.random.choice(range(2, 25), 10, replace=False))
    return [form_continous_and_uniform_domain(dim=dim, lower_element=-1, higher_element=1) for dim in random_dim_list]


@pytest.fixture(scope="session")
def any_one_hot_domain_list(any_domain_list):
    return [domain.one_hot_domain for domain in any_domain_list]


@pytest.fixture(scope="session")
def any_gaussian_process_list(any_domain_list):
    return [
        form_gaussian_process_and_data(
            domain,
            num_sampled=14,
        )
        for domain in any_domain_list
    ]


@pytest.fixture(scope="session")
def gaussian_process_and_domain():
    dim = np.random.randint(1, 9)
    domain = form_continous_and_uniform_domain(dim)
    gaussian_process = form_gaussian_process_and_data(domain)
    return gaussian_process, domain


@pytest.fixture(scope="session")
def deterministic_gaussian_process():
    return form_deterministic_gaussian_process(dim=3, num_sampled=10)
