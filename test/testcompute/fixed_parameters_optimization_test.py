# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0

import numpy as np
import pytest

from libsigopt.compute.covariance import C2RadialMatern, SquareExponential
from libsigopt.compute.domain import CategoricalDomain, FixedIndicesOnContinuousDomain
from libsigopt.compute.expected_improvement import AugmentedExpectedImprovement, ExpectedImprovement
from libsigopt.compute.gaussian_process import GaussianProcess
from libsigopt.compute.misc.data_containers import HistoricalData
from libsigopt.compute.multitask_acquisition_function import MultitaskAcquisitionFunction
from libsigopt.compute.multitask_covariance import MultitaskTensorCovariance
from libsigopt.compute.vectorized_optimizers import AdamOptimizer, DEOptimizer

from testaux.numerical_test_case import assert_vector_within_relative
from testcompute.vectorized_optimizers_test import QuadraticFunction


class TestVectorizedOptimizersWithFixedParameters:
    def test_basic_optimization(self):
        cat_domain = CategoricalDomain([{"var_type": "double", "elements": (-2, 2)}] * 5)
        fixed_indices = {0: 1, 3: -1}
        domain = FixedIndicesOnContinuousDomain(cat_domain.one_hot_domain, fixed_indices)

        true_sol = np.zeros(5)
        af = QuadraticFunction(domain, true_sol)
        optimizer = DEOptimizer(acquisition_function=af, domain=domain, num_multistarts=30, maxiter=100)
        best_solution, _ = optimizer.optimize(np.atleast_2d(true_sol))

        fixed_sol_full = np.array([1, 0, 0, -1, 0])
        assert_vector_within_relative(best_solution, fixed_sol_full, 1e-7)

    def test_constrained_optimization(self):
        cat_domain = CategoricalDomain(
            domain_components=[{"var_type": "double", "elements": (-2, 2)}] * 5,
            constraint_list=[
                {
                    "rhs": 1,
                    "var_type": "double",
                    "weights": [0, 1, 1, 0, 1],
                },
            ],
        )

        fixed_indices = {0: 1, 3: -1}
        domain = FixedIndicesOnContinuousDomain(cat_domain.one_hot_domain, fixed_indices)

        true_sol = np.ones(5) * 0.5
        af = QuadraticFunction(domain, true_sol)
        optimizer = DEOptimizer(acquisition_function=af, domain=domain, num_multistarts=30, maxiter=100)
        best_solution, _ = optimizer.optimize(np.atleast_2d(true_sol))

        fixed_sol_full = np.array([1, 0.5, 0.5, -1, 0.5])
        assert_vector_within_relative(best_solution, fixed_sol_full, 1e-7)


class TestAcquisitionFunctionWithFixedParameters:
    @pytest.fixture(scope="class")
    def setup_data(self):
        domain = CategoricalDomain(
            [
                {"var_type": "double", "elements": (-2, 3)},
                {"var_type": "double", "elements": (-1, 1)},
                {"var_type": "double", "elements": (0.1, 1.0)},
            ]
        ).one_hot_domain

        list_hyps = [1.0, 0.3, 0.3, 0.4]
        cov = SquareExponential(list_hyps)
        mtcov = MultitaskTensorCovariance(list_hyps, C2RadialMatern, SquareExponential)

        x = domain.generate_quasi_random_points_in_domain(14)
        y = np.sum(x**2, axis=1)
        v = np.full_like(y, 1e-3)
        data = HistoricalData(domain.dim)
        data.append_historical_data(x, y, v)

        mpi = [[0] * domain.dim]
        gp = GaussianProcess(cov, data, mpi)
        return {
            "domain": domain,
            "cov": cov,
            "mtcov": mtcov,
            "data": data,
            "mpi": mpi,
            "gp": gp,
        }

    @pytest.mark.parametrize("acquisition_function", [ExpectedImprovement, AugmentedExpectedImprovement])
    @pytest.mark.parametrize("optimizer_class", [DEOptimizer, AdamOptimizer])
    def test_fixed_parameter_af_evaluation(self, setup_data, acquisition_function, optimizer_class):
        domain = setup_data["domain"]
        data = setup_data["data"]
        mtcov = setup_data["mtcov"]
        mpi = setup_data["mpi"]
        fixed_values = domain.generate_quasi_random_points_in_domain(1)[0]
        fixed_indices = {domain.dim - 1: fixed_values[-1]}
        fixed_domain = FixedIndicesOnContinuousDomain(domain, fixed_indices)
        mtaf = MultitaskAcquisitionFunction(acquisition_function(GaussianProcess(mtcov, data, mpi)))
        opt = optimizer_class(
            acquisition_function=mtaf,
            domain=fixed_domain,
            num_multistarts=50,
            maxiter=100,
        )
        best_location, _ = opt.optimize()
        assert domain.check_point_acceptable(best_location)
        assert best_location[-1] == fixed_values[-1]
