# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
"""Test cases for the covariance functions and any potential gradients."""

import inspect
import itertools

import numpy as np
import pytest

from libsigopt.compute import covariance
from libsigopt.compute.covariance_base import DifferentiableCovariance, HyperparameterInvalidError
from libsigopt.compute.misc.constant import TASK_LENGTH_LOWER_BOUND, CovarianceType
from libsigopt.compute.multitask_covariance import MultitaskTensorCovariance

from testaux.numerical_test_case import (
    assert_scalar_within_relative,
    assert_vector_within_relative,
    assert_vector_within_relative_norm,
    check_gradient_with_finite_difference,
)


@pytest.fixture(scope="module")
def covariance_data():
    all_covariance_bases = [
        f[1] for f in inspect.getmembers(covariance, inspect.isclass) if hasattr(f[1], "covariance_type")
    ]
    all_covariances = [cb for cb in all_covariance_bases]
    differentiable_covariances = [
        f for f in all_covariances if inspect.isclass(f) and issubclass(f, DifferentiableCovariance)
    ]

    num_tests = 10
    dim_array = np.random.randint(1, 11, (num_tests,)).tolist()
    num_points_array = np.random.randint(30, 80, (num_tests,)).tolist()
    test_z = []
    test_x = []
    test_hparams = []
    for dim, num_points in zip(dim_array, num_points_array):
        test_z.append(np.random.random((num_points, dim)))
        test_x.append(np.random.random((num_points - 10, dim)))
        test_hparams.append(np.random.random((dim + 1,)))

    return {
        "all_covariances": all_covariances,
        "differentiable_covariances": differentiable_covariances,
        "test_x": test_x,
        "test_z": test_z,
        "test_hparams": test_hparams,
    }


class TestCovariances:
    def test_covariance_symmetric(self, covariance_data):
        for x, z, hparams in zip(covariance_data["test_x"], covariance_data["test_z"], covariance_data["test_hparams"]):
            z_trunc = z[: len(x), :]
            for covariance_object in covariance_data["all_covariances"]:
                cov = covariance_object(hparams)
                kernel_at_xz = cov.covariance(x, z_trunc)
                kernel_at_zx = cov.covariance(z_trunc, x)
                assert_vector_within_relative(kernel_at_xz, kernel_at_zx, 1e-8)

    def test_grad_covariance_antisymmetric(self, covariance_data):
        for x, z, hparams in zip(covariance_data["test_x"], covariance_data["test_z"], covariance_data["test_hparams"]):
            xn, d = x.shape
            zn, _ = z.shape
            full_zx = np.tile(z, (xn, 1))
            full_xz = np.reshape(np.tile(x, (1, zn)), (xn * zn, d))
            for covariance_object in covariance_data["differentiable_covariances"]:
                cov = covariance_object(hparams)
                basic_gtensor = cov.grad_covariance(full_xz, full_zx)
                basic_gtensor_flipped = cov.grad_covariance(full_zx, full_xz)
                assert_vector_within_relative(
                    -basic_gtensor_flipped,
                    basic_gtensor,
                    1e-8,
                )

    @pytest.mark.flaky(reruns=1)
    def test_kernel_matrix_evaluation(self, covariance_data):
        for x, z, hparams in zip(covariance_data["test_x"], covariance_data["test_z"], covariance_data["test_hparams"]):
            xn, d = x.shape
            zn, _ = z.shape
            full_zx = np.tile(z, (xn, 1))
            full_xz = np.reshape(np.tile(x, (1, zn)), (xn * zn, d))
            full_zz = np.tile(z, (zn, 1))
            full_zz_trans = np.reshape(np.tile(z, (1, zn)), (zn * zn, d))
            # Note that, because of the faster computation that we use to compute kernel
            # matrices, it's possible the diagonal errors scale with sqrt(machine precision)
            # In reality, probably never gonna be close to this, but even this accuracy is much higher than need be
            for covariance_object in covariance_data["all_covariances"]:
                cov = covariance_object(hparams)
                standard_kernel_matrix = cov.build_kernel_matrix(z, x)
                basic_kernel_matrix = cov.covariance(full_xz, full_zx)
                assert_vector_within_relative_norm(
                    standard_kernel_matrix.reshape(-1),
                    basic_kernel_matrix,
                    d * xn * zn * 1e-8,
                )
                standard_symm_kernel_matrix = cov.build_kernel_matrix(z)
                basic_symm_kernel_matrix = cov.covariance(full_zz, full_zz_trans)
                assert_vector_within_relative_norm(
                    standard_symm_kernel_matrix.reshape(-1),
                    basic_symm_kernel_matrix,
                    d * zn * zn * 1e-8,
                )
                noise = np.full(zn, np.random.random() * 1e-4)
                standard_symmetric_noisy_kernel_matrix = cov.build_kernel_matrix(z, noise_variance=noise)
                kernel_diff = np.diag(standard_symmetric_noisy_kernel_matrix - standard_symm_kernel_matrix)
                assert_vector_within_relative(kernel_diff, noise, 1e-6)

    def test_kernel_gradient_tensor_evaluation(self, covariance_data):
        for x, z, hparams in zip(covariance_data["test_x"], covariance_data["test_z"], covariance_data["test_hparams"]):
            xn, d = x.shape
            zn, _ = z.shape
            full_zx = np.tile(z, (xn, 1))
            full_xz = np.reshape(np.tile(x, (1, zn)), (xn * zn, d))
            full_zz = np.tile(z, (zn, 1))
            full_zz_trans = np.reshape(np.tile(z, (1, zn)), (zn * zn, d))
            for covariance_object in covariance_data["differentiable_covariances"]:
                cov = covariance_object(hparams)
                standard_gtensor = cov.build_kernel_grad_tensor(z, x)
                basic_gtensor = cov.grad_covariance(full_xz, full_zx)
                standard_gtensor_flat = np.array([standard_gtensor[:, :, i].reshape(-1) for i in range(d)]).T
                assert_vector_within_relative_norm(
                    standard_gtensor_flat,
                    basic_gtensor,
                    d * xn * zn * 1e-8,
                )
                standard_sgtensor = cov.build_kernel_grad_tensor(z)
                basic_sgtensor = cov.grad_covariance(full_zz_trans, full_zz)
                standard_sgtensor_flat = np.array([standard_sgtensor[:, :, i].reshape(-1) for i in range(d)]).T
                assert_vector_within_relative_norm(
                    standard_sgtensor_flat,
                    basic_sgtensor,
                    d * zn * zn * 1e-8,
                )

    def test_kernel_gradient_against_finite_difference(self, covariance_data):
        for x, z, hparams in zip(covariance_data["test_x"], covariance_data["test_z"], covariance_data["test_hparams"]):
            xn, d = x.shape
            zn, _ = z.shape
            n = min(xn, zn)
            for covariance_object in covariance_data["differentiable_covariances"]:
                cov = covariance_object(hparams)
                for i in range(n):
                    func = lambda u: cov.covariance(u, np.reshape(z[i, :], (1, -1)))
                    grad = lambda u: cov.grad_covariance(u, np.reshape(z[i, :], (1, -1)))
                    h = 1e-8
                    check_gradient_with_finite_difference(
                        np.reshape(x[i, :], (1, -1)),
                        func,
                        grad,
                        tol=d * 2e-6,
                        fd_step=h * np.ones(d),
                        use_complex=True,
                    )

    def test_kernel_hyperparameter_gradient_against_finite_difference(self, covariance_data):
        for x, z, hparams in zip(covariance_data["test_x"], covariance_data["test_z"], covariance_data["test_hparams"]):
            xn, d = x.shape
            zn, _ = z.shape
            n = min(xn, zn)
            x = x[:n, :]
            z = z[:n, :]
            for covariance_object in covariance_data["differentiable_covariances"]:

                def func(hparams):
                    cov = covariance_object(hparams.squeeze())
                    return cov.covariance(x, z)

                def grad(hparams):
                    cov = covariance_object(hparams.squeeze())
                    return cov.hyperparameter_grad_covariance(x, z)

                check_gradient_with_finite_difference(
                    np.reshape(hparams, (1, -1)),
                    func,
                    grad,
                    tol=d * n * n * 1e-6,
                    fd_step=1e-8 * hparams,
                )

    def test_kernel_hyperparameter_gradient_tensor_evaluation(self, covariance_data):
        for x, z, hparams in zip(covariance_data["test_x"], covariance_data["test_z"], covariance_data["test_hparams"]):
            xn, d = x.shape
            zn, _ = z.shape
            full_zx = np.tile(z, (xn, 1))
            full_xz = np.reshape(np.tile(x, (1, zn)), (xn * zn, d))
            full_zz = np.tile(z, (zn, 1))
            full_zz_trans = np.reshape(np.tile(z, (1, zn)), (zn * zn, d))
            for covariance_object in covariance_data["differentiable_covariances"]:
                cov = covariance_object(hparams)
                dh = cov.num_hyperparameters
                standard_hgtensor = cov.build_kernel_hparam_grad_tensor(z, x)
                basic_hgtensor = cov.hyperparameter_grad_covariance(full_xz, full_zx)
                standard_hgtensor_flat = np.array([standard_hgtensor[:, :, i].reshape(-1) for i in range(dh)]).T
                assert_vector_within_relative_norm(
                    standard_hgtensor_flat,
                    basic_hgtensor,
                    dh * xn * zn * 1e-8,
                )
                standard_shgtensor = cov.build_kernel_hparam_grad_tensor(z)
                basic_shgtensor = cov.hyperparameter_grad_covariance(full_zz_trans, full_zz)
                standard_shgtensor_flat = np.array([standard_shgtensor[:, :, i].reshape(-1) for i in range(dh)]).T
                assert_vector_within_relative_norm(
                    standard_shgtensor_flat,
                    basic_shgtensor,
                    dh * zn * zn * 1e-8,
                )


class TestMultitaskCovariance:
    tasks = [0.1, 0.3, 1.0]

    def test_covariance_creation(self):
        with pytest.raises(AssertionError):
            MultitaskTensorCovariance([1.0, 1.2, 1.4], covariance.C0RadialMatern, covariance.C2RadialMatern)
        with pytest.raises(AssertionError):
            MultitaskTensorCovariance([1.0, 1.2], covariance.C2RadialMatern, covariance.C2RadialMatern)
        with pytest.raises(HyperparameterInvalidError):
            MultitaskTensorCovariance([1.0, 1.2, -1.4], covariance.C2RadialMatern, covariance.C2RadialMatern)
        with pytest.raises(HyperparameterInvalidError):
            MultitaskTensorCovariance([1.0, 1.2, np.nan], covariance.C2RadialMatern, covariance.C2RadialMatern)
        with pytest.raises(HyperparameterInvalidError):
            MultitaskTensorCovariance([1.0, 1.2, None], covariance.C2RadialMatern, covariance.C2RadialMatern)
        with pytest.raises(HyperparameterInvalidError):
            MultitaskTensorCovariance([1.0, 1.2, np.inf], covariance.C2RadialMatern, covariance.C2RadialMatern)

        cov = MultitaskTensorCovariance([2.3, 4.5, 6.7], covariance.SquareExponential, covariance.C2RadialMatern)
        assert np.all(cov.hyperparameters == [2.3, 4.5, 6.7])
        assert cov.process_variance == 2.3
        assert cov.dim == 2
        assert cov.physical_covariance is not None
        assert np.all(cov.physical_covariance.hyperparameters == [1.0, 4.5])
        assert cov.process_variance == 2.3  # cov.physical_covariance.process_variance is 1.0, cov's is 2.3
        assert cov.physical_covariance.dim == 1
        assert cov.task_covariance is not None
        assert np.all(cov.task_covariance.hyperparameters == [1.0, 6.7])
        assert cov.task_covariance.process_variance == 1.0
        assert cov.task_covariance.dim == 1

        cov.hyperparameters = [1.2, 3.4, 5.6, 7.8]
        assert np.all(cov.hyperparameters == [1.2, 3.4, 5.6, 7.8])
        assert cov.physical_covariance.dim == 2
        assert cov.task_covariance.dim == 1

    def test_covariance_symmetric(self, covariance_data):
        for x, z, hparams in zip(covariance_data["test_x"], covariance_data["test_z"], covariance_data["test_hparams"]):
            z_trunc = z[: len(x), :]
            x = np.concatenate((x, np.random.choice(self.tasks, size=(len(x), 1))), axis=1)
            z = np.concatenate((z_trunc, np.random.choice(self.tasks, size=(len(z_trunc), 1))), axis=1)
            for c1, c2 in itertools.product(
                covariance_data["differentiable_covariances"], covariance_data["differentiable_covariances"]
            ):
                hparams_with_task = np.append(hparams, np.random.random())
                cov = MultitaskTensorCovariance(hparams_with_task, c1, c2)
                kernel_at_xz = cov.covariance(x=x, z=z)
                kernel_at_zx = cov.covariance(x=z, z=x)
                assert_vector_within_relative(kernel_at_xz, kernel_at_zx, 1e-8)

    def test_kernel_matrix_evaluation(self, covariance_data):
        for x, z, hparams in zip(covariance_data["test_x"], covariance_data["test_z"], covariance_data["test_hparams"]):
            x = np.concatenate((x, np.random.choice(self.tasks, size=(len(x), 1))), axis=1)
            z = np.concatenate((z, np.random.choice(self.tasks, size=(len(z), 1))), axis=1)
            xn, d = x.shape
            zn, _ = z.shape
            full_zx = np.tile(z, (xn, 1))
            full_xz = np.reshape(np.tile(x, (1, zn)), (xn * zn, d))
            full_zz = np.tile(z, (zn, 1))
            full_zz_trans = np.reshape(np.tile(z, (1, zn)), (zn * zn, d))
            for c1, c2 in itertools.product(
                covariance_data["differentiable_covariances"], covariance_data["differentiable_covariances"]
            ):
                hparams_with_task = np.append(hparams, np.random.random())
                cov = MultitaskTensorCovariance(hparams_with_task, c1, c2)
                standard_kernel_matrix = cov.build_kernel_matrix(z, x)
                basic_kernel_matrix = cov.covariance(full_xz, full_zx)
                assert_vector_within_relative_norm(
                    standard_kernel_matrix.reshape(-1),
                    basic_kernel_matrix,
                    xn * zn * 1e-8,
                )
                standard_symm_kernel_matrix = cov.build_kernel_matrix(z)
                basic_symm_kernel_matrix = cov.covariance(full_zz, full_zz_trans)
                assert_vector_within_relative_norm(
                    standard_symm_kernel_matrix.reshape(-1),
                    basic_symm_kernel_matrix,
                    zn * zn * 1e-8,
                )
                noise = np.full(zn, np.random.random() * 1e-4)
                standard_symmetric_noisy_kernel_matrix = cov.build_kernel_matrix(z, noise_variance=noise)
                kernel_diff = np.diag(standard_symmetric_noisy_kernel_matrix - standard_symm_kernel_matrix)
                assert_vector_within_relative(kernel_diff, noise, 1e-6)

    def test_kernel_gradient_tensor_evaluation(self, covariance_data):
        for x, z, hparams in zip(covariance_data["test_x"], covariance_data["test_z"], covariance_data["test_hparams"]):
            x = np.concatenate((x, np.random.choice(self.tasks, size=(len(x), 1))), axis=1)
            z = np.concatenate((z, np.random.choice(self.tasks, size=(len(z), 1))), axis=1)
            xn, d = x.shape
            zn, _ = z.shape
            full_zx = np.tile(z, (xn, 1))
            full_xz = np.reshape(np.tile(x, (1, zn)), (xn * zn, d))
            full_zz = np.tile(z, (zn, 1))
            full_zz_trans = np.reshape(np.tile(z, (1, zn)), (zn * zn, d))
            for c1, c2 in itertools.product(
                covariance_data["differentiable_covariances"], covariance_data["differentiable_covariances"]
            ):
                hparams_with_task = np.append(hparams, np.random.random())
                cov = MultitaskTensorCovariance(hparams_with_task, c1, c2)
                standard_gtensor = cov.build_kernel_grad_tensor(z, x)
                basic_gtensor = cov.grad_covariance(full_xz, full_zx)
                standard_gtensor_flat = np.array([standard_gtensor[:, :, i].reshape(-1) for i in range(d)]).T
                assert_vector_within_relative_norm(
                    standard_gtensor_flat,
                    basic_gtensor,
                    d * xn * zn * 1e-8,
                )
                standard_sgtensor = cov.build_kernel_grad_tensor(z)
                basic_sgtensor = cov.grad_covariance(full_zz_trans, full_zz)
                standard_sgtensor_flat = np.array([standard_sgtensor[:, :, i].reshape(-1) for i in range(d)]).T
                assert_vector_within_relative_norm(
                    standard_sgtensor_flat,
                    basic_sgtensor,
                    d * zn * zn * 1e-8,
                )

    def test_kernel_gradient_against_finite_difference(self, covariance_data):
        for x, z, hparams in zip(covariance_data["test_x"], covariance_data["test_z"], covariance_data["test_hparams"]):
            x = np.concatenate((x, np.random.choice(self.tasks, size=(len(x), 1))), axis=1)
            z = np.concatenate((z, np.random.choice(self.tasks, size=(len(z), 1))), axis=1)
            xn, d = x.shape
            zn, _ = z.shape
            n = min(xn, zn)
            for c1, c2 in itertools.product(
                covariance_data["differentiable_covariances"], covariance_data["differentiable_covariances"]
            ):
                hparams_with_task = np.append(hparams, max(np.random.random(), TASK_LENGTH_LOWER_BOUND))
                cov = MultitaskTensorCovariance(hparams_with_task, c1, c2)
                for i in range(n):
                    func = lambda u: cov.covariance(u, np.reshape(z[i, :], (1, -1)))
                    grad = lambda u: cov.grad_covariance(u, np.reshape(z[i, :], (1, -1)))
                    h = 1e-8
                    check_gradient_with_finite_difference(
                        np.reshape(x[i, :], (1, -1)),
                        func,
                        grad,
                        tol=d * 1e-6,
                        fd_step=h * np.ones(d),
                        use_complex=True,
                    )

    def test_kernel_hyperparameter_gradient_against_finite_difference(self, covariance_data):
        for x, z, hparams in zip(covariance_data["test_x"], covariance_data["test_z"], covariance_data["test_hparams"]):
            x = np.concatenate((x, np.random.choice(self.tasks, size=(len(x), 1))), axis=1)
            z = np.concatenate((z, np.random.choice(self.tasks, size=(len(z), 1))), axis=1)
            xn, d = x.shape
            zn, _ = z.shape
            n = min(xn, zn)
            x = x[:n, :]
            z = z[:n, :]
            for c1, c2 in itertools.product(
                covariance_data["differentiable_covariances"], covariance_data["differentiable_covariances"]
            ):
                hparams_with_task = np.append(hparams, np.random.random())

                def func(hparams):
                    cov = MultitaskTensorCovariance(hparams.squeeze(), c1, c2)
                    return cov.covariance(x, z)

                def grad(hparams):
                    cov = MultitaskTensorCovariance(hparams.squeeze(), c1, c2)
                    return cov.hyperparameter_grad_covariance(x, z)

                check_gradient_with_finite_difference(
                    np.reshape(hparams_with_task, (1, -1)),
                    func,
                    grad,
                    tol=d * n * n * 1e-6,
                    fd_step=1e-8 * hparams_with_task,
                )

    def test_kernel_hyperparameter_gradient_tensor_evaluation(self, covariance_data):
        for x, z, hparams in zip(covariance_data["test_x"], covariance_data["test_z"], covariance_data["test_hparams"]):
            x = np.concatenate((x, np.random.choice(self.tasks, size=(len(x), 1))), axis=1)
            z = np.concatenate((z, np.random.choice(self.tasks, size=(len(z), 1))), axis=1)
            xn, d = x.shape
            zn, _ = z.shape
            full_zx = np.tile(z, (xn, 1))
            full_xz = np.reshape(np.tile(x, (1, zn)), (xn * zn, d))
            full_zz = np.tile(z, (zn, 1))
            full_zz_trans = np.reshape(np.tile(z, (1, zn)), (zn * zn, d))
            for c1, c2 in itertools.product(
                covariance_data["differentiable_covariances"], covariance_data["differentiable_covariances"]
            ):
                hparams_with_task = np.append(hparams, np.random.random())
                cov = MultitaskTensorCovariance(hparams_with_task, c1, c2)
                dh = cov.num_hyperparameters
                standard_hgtensor = cov.build_kernel_hparam_grad_tensor(z, x)
                basic_hgtensor = cov.hyperparameter_grad_covariance(full_xz, full_zx)
                standard_hgtensor_flat = np.array([standard_hgtensor[:, :, i].reshape(-1) for i in range(dh)]).T
                assert_vector_within_relative_norm(
                    standard_hgtensor_flat,
                    basic_hgtensor,
                    dh * xn * zn * 1e-8,
                )
                standard_shgtensor = cov.build_kernel_hparam_grad_tensor(z)
                basic_shgtensor = cov.hyperparameter_grad_covariance(full_zz_trans, full_zz)
                standard_shgtensor_flat = np.array([standard_shgtensor[:, :, i].reshape(-1) for i in range(dh)]).T
                assert_vector_within_relative_norm(
                    standard_shgtensor_flat,
                    basic_shgtensor,
                    dh * zn * zn * 1e-8,
                )


# NOTE: The min/max values are chosen to avoid overflow ... in reality _scale_difference_matrix is unneeded
@pytest.mark.parametrize("dim", [1, 3, 5])
@pytest.mark.parametrize("num_points", [1, 4, 10])
def test_scale_difference_matrix2(dim, num_points):
    def _generate_random_scale_and_difference_matrix(dim, num_points):
        scale = np.random.uniform(-1000, 1000, size=(num_points, num_points))
        difference_matrix = np.random.uniform(-1000, 1000, size=(num_points, num_points, dim))
        return scale, difference_matrix

    scale, difference_matrix = _generate_random_scale_and_difference_matrix(dim, num_points)
    scaled = covariance._scale_difference_matrix(scale, difference_matrix)
    for i in range(dim):
        np.testing.assert_array_equal(scaled[:, :, i], scale * difference_matrix[:, :, i])


class TestLinkers(object):
    """Tests that linkers contain all possible types defined in constants."""

    def test_covariance_links_have_all_covariance_types(self):
        """Test each covariance type is in a linker, and every linker key is a covariance type."""
        assert set(CovarianceType) == set(covariance.COVARIANCE_TYPES_TO_CLASSES.keys())
