# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
"""Test cases for the covariance functions and any potential gradients."""

import inspect

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


ALL_COVARIANCE_CLASSES = [
    f[1] for f in inspect.getmembers(covariance, inspect.isclass) if hasattr(f[1], "covariance_type")
]
DIFF_COVARIANCE_CLASSES = [f for f in ALL_COVARIANCE_CLASSES if issubclass(f, DifferentiableCovariance)]


@pytest.fixture(scope="session")
def covariance_test_samples():
    num_tests = 10
    dim_array = np.random.randint(1, 11, (num_tests,)).tolist()
    num_points_array = np.random.randint(30, 80, (num_tests,)).tolist()
    samples = []
    for dim, num_points in zip(dim_array, num_points_array):
        samples.append(
            {
                "z": np.random.random((num_points, dim)),
                "x": np.random.random((num_points - 10, dim)),
                "hparams": np.random.random((dim + 1,)),
            }
        )
    return samples


@pytest.mark.parametrize("covariance_object", ALL_COVARIANCE_CLASSES)
@pytest.mark.parametrize("sample_idx", range(10))
def test_covariance_symmetric(covariance_test_samples, covariance_object, sample_idx):
    sample = covariance_test_samples[sample_idx]
    x, z, hparams = sample["x"], sample["z"], sample["hparams"]
    z_trunc = z[: len(x), :]
    cov = covariance_object(hparams)
    kernel_at_xz = cov.covariance(x, z_trunc)
    kernel_at_zx = cov.covariance(z_trunc, x)
    assert_vector_within_relative(kernel_at_xz, kernel_at_zx, 1e-8)


@pytest.mark.parametrize("covariance_object", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("sample_idx", range(10))
def test_grad_covariance_antisymmetric(covariance_test_samples, covariance_object, sample_idx):
    sample = covariance_test_samples[sample_idx]
    x, z, hparams = sample["x"], sample["z"], sample["hparams"]
    xn, d = x.shape
    zn, _ = z.shape
    full_zx = np.tile(z, (xn, 1))
    full_xz = np.reshape(np.tile(x, (1, zn)), (xn * zn, d))
    cov = covariance_object(hparams)
    basic_gtensor = cov.grad_covariance(full_xz, full_zx)
    basic_gtensor_flipped = cov.grad_covariance(full_zx, full_xz)
    assert_vector_within_relative(
        -basic_gtensor_flipped,
        basic_gtensor,
        1e-8,
    )


@pytest.mark.flaky(reruns=1)
@pytest.mark.parametrize("covariance_object", ALL_COVARIANCE_CLASSES)
@pytest.mark.parametrize("sample_idx", range(10))
def test_kernel_matrix_evaluation(covariance_test_samples, covariance_object, sample_idx):
    sample = covariance_test_samples[sample_idx]
    x, z, hparams = sample["x"], sample["z"], sample["hparams"]
    xn, d = x.shape
    zn, _ = z.shape
    full_zx = np.tile(z, (xn, 1))
    full_xz = np.reshape(np.tile(x, (1, zn)), (xn * zn, d))
    full_zz = np.tile(z, (zn, 1))
    full_zz_trans = np.reshape(np.tile(z, (1, zn)), (zn * zn, d))
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


@pytest.mark.parametrize("covariance_object", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("sample_idx", range(10))
def test_kernel_gradient_tensor_evaluation(covariance_test_samples, covariance_object, sample_idx):
    sample = covariance_test_samples[sample_idx]
    x, z, hparams = sample["x"], sample["z"], sample["hparams"]
    xn, d = x.shape
    zn, _ = z.shape
    full_zx = np.tile(z, (xn, 1))
    full_xz = np.reshape(np.tile(x, (1, zn)), (xn * zn, d))
    full_zz = np.tile(z, (zn, 1))
    full_zz_trans = np.reshape(np.tile(z, (1, zn)), (zn * zn, d))
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


@pytest.mark.parametrize("covariance_object", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("sample_idx", range(10))
def test_kernel_gradient_against_finite_difference(covariance_test_samples, covariance_object, sample_idx):
    sample = covariance_test_samples[sample_idx]
    x, z, hparams = sample["x"], sample["z"], sample["hparams"]
    xn, d = x.shape
    zn, _ = z.shape
    n = min(xn, zn)
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


@pytest.mark.parametrize("covariance_object", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("sample_idx", range(10))
def test_kernel_hyperparameter_gradient_against_finite_difference(
    covariance_test_samples, covariance_object, sample_idx
):
    sample = covariance_test_samples[sample_idx]
    x, z, hparams = sample["x"], sample["z"], sample["hparams"]
    xn, d = x.shape
    zn, _ = z.shape
    n = min(xn, zn)
    x = x[:n, :]
    z = z[:n, :]

    def func(hparams_in):
        cov = covariance_object(hparams_in.squeeze())
        return cov.covariance(x, z)

    def grad(hparams_in):
        cov = covariance_object(hparams_in.squeeze())
        return cov.hyperparameter_grad_covariance(x, z)

    check_gradient_with_finite_difference(
        np.reshape(hparams, (1, -1)),
        func,
        grad,
        tol=d * n * n * 1e-6,
        fd_step=1e-8 * hparams,
    )


@pytest.mark.parametrize("covariance_object", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("sample_idx", range(10))
def test_kernel_hyperparameter_gradient_tensor_evaluation(covariance_test_samples, covariance_object, sample_idx):
    sample = covariance_test_samples[sample_idx]
    x, z, hparams = sample["x"], sample["z"], sample["hparams"]
    xn, d = x.shape
    zn, _ = z.shape
    full_zx = np.tile(z, (xn, 1))
    full_xz = np.reshape(np.tile(x, (1, zn)), (xn * zn, d))
    full_zz = np.tile(z, (zn, 1))
    full_zz_trans = np.reshape(np.tile(z, (1, zn)), (zn * zn, d))
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


MULTITASK_TASKS = [0.1, 0.3, 1.0]


def test_multitask_covariance_creation():
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
    assert cov.process_variance == 2.3
    assert cov.physical_covariance.dim == 1
    assert cov.task_covariance is not None
    assert np.all(cov.task_covariance.hyperparameters == [1.0, 6.7])
    assert cov.task_covariance.process_variance == 1.0
    assert cov.task_covariance.dim == 1

    cov.hyperparameters = [1.2, 3.4, 5.6, 7.8]
    assert np.all(cov.hyperparameters == [1.2, 3.4, 5.6, 7.8])
    assert cov.physical_covariance.dim == 2
    assert cov.task_covariance.dim == 1


@pytest.mark.parametrize("c1", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("c2", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("sample_idx", range(10))
def test_multitask_covariance_symmetric(covariance_test_samples, c1, c2, sample_idx):
    sample = covariance_test_samples[sample_idx]
    x, z, hparams = sample["x"], sample["z"], sample["hparams"]
    z_trunc = z[: len(x), :]
    x = np.concatenate((x, np.random.choice(MULTITASK_TASKS, size=(len(x), 1))), axis=1)
    z = np.concatenate((z_trunc, np.random.choice(MULTITASK_TASKS, size=(len(z_trunc), 1))), axis=1)
    hparams_with_task = np.append(hparams, np.random.random())
    cov = MultitaskTensorCovariance(hparams_with_task, c1, c2)
    kernel_at_xz = cov.covariance(x=x, z=z)
    kernel_at_zx = cov.covariance(x=z, z=x)
    assert_vector_within_relative(kernel_at_xz, kernel_at_zx, 1e-8)


@pytest.mark.parametrize("c1", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("c2", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("sample_idx", range(10))
def test_multitask_kernel_matrix_evaluation(covariance_test_samples, c1, c2, sample_idx):
    sample = covariance_test_samples[sample_idx]
    x, z, hparams = sample["x"], sample["z"], sample["hparams"]
    x = np.concatenate((x, np.random.choice(MULTITASK_TASKS, size=(len(x), 1))), axis=1)
    z = np.concatenate((z, np.random.choice(MULTITASK_TASKS, size=(len(z), 1))), axis=1)
    xn, d = x.shape
    zn, _ = z.shape
    full_zx = np.tile(z, (xn, 1))
    full_xz = np.reshape(np.tile(x, (1, zn)), (xn * zn, d))
    full_zz = np.tile(z, (zn, 1))
    full_zz_trans = np.reshape(np.tile(z, (1, zn)), (zn * zn, d))
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


@pytest.mark.parametrize("c1", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("c2", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("sample_idx", range(10))
def test_multitask_kernel_gradient_tensor_evaluation(covariance_test_samples, c1, c2, sample_idx):
    sample = covariance_test_samples[sample_idx]
    x, z, hparams = sample["x"], sample["z"], sample["hparams"]
    x = np.concatenate((x, np.random.choice(MULTITASK_TASKS, size=(len(x), 1))), axis=1)
    z = np.concatenate((z, np.random.choice(MULTITASK_TASKS, size=(len(z), 1))), axis=1)
    xn, d = x.shape
    zn, _ = z.shape
    full_zx = np.tile(z, (xn, 1))
    full_xz = np.reshape(np.tile(x, (1, zn)), (xn * zn, d))
    full_zz = np.tile(z, (zn, 1))
    full_zz_trans = np.reshape(np.tile(z, (1, zn)), (zn * zn, d))
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


@pytest.mark.parametrize("c1", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("c2", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("sample_idx", range(10))
def test_multitask_kernel_gradient_against_finite_difference(covariance_test_samples, c1, c2, sample_idx):
    sample = covariance_test_samples[sample_idx]
    x, z, hparams = sample["x"], sample["z"], sample["hparams"]
    x = np.concatenate((x, np.random.choice(MULTITASK_TASKS, size=(len(x), 1))), axis=1)
    z = np.concatenate((z, np.random.choice(MULTITASK_TASKS, size=(len(z), 1))), axis=1)
    xn, d = x.shape
    zn, _ = z.shape
    n = min(xn, zn)
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


@pytest.mark.parametrize("c1", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("c2", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("sample_idx", range(10))
def test_multitask_kernel_hyperparameter_gradient_against_finite_difference(
    covariance_test_samples, c1, c2, sample_idx
):
    sample = covariance_test_samples[sample_idx]
    x, z, hparams = sample["x"], sample["z"], sample["hparams"]
    x = np.concatenate((x, np.random.choice(MULTITASK_TASKS, size=(len(x), 1))), axis=1)
    z = np.concatenate((z, np.random.choice(MULTITASK_TASKS, size=(len(z), 1))), axis=1)
    xn, d = x.shape
    zn, _ = z.shape
    n = min(xn, zn)
    x = x[:n, :]
    z = z[:n, :]
    hparams_with_task = np.append(hparams, np.random.random())

    def func(hparams_in):
        cov = MultitaskTensorCovariance(hparams_in.squeeze(), c1, c2)
        return cov.covariance(x, z)

    def grad(hparams_in):
        cov = MultitaskTensorCovariance(hparams_in.squeeze(), c1, c2)
        return cov.hyperparameter_grad_covariance(x, z)

    check_gradient_with_finite_difference(
        np.reshape(hparams_with_task, (1, -1)),
        func,
        grad,
        tol=d * n * n * 1e-6,
        fd_step=1e-8 * hparams_with_task,
    )


@pytest.mark.parametrize("c1", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("c2", DIFF_COVARIANCE_CLASSES)
@pytest.mark.parametrize("sample_idx", range(10))
def test_multitask_kernel_hyperparameter_gradient_tensor_evaluation(covariance_test_samples, c1, c2, sample_idx):
    sample = covariance_test_samples[sample_idx]
    x, z, hparams = sample["x"], sample["z"], sample["hparams"]
    x = np.concatenate((x, np.random.choice(MULTITASK_TASKS, size=(len(x), 1))), axis=1)
    z = np.concatenate((z, np.random.choice(MULTITASK_TASKS, size=(len(z), 1))), axis=1)
    xn, d = x.shape
    zn, _ = z.shape
    full_zx = np.tile(z, (xn, 1))
    full_xz = np.reshape(np.tile(x, (1, zn)), (xn * zn, d))
    full_zz = np.tile(z, (zn, 1))
    full_zz_trans = np.reshape(np.tile(z, (1, zn)), (zn * zn, d))
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
    def _generate_random_scale_and_difference_matrix(dim_in, num_points_in):
        scale = np.random.uniform(-1000, 1000, size=(num_points_in, num_points_in))
        difference_matrix = np.random.uniform(-1000, 1000, size=(num_points_in, num_points_in, dim_in))
        return scale, difference_matrix

    scale, difference_matrix = _generate_random_scale_and_difference_matrix(dim, num_points)
    scaled = covariance._scale_difference_matrix(scale, difference_matrix)
    for i in range(dim):
        np.testing.assert_array_equal(scaled[:, :, i], scale * difference_matrix[:, :, i])


def test_covariance_links_have_all_covariance_types():
    """Test each covariance type is in a linker, and every linker key is a covariance type."""
    assert set(CovarianceType) == set(covariance.COVARIANCE_TYPES_TO_CLASSES.keys())
