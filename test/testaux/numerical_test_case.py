# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


DEFAULT_ABS_TOL: float = 1e-9  # if the absolute difference is bellow this value, values will be considered close


def assert_scalar_within_relative(value: float, truth: float, tol: float) -> None:
    denom = np.fabs(truth)
    if denom < 2.2250738585072014e-308:  # np.finfo(np.float64).tiny:
        denom = 1.0  # do not divide by 0
    diff = np.fabs((value - truth) / denom)
    assert diff <= tol, f"value = {value:.18E}, truth = {truth:.18E}, diff = {diff:.18E}, tol = {tol:.18E}"


def assert_scalar_is_close(value: float, truth: float, tol: float, abs_tol: float = DEFAULT_ABS_TOL) -> None:
    diff = np.fabs(value - truth)
    is_close = np.isclose(value, truth, rtol=tol, atol=abs_tol)
    assert is_close, (
        f"value = {value:.18E}, truth = {truth:.18E}\ndiff = {diff:.18E}, tol = {tol:.18E}, abs_tol = {abs_tol:.18E}"
    )


def assert_vector_within_relative(value: NDArray[np.number], truth: NDArray[np.number], tol: float) -> None:
    assert value.shape == truth.shape, f"value.shape = {value.shape} != truth.shape = {truth.shape}"
    for index in np.ndindex(value.shape):
        assert_scalar_within_relative(value[index], truth[index], tol)


def assert_vector_row_wise_norm_is_close(
    value: NDArray[np.number],
    truth: NDArray[np.number],
    tol: float,
    norm: int | float | str | None = 2,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> None:
    assert value.shape == truth.shape, f"value.shape = {value.shape} != truth.shape = {truth.shape}"
    value_norms = np.linalg.norm(value, axis=1, ord=norm)  # ty: ignore[no-matching-overload]
    truth_norms = np.linalg.norm(truth, axis=1, ord=norm)  # ty: ignore[no-matching-overload]
    diff = np.fabs(value_norms - truth_norms)
    denom = np.fabs(truth_norms)
    bound = np.maximum(denom * tol, abs_tol)
    failed_assert = np.flatnonzero(diff > bound)
    for index in failed_assert:
        assert diff[index] <= max(tol * bound[index], abs_tol), (
            f"truth and value vectors are different on indices {failed_assert.tolist()}. First error: \n"
            f"value[{index}, :] = {value[index]}\n"
            f"truth[{index}, :] = {truth[index]}\n"
            f"diff norm = {diff[index]} <= max(tol = {tol} * {bound[index]}, abs_tol = {abs_tol})"
        )
    assert len(failed_assert) == 0


def assert_vector_within_relative_norm(
    value: NDArray[np.number],
    truth: NDArray[np.number],
    tol: float,
    norm: int | float | str | None = 2,
) -> None:
    size = value.size
    assert size == truth.size, f"value.size = {size} != truth.size = {truth.size}"
    v = np.reshape(value, (size,))
    t = np.reshape(truth, (size,))
    err = np.linalg.norm(v - t, ord=norm)  # ty: ignore[invalid-argument-type]
    mag = np.linalg.norm(t, ord=norm) if np.linalg.norm(t, ord=norm) > np.finfo(np.float64).eps else 1.0  # ty: ignore[invalid-argument-type]
    assert err / mag < tol, f"error = {err} / magnitude = {mag} > tol = {tol}"


def check_gradient_with_finite_difference(
    x: NDArray[np.number],
    func: Callable[[NDArray[np.number]], NDArray[np.number]],
    grad: Callable[[NDArray[np.number]], NDArray[np.number]],
    fd_step: NDArray[np.number],
    tol: float,
    *,
    use_complex: bool = False,
) -> None:
    """
    Approximate gradient using finite difference using either the centered method or complex step.
    """

    def fd_centered_method(
        x: NDArray[np.number],
        func: Callable[[NDArray[np.number]], NDArray[np.number]],
        fd_step: NDArray[np.number],
        g_approx: NDArray[np.number],
    ) -> NDArray[np.number]:
        x_plus_perturbation = x[:, :, None] + np.diag(fd_step)
        x_min_perturbation = x[:, :, None] - np.diag(fd_step)
        dim = x.shape[1]
        for i in range(dim):
            g_approx[:, i] = (func(x_plus_perturbation[:, :, i]) - func(x_min_perturbation[:, :, i])) / (2 * fd_step[i])
        return g_approx

    def fd_complex_step(
        x: NDArray[np.number],
        func: Callable[[NDArray[np.number]], Any],
        fd_step: NDArray[np.number],
        g_approx: NDArray[np.number],
    ) -> NDArray[np.number]:
        dim = x.shape[1]
        z = x + 0j
        for i in range(dim):
            z[:, i] += fd_step[i] * 1j
            g_approx[:, i] = func(z).imag / fd_step[i]
            z[:, i] -= fd_step[i] * 1j
        return g_approx

    assert len(x.shape) == 2
    assert x.shape[1] == fd_step.shape[0]
    g = grad(x)
    if len(g.shape) == 1:
        g = g.reshape((1, -1))
    g_approx = np.empty_like(g)
    finite_difference_method: Callable = fd_complex_step if use_complex else fd_centered_method
    g_approx = finite_difference_method(x, func, fd_step, g_approx)
    assert_vector_row_wise_norm_is_close(g_approx, g, tol)
