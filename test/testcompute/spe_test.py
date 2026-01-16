# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import random

import numpy as np
import pytest
from testviews.zigopt_input_utils import form_points_sampled, form_random_unconstrained_categorical_domain

from libsigopt.compute.covariance import C0RadialMatern, C4RadialMatern
from libsigopt.compute.misc.multimetric import (
    MultimetricMethod,
    MultimetricOptPhase,
    filter_multimetric_points_sampled_spe,
    form_multimetric_info_from_phase,
)
from libsigopt.compute.sigopt_parzen_estimator import (
    SPE_MINIMUM_UNFORGOTTEN_POINT_TOTAL,
    SigOptParzenEstimator,
    SPEInsufficientDataError,
)


# NumericalTestCase import removed as it's no longer needed for this file


domain = form_random_unconstrained_categorical_domain(np.random.randint(4, 12)).one_hot_domain
hparams = [1.0] + (0.2 * np.diff(domain.get_lower_upper_bounds(), axis=0)[0]).tolist()
greater_covariance = C4RadialMatern(hparams)
gamma = 0.5


@pytest.fixture(scope="module")
def form_multimetric_info():
    def _form_multimetric_info(method_name):
        if method_name == MultimetricMethod.CONVEX_COMBINATION:
            phase = random.choice(
                [
                    MultimetricOptPhase.CONVEX_COMBINATION_RANDOM_SPREAD,
                    MultimetricOptPhase.CONVEX_COMBINATION_SEQUENTIAL,
                ]
            )
            phase_kwargs = {"fraction_of_phase_completed": np.random.random()}
        elif method_name == MultimetricMethod.EPSILON_CONSTRAINT:
            phase = random.choice(
                [
                    MultimetricOptPhase.EPSILON_CONSTRAINT_OPTIMIZE_0,
                    MultimetricOptPhase.EPSILON_CONSTRAINT_OPTIMIZE_1,
                ]
            )
            phase_kwargs = {"fraction_of_phase_completed": np.random.random()}
        elif method_name == MultimetricMethod.OPTIMIZING_ONE_METRIC:
            phase = random.choice(
                [
                    MultimetricOptPhase.OPTIMIZING_ONE_METRIC_OPTIMIZE_0,
                    MultimetricOptPhase.OPTIMIZING_ONE_METRIC_OPTIMIZE_1,
                ]
            )
            phase_kwargs = {}
        else:
            phase = MultimetricOptPhase.NOT_MULTIMETRIC
            phase_kwargs = {}
        return form_multimetric_info_from_phase(phase, phase_kwargs)

    return _form_multimetric_info


@pytest.mark.parametrize(
    "phase",
    [
        MultimetricMethod.CONVEX_COMBINATION,
        MultimetricMethod.EPSILON_CONSTRAINT,
        MultimetricMethod.OPTIMIZING_ONE_METRIC,
        MultimetricOptPhase.NOT_MULTIMETRIC,
    ],
)
def test_form_multimetric_info_fixture(form_multimetric_info, phase):
    multimetric_info = form_multimetric_info(phase)
    if phase == MultimetricOptPhase.NOT_MULTIMETRIC:
        assert multimetric_info.method is None
    else:
        assert multimetric_info.method == phase


@pytest.mark.parametrize(
    "phase",
    [
        MultimetricMethod.CONVEX_COMBINATION,
        MultimetricMethod.EPSILON_CONSTRAINT,
        MultimetricMethod.OPTIMIZING_ONE_METRIC,
        MultimetricOptPhase.NOT_MULTIMETRIC,
    ],
)
def test_default(form_multimetric_info, phase):
    num_metrics = 1 if phase == MultimetricOptPhase.NOT_MULTIMETRIC else 2
    points_sampled = form_points_sampled(
        domain=domain,
        num_sampled=np.random.randint(100, 200),
        noise_per_point=0,
        num_metrics=num_metrics,
        task_options=np.array([]),
        failure_prob=0.1,
    )
    multimetric_info = form_multimetric_info(phase)
    lie_values = np.empty(num_metrics)
    points_to_sample = domain.generate_quasi_random_points_in_domain(np.random.randint(100, 200))
    (
        points_sampled.points,
        points_sampled.values,
    ) = filter_multimetric_points_sampled_spe(
        multimetric_info,
        points_sampled.points,
        points_sampled.values,
        points_sampled.failures,
        lie_values,
    )

    if len(points_sampled.values) < SPE_MINIMUM_UNFORGOTTEN_POINT_TOTAL:
        assert phase == MultimetricMethod.EPSILON_CONSTRAINT
        with pytest.raises(SPEInsufficientDataError):
            SigOptParzenEstimator(
                lower_covariance=C0RadialMatern(hparams),
                greater_covariance=greater_covariance,
                points_sampled_points=points_sampled.points,
                points_sampled_values=points_sampled.values,
                gamma=gamma,
            )
        return

    spe = SigOptParzenEstimator(
        lower_covariance=C0RadialMatern(hparams),
        greater_covariance=greater_covariance,
        points_sampled_points=points_sampled.points,
        points_sampled_values=points_sampled.values,
        gamma=gamma,
    )
    lpdf, gpdf, ei_vals = spe.evaluate_expected_improvement(points_to_sample)
    assert all(lpdf) > 0 and all(gpdf) > 0 and all(ei_vals) > 0 and len(ei_vals) == len(points_to_sample)
    assert not spe.differentiable
    with pytest.raises(AssertionError):
        spe.evaluate_grad_expected_improvement(points_to_sample)

    spe = SigOptParzenEstimator(
        lower_covariance=C4RadialMatern(hparams),
        greater_covariance=greater_covariance,
        points_sampled_points=points_sampled.points,
        points_sampled_values=points_sampled.values,
        gamma=gamma,
    )
    assert spe.differentiable
    ei_grad = spe.evaluate_grad_expected_improvement(points_to_sample)
    assert ei_grad.shape == points_to_sample.shape


@pytest.mark.parametrize(
    "phase",
    [
        MultimetricMethod.CONVEX_COMBINATION,
        MultimetricMethod.OPTIMIZING_ONE_METRIC,
        MultimetricOptPhase.NOT_MULTIMETRIC,
    ],
)
def test_greater_lower_split(form_multimetric_info, phase):
    num_metrics = 1 if phase == MultimetricOptPhase.NOT_MULTIMETRIC else 2
    points_sampled = form_points_sampled(
        domain=domain,
        num_sampled=np.random.randint(100, 200),
        noise_per_point=0,
        num_metrics=num_metrics,
        task_options=np.array([]),
        failure_prob=0.1,
    )
    multimetric_info = form_multimetric_info(phase)
    lie_values = np.empty(num_metrics)
    points = points_sampled.points
    values = points_sampled.values
    failures = points_sampled.failures
    points, values = filter_multimetric_points_sampled_spe(
        multimetric_info,
        points,
        values,
        failures,
        lie_values,
    )
    # NOTE: max is used here since we don't apply MMI to values when creating SigOptParzenEstimator
    np.place(values, failures, np.max(values))
    spe = SigOptParzenEstimator(
        lower_covariance=C0RadialMatern(hparams),
        greater_covariance=greater_covariance,
        points_sampled_points=points,
        points_sampled_values=values,
        gamma=gamma,
    )
    sorted_indexed = np.argsort(values)
    points_sorted = points[sorted_indexed, :]
    lower, greater = (
        points_sorted[: len(points_sorted) // 2],
        points_sorted[len(points_sorted) // 2 :],
    )
    assert spe.lower_points is not None
    assert spe.greater_points is not None
    assert sorted([tuple(l) for l in spe.lower_points]) == sorted([tuple(l) for l in lower])
    assert sorted([tuple(l) for l in spe.greater_points]) == sorted([tuple(l) for l in greater])


@pytest.mark.parametrize(
    "phase",
    [
        MultimetricMethod.CONVEX_COMBINATION,
        MultimetricMethod.EPSILON_CONSTRAINT,
        MultimetricMethod.OPTIMIZING_ONE_METRIC,
        MultimetricOptPhase.NOT_MULTIMETRIC,
    ],
)
def test_insufficient_data(form_multimetric_info, phase):
    num_metrics = 1 if phase == MultimetricOptPhase.NOT_MULTIMETRIC else 2
    points_sampled = form_points_sampled(
        domain=domain,
        num_sampled=SPE_MINIMUM_UNFORGOTTEN_POINT_TOTAL - 1,
        noise_per_point=0,
        num_metrics=num_metrics,
        task_options=np.array([]),
    )
    multimetric_info = form_multimetric_info(phase)
    lie_values = np.empty(num_metrics)
    (
        points_sampled.points,
        points_sampled.values,
    ) = filter_multimetric_points_sampled_spe(
        multimetric_info,
        points_sampled.points,
        points_sampled.values,
        points_sampled.failures,
        lie_values,
    )
    with pytest.raises(SPEInsufficientDataError):
        SigOptParzenEstimator(
            lower_covariance=C0RadialMatern(hparams),
            greater_covariance=greater_covariance,
            points_sampled_points=points_sampled.points,
            points_sampled_values=points_sampled.values,
            gamma=gamma,
        )


@pytest.mark.parametrize(
    "phase",
    [
        MultimetricMethod.CONVEX_COMBINATION,
        MultimetricMethod.EPSILON_CONSTRAINT,
        MultimetricMethod.OPTIMIZING_ONE_METRIC,
        MultimetricOptPhase.NOT_MULTIMETRIC,
    ],
)
def test_insufficient_data_with_forget_factor(form_multimetric_info, phase):
    num_metrics = 1 if phase == MultimetricOptPhase.NOT_MULTIMETRIC else 2
    points_sampled = form_points_sampled(
        domain=domain,
        num_sampled=SPE_MINIMUM_UNFORGOTTEN_POINT_TOTAL + 1,
        noise_per_point=0,
        num_metrics=num_metrics,
        task_options=np.array([]),
    )
    multimetric_info = form_multimetric_info(phase)
    lie_values = np.empty(num_metrics)
    (
        points_sampled.points,
        points_sampled.values,
    ) = filter_multimetric_points_sampled_spe(
        multimetric_info,
        points_sampled.points,
        points_sampled.values,
        points_sampled.failures,
        lie_values,
    )
    with pytest.raises(SPEInsufficientDataError):
        SigOptParzenEstimator(
            lower_covariance=C0RadialMatern(hparams),
            greater_covariance=greater_covariance,
            points_sampled_points=points_sampled.points,
            points_sampled_values=points_sampled.values,
            gamma=gamma,
            forget_factor=0.5,
        )


def test_append_and_clear_lies():
    num_sampled = np.random.randint(100, 200)
    points_sampled_points = domain.generate_quasi_random_points_in_domain(num_sampled)
    points_sampled_values = np.random.rand(num_sampled)
    spe = SigOptParzenEstimator(
        lower_covariance=C0RadialMatern(hparams),
        greater_covariance=greater_covariance,
        points_sampled_points=points_sampled_points,
        points_sampled_values=points_sampled_values,
        gamma=gamma,
    )
    assert spe.lower_points is not None
    assert spe.greater_points is not None
    old_num_lower_points = len(spe.lower_points)
    old_num_greater_points = len(spe.greater_points)
    points_being_sampled_points = domain.generate_quasi_random_points_in_domain(15)
    spe.append_lies(list(points_being_sampled_points))
    assert len(spe.lower_points) == old_num_lower_points
    assert len(spe.greater_points) == old_num_greater_points + 15
    assert np.all(spe.greater_lies == points_being_sampled_points)
    assert np.all(spe.greater_points[-15:] == points_being_sampled_points)
    assert not spe.lower_lies

    spe.append_lies(list(points_being_sampled_points), lower=True)
    assert len(spe.lower_points) == old_num_lower_points + 15
    assert len(spe.greater_points) == old_num_greater_points + 15
    assert np.all(spe.lower_lies == points_being_sampled_points)
    assert np.all(spe.lower_points[-15:] == spe.greater_points[-15:])

    spe.clear_lies()
    assert len(spe.lower_points) == old_num_lower_points
    assert len(spe.greater_points) == old_num_greater_points
    assert not spe.lower_lies
    assert not spe.greater_lies

    sorted_indexed = np.argsort(points_sampled_values)
    points_sorted = points_sampled_points[sorted_indexed, :]
    lower, greater = (
        points_sorted[: int(len(points_sorted) * gamma)],
        points_sorted[int(len(points_sorted) * gamma) :],
    )
    assert sorted([tuple(l) for l in spe.lower_points]) == sorted([tuple(l) for l in lower])
    assert sorted([tuple(l) for l in spe.greater_points]) == sorted([tuple(l) for l in greater])
