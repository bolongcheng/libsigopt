# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import secrets
from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np

from libsigopt.aux.multimetric import find_pareto_frontier_observations_for_maximization
from libsigopt.aux.samplers import generate_grid_points, generate_halton_points
from libsigopt.compute.misc.constant import MULTIMETRIC_MIN_NUM_IN_BOUNDS_POINTS, MULTIMETRIC_MIN_NUM_SUCCESSFUL_POINTS


# These are the names of the multimetric optimization methods
class MultimetricMethod(StrEnum):
    CONVEX_COMBINATION = auto()
    EPSILON_CONSTRAINT = auto()
    PROBABILISTIC_FAILURES = auto()
    OPTIMIZING_ONE_METRIC = auto()
    MULTIMETRIC_INITIALIZATION = auto()


@dataclass(frozen=True, slots=True)
class ConvexCombinationParams:
    weights: np.ndarray


@dataclass(frozen=True, kw_only=True)
class OptimizingMetricParams:
    optimizing_metric: int


@dataclass(frozen=True, kw_only=True)
class ConstraintMetricParams:
    constraint_metric: int


@dataclass(frozen=True, slots=True, kw_only=True)
class OptimizeOneMetricParams(OptimizingMetricParams, ConstraintMetricParams):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class ProbabilisticFailuresParams(OptimizingMetricParams, ConstraintMetricParams):
    epsilon: float


AnyParams = ConvexCombinationParams | OptimizeOneMetricParams | ProbabilisticFailuresParams


@dataclass(frozen=True, slots=True)
class MultimetricInfo:
    method: MultimetricMethod | None
    params: AnyParams | None


MULTIMETRIC_INFO_NOT_MULTIMETRIC = MultimetricInfo(method=None, params=None)

BORDER_BUFFER = 0.1


# These labels help segment the code, but could be removed if desired
class MultimetricOptPhase(StrEnum):
    NOT_MULTIMETRIC = auto()
    INITIALIZATION = auto()
    OPTIMIZING_ONE_METRIC_OPTIMIZE_0 = auto()
    OPTIMIZING_ONE_METRIC_OPTIMIZE_1 = auto()
    CONVEX_COMBINATION_RANDOM_SPREAD = auto()
    CONVEX_COMBINATION_SEQUENTIAL = auto()
    EPSILON_CONSTRAINT_OPTIMIZE_0 = auto()
    EPSILON_CONSTRAINT_OPTIMIZE_1 = auto()
    COMPLETION = auto()


# NOTE: There will be somewhat degenerate behavior in the event that a massive number of failures are present.
#              I'm not sure how we would really want to deal with that circumstance.
def identify_multimetric_phase(
    has_optimized_metric_thresholds,
    observation_budget,
    observation_count,
    failure_count,
    num_open_suggestions,
):
    INITIALIZE_FRAC = 0.15
    OPTIMIZE_ONE_METRIC_FRAC = 0.3
    CONVEX_RANDOM_FRAC = 0.45
    CONVEX_SPREAD_FRAC = 0.55
    POLISH_ONE_METRIC_FRAC = CONVEX_SPREAD_FRAC if has_optimized_metric_thresholds else 0.65
    EPSILON_CONSTRAINT_FRAC = 0.95

    bounded_below_num_open_suggestions = max(num_open_suggestions, 1)
    adjusted_budget = max(observation_budget - failure_count, bounded_below_num_open_suggestions)

    fraction_served = (observation_count + num_open_suggestions) / adjusted_budget
    fraction_completed = observation_count / adjusted_budget
    if fraction_served <= INITIALIZE_FRAC or fraction_completed <= 0.1:
        return MultimetricOptPhase.INITIALIZATION, {}
    elif fraction_served <= OPTIMIZE_ONE_METRIC_FRAC:
        return (
            MultimetricOptPhase.OPTIMIZING_ONE_METRIC_OPTIMIZE_1
            if observation_count % 2
            else MultimetricOptPhase.OPTIMIZING_ONE_METRIC_OPTIMIZE_0,
            {},
        )
    elif fraction_served <= CONVEX_RANDOM_FRAC:
        completed_frac = (fraction_served - OPTIMIZE_ONE_METRIC_FRAC) / (CONVEX_RANDOM_FRAC - OPTIMIZE_ONE_METRIC_FRAC)
        kwargs = {"fraction_of_phase_completed": completed_frac}
        return MultimetricOptPhase.CONVEX_COMBINATION_RANDOM_SPREAD, kwargs
    elif fraction_served <= CONVEX_SPREAD_FRAC:
        completed_frac = (fraction_served - CONVEX_RANDOM_FRAC) / (CONVEX_SPREAD_FRAC - CONVEX_RANDOM_FRAC)
        kwargs = {"fraction_of_phase_completed": completed_frac}
        return MultimetricOptPhase.CONVEX_COMBINATION_SEQUENTIAL, kwargs
    elif fraction_served <= POLISH_ONE_METRIC_FRAC:
        return (
            MultimetricOptPhase.OPTIMIZING_ONE_METRIC_OPTIMIZE_1
            if observation_count % 2
            else MultimetricOptPhase.OPTIMIZING_ONE_METRIC_OPTIMIZE_0,
            {},
        )
    elif fraction_served <= EPSILON_CONSTRAINT_FRAC:
        completed_frac = (fraction_served - POLISH_ONE_METRIC_FRAC) / (EPSILON_CONSTRAINT_FRAC - POLISH_ONE_METRIC_FRAC)
        kwargs = {"fraction_of_phase_completed": completed_frac}
        return (
            MultimetricOptPhase.EPSILON_CONSTRAINT_OPTIMIZE_1
            if observation_count % 2
            else MultimetricOptPhase.EPSILON_CONSTRAINT_OPTIMIZE_0,
            kwargs,
        )
    else:
        return MultimetricOptPhase.COMPLETION, {}


# The structure of this is intentionally imprecise to simplify the weight decision structure
# We consider only 100 possible weights and choose from among them, rather than something budget-dependent
def form_convex_combination_weights(phase, fraction_of_phase_completed):
    if not (0 <= fraction_of_phase_completed <= 1):  # Shouldn't be an issue, but just in case
        fraction_of_phase_completed = np.random.random()

    assert phase in (
        MultimetricOptPhase.CONVEX_COMBINATION_RANDOM_SPREAD,
        MultimetricOptPhase.CONVEX_COMBINATION_SEQUENTIAL,
    )
    interval = np.array([[BORDER_BUFFER, 1 - BORDER_BUFFER]])
    weight_index = int(100 * fraction_of_phase_completed)

    if phase == MultimetricOptPhase.CONVEX_COMBINATION_RANDOM_SPREAD:
        all_weights = generate_halton_points(101, interval, skip=1)[:, 0]
    else:
        all_weights = generate_grid_points(101, interval)[:, 0]

    weight = all_weights[weight_index]
    return np.array([weight, 1 - weight])


def form_epsilon_constraint_epsilon(fraction_of_phase_completed):
    if not (0 <= fraction_of_phase_completed <= 1):  # Shouldn't be an issue, but just in case
        fraction_of_phase_completed = np.random.random()

    interval = np.array([[BORDER_BUFFER, 1 - BORDER_BUFFER]])
    epsilon_index = int(100 * fraction_of_phase_completed)
    all_epsilons = generate_grid_points(101, interval)[:, 0]
    return all_epsilons[epsilon_index]


def form_multimetric_info_from_phase(phase, phase_kwargs):
    params: AnyParams
    if phase == MultimetricOptPhase.NOT_MULTIMETRIC:
        multimetric_info = MULTIMETRIC_INFO_NOT_MULTIMETRIC
    elif phase == MultimetricOptPhase.INITIALIZATION:
        initialization_phase = secrets.choice(
            (MultimetricOptPhase.OPTIMIZING_ONE_METRIC_OPTIMIZE_0, MultimetricOptPhase.OPTIMIZING_ONE_METRIC_OPTIMIZE_1)
        )
        multimetric_info = form_multimetric_info_from_phase(initialization_phase, {})
    elif phase in (
        MultimetricOptPhase.OPTIMIZING_ONE_METRIC_OPTIMIZE_0,
        MultimetricOptPhase.OPTIMIZING_ONE_METRIC_OPTIMIZE_1,
    ):
        if phase == MultimetricOptPhase.OPTIMIZING_ONE_METRIC_OPTIMIZE_0:
            params = OptimizeOneMetricParams(optimizing_metric=0, constraint_metric=1)
        else:
            params = OptimizeOneMetricParams(optimizing_metric=1, constraint_metric=0)
        multimetric_info = MultimetricInfo(method=MultimetricMethod.OPTIMIZING_ONE_METRIC, params=params)
    elif phase in (
        MultimetricOptPhase.CONVEX_COMBINATION_RANDOM_SPREAD,
        MultimetricOptPhase.CONVEX_COMBINATION_SEQUENTIAL,
    ):
        params = ConvexCombinationParams(
            weights=form_convex_combination_weights(phase, phase_kwargs["fraction_of_phase_completed"])
        )
        multimetric_info = MultimetricInfo(method=MultimetricMethod.CONVEX_COMBINATION, params=params)
    elif phase in (
        MultimetricOptPhase.EPSILON_CONSTRAINT_OPTIMIZE_0,
        MultimetricOptPhase.EPSILON_CONSTRAINT_OPTIMIZE_1,
    ):
        epsilon = form_epsilon_constraint_epsilon(phase_kwargs["fraction_of_phase_completed"])
        if phase == MultimetricOptPhase.EPSILON_CONSTRAINT_OPTIMIZE_0:
            params = ProbabilisticFailuresParams(optimizing_metric=0, constraint_metric=1, epsilon=epsilon)
        else:
            params = ProbabilisticFailuresParams(optimizing_metric=1, constraint_metric=0, epsilon=epsilon)
        multimetric_info = MultimetricInfo(method=MultimetricMethod.EPSILON_CONSTRAINT, params=params)
    else:
        assert phase == MultimetricOptPhase.COMPLETION
        completion_phase = secrets.choice(
            (MultimetricOptPhase.EPSILON_CONSTRAINT_OPTIMIZE_0, MultimetricOptPhase.EPSILON_CONSTRAINT_OPTIMIZE_1)
        )
        phase_kwargs = {"fraction_of_phase_completed": np.random.random()}
        multimetric_info = form_multimetric_info_from_phase(completion_phase, phase_kwargs)

    return multimetric_info


# Requires that values already be clean of nan/inf
def _find_sorted_pareto_frontier_values_minimization(values):
    pareto_ind, _ = find_pareto_frontier_observations_for_maximization(-values, np.arange(len(values)))
    pareto_values = values[pareto_ind, :]
    sort_ind = np.argsort(pareto_values[:, 0])
    return pareto_values[sort_ind, :]


"""
This function computes the constraint value for a given epsilon, the current constrained metric and
a matrix of sampled points.
It uses the best point of each metric over the constrained metric to limit the range of the constrained value.
Since epsilon is given in ascending order in zigopt, this function returns a constraint value from
closest to the maximum value to closest to the minimum value (~decreasing order). This (and the following)
step is called after the values are flipped and/or scaled through MetricMidpointInfo

metric_thresholds is a tuple that a user can specify as max values for which to consider a given metric.  If
this is provided, we use that to determine what are the appropriate bounds on which to base our application
of the epsilon value.
"""


def find_epsilon_constraint_value(
    epsilon_fraction,
    constraint_metric,
    points_sampled_values,
    metric_thresholds=tuple([None]),
):
    # NOTE: The idea is to adjust the way we calculate the constraint values
    # so that any epsilon value ( 0 < epsilon < 1) always is a value inside the pareto frontier.
    assert 0 < epsilon_fraction < 1
    assert len(points_sampled_values.shape) == 2, f"{points_sampled_values.shape} is not a 2D array"
    assert points_sampled_values.shape[1] == 2
    assert constraint_metric in (0, 1)

    if all(metric_threshold is None for metric_threshold in metric_thresholds):
        return _find_epsilon_constraint_value_no_bounds(epsilon_fraction, constraint_metric, points_sampled_values)

    assert len(metric_thresholds) == 2
    return _find_epsilon_constraint_value_with_bounds(
        epsilon_fraction,
        constraint_metric,
        points_sampled_values,
        metric_thresholds,
    )


def _find_epsilon_constraint_value_no_bounds(epsilon_fraction, constraint_metric, points_sampled_values):
    best_points = np.nanargmin(points_sampled_values, 0)
    min_bound = np.nanmin(points_sampled_values[best_points, constraint_metric])
    max_bound = np.nanmax(points_sampled_values[best_points, constraint_metric])
    return (1 - epsilon_fraction) * min_bound + epsilon_fraction * max_bound


# TODO(RTL-59): This can all be cleaned up to provide a single workflow
def _find_epsilon_constraint_value_with_bounds(
    epsilon_fraction,
    constraint_metric,
    points_sampled_values,
    metric_thresholds,
):
    clean_values = points_sampled_values[np.all(np.isfinite(points_sampled_values), axis=1), :]
    in_bounds = np.full(len(clean_values), True, dtype=bool)
    if metric_thresholds[0] is not None:
        in_bounds *= clean_values[:, 0] < metric_thresholds[0]
    if metric_thresholds[1] is not None:
        in_bounds *= clean_values[:, 1] < metric_thresholds[1]

    if np.sum(in_bounds) < MULTIMETRIC_MIN_NUM_IN_BOUNDS_POINTS:
        return _find_epsilon_constraint_value_no_bounds(epsilon_fraction, constraint_metric, points_sampled_values)

    sorted_pareto = _find_sorted_pareto_frontier_values_minimization(clean_values)
    if len(sorted_pareto) < 2:
        return _find_epsilon_constraint_value_no_bounds(epsilon_fraction, constraint_metric, points_sampled_values)

    min_bound = sorted_pareto[0 if constraint_metric == 0 else -1, constraint_metric]
    max_bound = sorted_pareto[0 if constraint_metric == 1 else -1, constraint_metric]
    if metric_thresholds[0] is not None:
        metric_0_pareto_outside_bounds = sorted_pareto[sorted_pareto[:, 0] > metric_thresholds[0], :]
        if metric_0_pareto_outside_bounds.size:
            if constraint_metric == 0:
                max_bound = metric_0_pareto_outside_bounds[0, constraint_metric]
            if constraint_metric == 1:
                min_bound = metric_0_pareto_outside_bounds[0, constraint_metric]
    if metric_thresholds[1] is not None:
        metric_1_pareto_outside_bounds = sorted_pareto[sorted_pareto[:, 1] > metric_thresholds[1], :]
        if metric_1_pareto_outside_bounds.size:
            if constraint_metric == 0:
                min_bound = metric_1_pareto_outside_bounds[-1, constraint_metric]
            if constraint_metric == 1:
                max_bound = metric_1_pareto_outside_bounds[-1, constraint_metric]

    return (1 - epsilon_fraction) * min_bound + epsilon_fraction * max_bound


"""
This function labels all points_sampled that are greater than the computed epsilon threshold value as failures
"""


def _create_epsilon_constraint_failures(
    epsilon,
    constraint_metric,
    points_sampled_values,
    points_sampled_failures,
):
    assert 0 < epsilon < 1
    assert len(points_sampled_values.shape) == 2, f"{points_sampled_values.shape} is not a 2D array"
    assert constraint_metric in (0, 1)

    successful_points = points_sampled_values[np.logical_not(points_sampled_failures), :]

    epsilon_constraint_value = find_epsilon_constraint_value(epsilon, constraint_metric, successful_points)
    epsilon_constraint_failures = points_sampled_values[:, constraint_metric] >= epsilon_constraint_value
    return epsilon_constraint_failures


"""
This function checks if there is at least a certain amount of successful points.
If not, it forces points with the lowest values to be non-failures.
"""


def force_minimum_successful_points(optimizing_metric, points_sampled_values, points_sampled_failures):
    # NOTE: With adjusted bounds, all failures are more likely to happen.
    # NOTE: This method can also pick real failures and flip them as successful,
    # but as we should receive here a constant liar instead of NaN value, it shouldn't be a problem.
    assert len(points_sampled_values.shape) == 2, f"{points_sampled_values.shape} is not a 2D array"
    assert points_sampled_values.shape[0] == len(points_sampled_failures)
    assert optimizing_metric in (0, 1)

    modified_points_sampled_failures = np.copy(points_sampled_failures)
    num_successful = sum(~points_sampled_failures)
    if num_successful < MULTIMETRIC_MIN_NUM_SUCCESSFUL_POINTS:
        diff = MULTIMETRIC_MIN_NUM_SUCCESSFUL_POINTS - num_successful

        failures_index = np.nonzero(points_sampled_failures)[0]
        order_index = np.argsort(points_sampled_values[failures_index, optimizing_metric])[:diff]

        index = failures_index[order_index]
        modified_points_sampled_failures[index] = False

    return modified_points_sampled_failures


def filter_convex_combination(
    multimetric_info,
    points_sampled_points,
    points_sampled_values,
    points_sampled_value_vars,
    points_sampled_failures,
    lie_values,
):
    modified_points_sampled_points = np.copy(points_sampled_points)

    weights = multimetric_info.params.weights
    modified_points_sampled_values = np.dot(points_sampled_values, weights)
    modified_points_sampled_value_vars = np.dot(points_sampled_value_vars, weights**2)
    modified_lie_value = np.dot(lie_values, weights)
    return (
        modified_points_sampled_points,
        modified_points_sampled_values,
        modified_points_sampled_value_vars,
        modified_lie_value,
    )


def filter_convex_combination_sum_of_gps(
    multimetric_info,
    points_sampled_points,
    points_sampled_values,
    points_sampled_value_vars,
    points_sampled_failures,
    lie_values,
):
    modified_points_sampled_points = np.copy(points_sampled_points)
    modified_points_sampled_values = np.copy(points_sampled_values)
    modified_points_sampled_value_vars = np.copy(points_sampled_value_vars)
    modified_lie_value = np.copy(lie_values)
    return (
        modified_points_sampled_points,
        modified_points_sampled_values,
        modified_points_sampled_value_vars,
        modified_lie_value,
    )


def filter_epsilon_contraint(
    multimetric_info,
    points_sampled_points,
    points_sampled_values,
    points_sampled_value_vars,
    points_sampled_failures,
    lie_values,
):
    modified_points_sampled_points = np.copy(points_sampled_points)

    epsilon = multimetric_info.params.epsilon
    optimizing_metric = multimetric_info.params.optimizing_metric
    constraint_metric = multimetric_info.params.constraint_metric
    modified_points_sampled_failures = _create_epsilon_constraint_failures(
        epsilon,
        constraint_metric,
        points_sampled_values,
        points_sampled_failures,
    )
    # Treat actual failures and epsilon constraint failures as the same
    modified_points_sampled_failures = modified_points_sampled_failures | points_sampled_failures
    modified_points_sampled_failures = force_minimum_successful_points(
        optimizing_metric,
        points_sampled_values,
        modified_points_sampled_failures,
    )
    modified_points_sampled_values = np.copy(points_sampled_values[:, optimizing_metric])
    modified_points_sampled_value_vars = np.copy(points_sampled_value_vars[:, optimizing_metric])

    modified_points_sampled_values[modified_points_sampled_failures] = lie_values[optimizing_metric]
    modified_lie_value = lie_values[optimizing_metric]

    return (
        modified_points_sampled_points,
        modified_points_sampled_values,
        modified_points_sampled_value_vars,
        modified_lie_value,
    )


def filter_probabilistic_failure(
    multimetric_info,
    points_sampled_points,
    points_sampled_values,
    points_sampled_value_vars,
    points_sampled_failures,
    lie_values,
):
    modified_points_sampled_points = np.copy(points_sampled_points)
    epsilon = multimetric_info.params.epsilon
    optimizing_metric = multimetric_info.params.optimizing_metric
    constraint_metric = multimetric_info.params.constraint_metric
    modified_points_sampled_failures = _create_epsilon_constraint_failures(
        epsilon,
        constraint_metric,
        points_sampled_values,
        points_sampled_failures,
    )
    modified_points_sampled_failures = force_minimum_successful_points(
        optimizing_metric,
        points_sampled_values,
        modified_points_sampled_failures,
    )
    modified_points_sampled_points = np.copy(points_sampled_points[~modified_points_sampled_failures, :])
    modified_points_sampled_values = np.copy(
        points_sampled_values[~modified_points_sampled_failures, optimizing_metric]
    )
    modified_points_sampled_value_vars = np.copy(
        points_sampled_value_vars[~modified_points_sampled_failures, optimizing_metric]
    )
    modified_lie_value = lie_values[optimizing_metric]

    return (
        modified_points_sampled_points,
        modified_points_sampled_values,
        modified_points_sampled_value_vars,
        modified_lie_value,
    )


def filter_optimizing_one_metric(
    multimetric_info,
    points_sampled_points,
    points_sampled_values,
    points_sampled_value_vars,
    points_sampled_failures,
    lie_values,
):
    modified_points_sampled_points = np.copy(points_sampled_points)
    optimizing_metric = multimetric_info.params.optimizing_metric
    modified_points_sampled_values = np.copy(points_sampled_values[:, optimizing_metric])
    modified_points_sampled_value_vars = np.copy(points_sampled_value_vars[:, optimizing_metric])
    modified_lie_value = lie_values[optimizing_metric]
    return (
        modified_points_sampled_points,
        modified_points_sampled_values,
        modified_points_sampled_value_vars,
        modified_lie_value,
    )


def filter_not_multimetric(
    multimetric_info,
    points_sampled_points,
    points_sampled_values,
    points_sampled_value_vars,
    points_sampled_failures,
    lie_values,
):
    modified_points_sampled_points = np.copy(points_sampled_points)
    modified_points_sampled_values = np.copy(points_sampled_values[:, 0])
    modified_points_sampled_value_vars = np.copy(points_sampled_value_vars[:, 0])
    modified_lie_value = lie_values[0]
    return (
        modified_points_sampled_points,
        modified_points_sampled_values,
        modified_points_sampled_value_vars,
        modified_lie_value,
    )


"""
The next two functions filter the point_sampled_values, points_sampled_value_vars, points_sampled_failures
arrays according to different multimetric optimization methods. The method is default to None is in the case
of single metric optimization.

NOTE: The input is being referenced not copied to create all "modified_" output.
Due to fear of input being reused after calling this function, I have just np.copy() everything.
TODO(RTL-133): Consider if historical_data creation can be absorbed into the view object.
The encapsulation will prevent the input from being reused.
"""


def filter_multimetric_points_sampled(
    multimetric_info,
    points_sampled_points,
    points_sampled_values,
    points_sampled_value_vars,
    points_sampled_failures,
    lie_values,
):
    # NOTE that EPSILON_CONSTRAINT here runs filter_probabilistic_failure
    assert multimetric_info.method is None or multimetric_info.method in (
        MultimetricMethod.CONVEX_COMBINATION,
        MultimetricMethod.EPSILON_CONSTRAINT,
        MultimetricMethod.OPTIMIZING_ONE_METRIC,
    ), f"{multimetric_info.method} method does not exist"
    if multimetric_info.method == MultimetricMethod.CONVEX_COMBINATION:
        filter_function = filter_convex_combination_sum_of_gps
    elif multimetric_info.method == MultimetricMethod.EPSILON_CONSTRAINT:
        filter_function = filter_probabilistic_failure
    elif multimetric_info.method == MultimetricMethod.OPTIMIZING_ONE_METRIC:
        filter_function = filter_optimizing_one_metric
    else:
        filter_function = filter_not_multimetric
    return filter_function(
        multimetric_info,
        points_sampled_points,
        points_sampled_values,
        points_sampled_value_vars,
        points_sampled_failures,
        lie_values,
    )


def filter_multimetric_points_sampled_spe(
    multimetric_info,
    points_sampled_points,
    points_sampled_values,
    points_sampled_failures,
    lie_values,
):
    assert multimetric_info.method is None or multimetric_info.method in (
        MultimetricMethod.CONVEX_COMBINATION,
        MultimetricMethod.EPSILON_CONSTRAINT,
        MultimetricMethod.OPTIMIZING_ONE_METRIC,
    ), f"{multimetric_info.method} method does not exist"
    match multimetric_info.method:
        case MultimetricMethod.CONVEX_COMBINATION:
            filter_function = filter_convex_combination
        case MultimetricMethod.EPSILON_CONSTRAINT:
            filter_function = filter_epsilon_contraint
        case MultimetricMethod.OPTIMIZING_ONE_METRIC:
            filter_function = filter_optimizing_one_metric
        case _:
            filter_function = filter_not_multimetric
    (
        modified_points_sampled_points,
        modified_points_sampled_values,
        _,
        modified_lie_value,
    ) = filter_function(
        multimetric_info,
        points_sampled_points,
        points_sampled_values,
        np.empty_like(points_sampled_values),
        points_sampled_failures,
        lie_values,
    )
    if multimetric_info.method != MultimetricMethod.EPSILON_CONSTRAINT:
        modified_points_sampled_values[points_sampled_failures] = modified_lie_value
    return modified_points_sampled_points, modified_points_sampled_values
