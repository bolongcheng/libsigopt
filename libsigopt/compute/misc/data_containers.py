# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import numpy as np

from libsigopt.aux.constant import MINIMUM_VALUE_VAR
from libsigopt.compute.misc.constant import DEFAULT_CONSTANT_LIAR_VALUE, ConstantLiarType


DEFAULT_VALUE_VAR = 0.0
MIDPOINT_NORMALIZATION_SCALE_FACTOR = 0.1
MINIMUM_METRIC_HALF_WIDTH = 1.0e-8


class MetricMidpointInfo:
    midpoint: np.ndarray
    scale: np.ndarray
    negate: np.ndarray

    def __init__(self):
        self.force_skip = False
        self.midpoint = np.array([])
        self.scale = np.array([])
        self.negate = np.array([])

    def __repr__(self):
        return f"{self.__class__.__name__}(mid={self.midpoint}, scale={self.scale}, skip={self.skip})"

    @property
    def skip(self):
        return self.force_skip or self.midpoint is None

    def get_negate_from_objective(self, objective):
        return 1 if objective == "minimize" else -1

    def compute_lie_value(self, lie_method):
        raise NotImplementedError()

    def relative_objective_value(self, values):
        if self.skip:
            return self.negate * values
        return self.negate * self.scale * (values - self.midpoint)

    def relative_objective_variance(self, value_vars):
        value_vars = value_vars if value_vars is not None else np.full_like(value_vars, DEFAULT_VALUE_VAR)
        if self.skip:
            return np.fmax(value_vars, 1e-6)  # Experimenting with MINIMUM_VALUE_VAR changes other things too
        return np.fmax(value_vars * self.scale**2, MINIMUM_VALUE_VAR)

    def undo_scaling(self, values):
        if self.skip:
            return self.negate * values
        return self.negate * values / self.scale + self.midpoint

    # NOTE: This will not reverse relative_objective_variance if the fmax were employed there
    def undo_scaling_variances(self, value_vars):
        if self.skip:
            return value_vars
        return value_vars / self.scale**2


class MultiMetricMidpointInfo(MetricMidpointInfo):
    def __init__(self, values, failures, objectives=None):
        """
        This is a wrapper for a list of SingleMetricMidpointInfo objects
        """
        super().__init__()

        assert len(np.asarray(values).shape) == 2, "values must be an n-D array"
        assert objectives is None or len(objectives) == values.shape[1], "there must be an objective for each metric"

        self.tuple_of_smmi = tuple(
            SingleMetricMidpointInfo(
                values=values[:, i],
                failures=failures,
                objective=objectives[i] if objectives else None,
            )
            for i in range(values.shape[1])
        )

        self.negate = np.array([m.negate for m in self.tuple_of_smmi])
        self.midpoint = np.array([m.midpoint for m in self.tuple_of_smmi])
        self.scale = np.array([m.scale for m in self.tuple_of_smmi])
        self.force_skip = any(m.force_skip for m in self.tuple_of_smmi)
        # sync each SingleMetricMidpointInfo in case we need to access one of them
        if self.force_skip:
            for m in self.tuple_of_smmi:
                m.force_skip = True

    def compute_lie_value(self, lie_method):
        return np.array([m.compute_lie_value(lie_method) for m in self.tuple_of_smmi])


# NOTE: Probably should rename this if moving lie stuff in here
# NOTE: Any benefit to masked_arrays ??
class SingleMetricMidpointInfo(MetricMidpointInfo):
    def __init__(self, values, failures, objective=None):
        """Rescales the data from
        [min, max]
        to
        [MIDPOINT_NORMALIZATION_SCALE_FACTOR, -MIDPOINT_NORMALIZATION_SCALE_FACTOR]
        Note the application of the -1 that is applied, to convert a max problem to a min problem.

        This also now computes the "lie" or "failure" value, which I think makes sense because it's
        doing essentially the same computation

        """
        super().__init__()

        self.non_fail_values = values[np.logical_not(failures)]
        assert len(self.non_fail_values.shape) == 1, "values must be an 1-D array"

        self.negate = self.get_negate_from_objective(objective)

        if len(self.non_fail_values) == 0:
            self.force_skip = True

        if not self.force_skip:
            self.min, self.max = np.min(self.non_fail_values), np.max(self.non_fail_values)
            self.midpoint = (self.max + self.min) * 0.5

            if (self.max - self.min) * 0.5 < MINIMUM_METRIC_HALF_WIDTH:
                if min(np.abs([self.max, self.min])) > 1:
                    self.scale = 1 / max(np.abs([self.min, self.max]))
                    self.midpoint = self.min
                else:
                    self.scale = 1
                    self.midpoint = 0
            else:
                self.scale = 2 * MIDPOINT_NORMALIZATION_SCALE_FACTOR / (self.max - self.min)

    def compute_lie_value(self, lie_method: ConstantLiarType):
        # TODO(RTL-56): Think about if, maybe, we should try/catch if this gets called with no non-failures
        if not len(self.non_fail_values):
            return DEFAULT_CONSTANT_LIAR_VALUE

        maximizing = bool(self.negate == -1)
        match lie_method:
            case ConstantLiarType.MIN:
                return np.min(self.non_fail_values) if maximizing else np.max(self.non_fail_values)
            case ConstantLiarType.MAX:
                return np.max(self.non_fail_values) if maximizing else np.min(self.non_fail_values)
            case ConstantLiarType.MEAN:
                return np.mean(self.non_fail_values)
            case _:
                raise ValueError(f"Unknown lie_method: {lie_method}")


class HistoricalData:
    def __init__(self, dim):
        self.dim = dim
        self.points_sampled = np.empty((0, self.dim))
        self.points_sampled_value = np.empty(0)
        self.points_sampled_noise_variance = np.empty(0)

    def __str__(self):
        """String representation of this HistoricalData object."""
        return "\n".join(
            [
                repr(self.points_sampled),
                repr(self.points_sampled_value),
                repr(self.points_sampled_noise_variance),
            ]
        )

    def append_lies(self, points_being_sampled, lie_value, lie_value_var):
        self.append_historical_data(
            np.asarray(points_being_sampled),
            lie_value * np.ones(len(points_being_sampled)),
            lie_value_var * np.ones(len(points_being_sampled)),
        )

    def append_historical_data(self, points_sampled, points_sampled_value, points_sampled_noise_variance):
        """Append lists of points_sampled, their values, and their noise variances to the data members of this class."""
        if points_sampled.size == 0:
            return

        assert len(points_sampled.shape) == 2
        assert len(points_sampled_value.shape) == len(points_sampled_noise_variance.shape) == 1
        assert len(points_sampled) == len(points_sampled_value) == len(points_sampled_noise_variance)
        assert points_sampled.shape[1] == self.dim

        self.points_sampled = np.append(self.points_sampled, points_sampled, axis=0)
        self.points_sampled_value = np.append(self.points_sampled_value, points_sampled_value)
        self.points_sampled_noise_variance = np.append(
            self.points_sampled_noise_variance, points_sampled_noise_variance
        )

    @property
    def num_sampled(self):
        """Return the number of sampled points."""
        return self.points_sampled.shape[0]
