# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True, slots=True)
class PredictorCoreComponents:
    x: np.ndarray
    mean: np.ndarray
    var: np.ndarray
    z: np.ndarray
    sqrt_var: np.ndarray
    cdf_z: np.ndarray
    pdf_z: np.ndarray
    grad_mean: np.ndarray
    grad_var: np.ndarray
    grad_sqrt_var: np.ndarray


class Predictor:
    """Interface for performing predictions"""

    @property
    def dim(self) -> int:
        raise NotImplementedError()

    @property
    def differentiable(self) -> bool:
        raise NotImplementedError()

    @property
    def num_sampled(self) -> int:
        raise NotImplementedError()

    @property
    def points_sampled(self):
        raise NotImplementedError()

    @property
    def points_sampled_value(self):
        raise NotImplementedError()

    @property
    def points_sampled_noise_variance(self):
        raise NotImplementedError()

    def compute_mean_of_points(self, points_to_sample):
        raise NotImplementedError()

    def compute_variance_of_points(self, points_to_sample):
        raise NotImplementedError()

    def compute_mean_and_variance_of_points(self, points_to_sample):
        raise NotImplementedError()

    def compute_grad_mean_of_points(self, points_to_sample):
        raise NotImplementedError()

    def compute_mean_variance_grad_of_points(self, points_to_sample):
        raise NotImplementedError()

    def compute_covariance_of_points(self, points_to_sample):
        raise NotImplementedError()


# TODO(RTL-44): Consider this definition of "best_value" or if something else is more appropriate
class HasPredictor:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor
        self.best_value = None

    # NOTE: This relies on var taking some minimum positive value since it divides by that quantity
    def compute_core_components(self, points_to_evaluate, option):
        eval_shape = points_to_evaluate.shape
        assert len(eval_shape) == 2 and eval_shape[1] == self.dim
        assert option in ("func", "grad", "both")

        grad_mean = grad_var = grad_sqrt_var = None
        if option in ("func",):
            mean, var = self.predictor.compute_mean_and_variance_of_points(points_to_evaluate)
        else:  # option in ("grad", "both")
            (
                mean,
                var,
                grad_mean,
                grad_var,
            ) = self.predictor.compute_mean_variance_grad_of_points(points_to_evaluate)
        sqrt_var = np.sqrt(var)
        if grad_var is not None:
            grad_sqrt_var = 0.5 * grad_var / sqrt_var[:, None]

        z = cdf_z = pdf_z = None
        if self.best_value is not None:
            z = (self.best_value - mean) / sqrt_var
            cdf_z = norm.cdf(z)
            pdf_z = norm.pdf(z)

        return PredictorCoreComponents(
            points_to_evaluate,
            mean,
            var,
            z,
            sqrt_var,
            cdf_z,
            pdf_z,
            grad_mean,
            grad_var,
            grad_sqrt_var,
        )

    @property
    def dim(self) -> int:
        return self.predictor.dim

    @property
    def differentiable(self) -> bool:
        return self.predictor.differentiable
