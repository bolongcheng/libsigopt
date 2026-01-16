# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
from enum import StrEnum, auto


# Covariance type names
class CovarianceType(StrEnum):
    SQUARE_EXPONENTIAL = auto()
    C4_RADIAL_MATERN = auto()
    C2_RADIAL_MATERN = auto()
    C0_RADIAL_MATERN = auto()


DEFAULT_COVARIANCE_KERNEL = CovarianceType.C4_RADIAL_MATERN
DEFAULT_TASK_COVARIANCE_KERNEL = CovarianceType.SQUARE_EXPONENTIAL


# Nonzero mean names (yes, zero is an acceptable nonzero mean)
class NonzeroMeanType(StrEnum):
    ZERO = auto()
    CONSTANT = auto()
    LINEAR = auto()
    CUSTOM = auto()


# Optimizer constants
L_BFGS_B_OPTIMIZER = "l_bfgs_b_optimizer"
SLSQP_OPTIMIZER = "slsqp_optimizer"

# Vectorized Optimizer names
ADAM_OPTIMIZER = "adam"
DE_OPTIMIZER = "differential evolution"

# Vectorized optimizer types
GRADIENT_BASED_OPTIMIZERS = [
    ADAM_OPTIMIZER,
]

EVOLUTIONARY_STRATEGY_OPTIMIZERS = [
    DE_OPTIMIZER,
]


# Constant Liar constants
class ConstantLiarType(StrEnum):
    MIN = auto()
    MAX = auto()
    MEAN = auto()


DEFAULT_CONSTANT_LIAR_VALUE = -0.0123456789  # In the event there is no data (should crash maybe??)

# TODO(GH-257): Find a better default.
DEFAULT_CONSTANT_LIAR_LIE_NOISE_VARIANCE = 1e-12

AF_OPT_NEAR_BEST_STD_DEV = 0.01

DEFAULT_MAX_SIMULTANEOUS_EI_POINTS = 10000
DEFAULT_MAX_SIMULTANEOUS_QEI_POINTS = 100

CATEGORICAL_POINT_UNIQUENESS_TOLERANCE = 1e-2
DISCRETE_UNIQUENESS_LENGTH_SCALE_MIN_BOUND = {
    CovarianceType.SQUARE_EXPONENTIAL: 0.5,
    CovarianceType.C4_RADIAL_MATERN: 0.14,
    CovarianceType.C2_RADIAL_MATERN: 0.18,
    CovarianceType.C0_RADIAL_MATERN: 0.25,
}
TASK_LENGTH_LOWER_BOUND = 0.43
QUANTIZED_LENGTH_SCALE_LOWER_FACTOR = 0.25

# In multimetric (only applied to epsilon constraint or probabilistic failures),
# we enforce a min amount of successful points so that it can't return all failures.
MULTIMETRIC_MIN_NUM_SUCCESSFUL_POINTS = 5
MULTIMETRIC_MIN_NUM_IN_BOUNDS_POINTS = 1
