from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from rfp._src.utils import batchify, split


class sqr_error:
    """Square Error"""

    def __init__(self, mlp):
        self.mlp = mlp
        self.aux_status: bool = False

    def __call__(self, params, data):
        """compute loss"""
        targets, inputs = data
        prediction = self.mlp.fwd_pass(params, inputs)
        return jnp.mean((prediction - targets) ** 2)


@dataclass
class feature_map_loss:
    """Computes the Feature Map Loss"""

    feature_map: callable
    reg_value: float = 0.0
    aux_status: bool = True

    def __call__(self, params, data):
        target, x = data
        prediction, penalty = self.feature_map(params, x)
        prediction_loss = jnp.mean((target - prediction) ** 2)
        return prediction_loss + self.reg_value * penalty, (prediction_loss, penalty)


@dataclass
class supervised_loss:
    linear_layer: callable
    feature_map: callable

    # @jax.jit
    def __call__(self, params, data):
        """We implement this function as composition of partially evaluated functions"""
        Y, D, X = data

        # Partial Evaluation
        partial_feature_map = partial(self.feature_map, X=X)
        partial_linear_layer = partial(self.linear_layer, Y, D)

        # Composition
        phiX, vector_field_penalty = partial_feature_map(params)
        prediction_error, prediction_penalty = partial_linear_layer(phiX)
        return prediction_error, vector_field_penalty + prediction_penalty


@dataclass
class supervised_loss_time:
    linear_layer: callable
    feature_map: callable
    reg_value: float = 1.0
    aux_status: bool = False

    # @jax.jit
    def supervised_loss(self, params, data):
        """We implement this function as composition of partially evaluated functions"""
        Y, D, T, X = split(data)  # This is the only difference

        # Partial Evaluation
        partial_feature_map = partial(self.feature_map, X=X)
        partial_linear_layer = partial(
            self.linear_layer, params["linear_params"], Y, D, T
        )  # And this too!

        # Composition
        phiX, vector_field_penalty = partial_feature_map(params["ode_params"])
        prediction_error, prediction_penalty = partial_linear_layer(phiX)
        return prediction_error, vector_field_penalty + prediction_penalty

    def __call__(self, params, data):
        prediction_error, penalty = self.supervised_loss(params, data)
        return prediction_error + self.reg_value * penalty, (prediction_error, penalty)


@dataclass
class cluster_loss:
    supervised_loss: callable
    inner_trainer: callable
    reg_value: float = 1.0
    aux_status: bool = False

    @batchify
    def cluster_loss(params, data):

        # Partial Evaluation
        partial_supervised_pass = partial(self.supervised_loss, data=data)
        partial_cluster_map = partial(self.inner_trainer.train, data=data)

        # Composition
        a, a1 = partial_cluster_map(params)
        b, b1 = partial_supervised_pass(a)
        return (
            b + self.reg_value * a1 + self.supervised_loss * b1
        )  # not exactly the composition we wanted (alas!)

    def __call__(params, data):
        return cluster_loss(params, data)


if __name__ == "__main__":
    from rfp import MLP

    mlp = MLP([32, 32, 1])
    print(type(mlp))

    loss = sqr_error(mlp)
    print(loss.aux_status)
