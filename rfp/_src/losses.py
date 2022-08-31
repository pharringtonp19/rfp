from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from rfp._src.utils import batchify, split


@dataclass 
class Sqr_Error:
    """Square Error"""
    mlp: callable 
    data_split: callable = lambda x : x 
    aux_status: bool = False 

    def __call__(self, params, data):
        """compute loss"""
        targets, inputs = self.data_split(data)
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
class Supervised_Loss_Time:
    linear_layer: callable
    feature_map: callable
    reg_value: float = 1.0
    aux_status: bool = False

    # @jax.jit
    def loss_fn(self, params, data):
        """We implement this function as composition of partially evaluated functions"""
        jax.debug.print("data_shape: {}", data.shape)
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
        """Why don't we ever return the sum of the losses???"""
        prediction_error, penalty = self.loss_fn(params, data)
        if self.aux_status:
            return prediction_error + self.reg_value * penalty, (
                prediction_error,
                penalty,
            )
        return prediction_error


@dataclass
class Cluster_Loss:
    supervised_loss: callable
    inner_trainer: callable
    reg_value: float = 1.0
    aux_status: bool = False

    def cluster_loss(self, params, data):

        # Partial Evaluation
        partial_cluster_map = partial(self.inner_trainer.train, data=data)
        partial_supervised_pass = partial(self.supervised_loss.loss_fn, data=data)

        # Composition (This is MESSY!)
        a, _ = partial_cluster_map(params)
        a1, _ = partial_supervised_pass(params)
        b, b1 = partial_supervised_pass(a)
        return b + self.reg_value * a1 + self.supervised_loss.reg_value * b1

    def __call__(self, params, data):
        cluster_losses = jax.tree_util.tree_map(
            partial(self.cluster_loss, params), data
        )
        loss = (1 / (len(data))) * jax.tree_util.tree_reduce(
            lambda a, b: a + b, cluster_losses
        )
        return loss


@dataclass
class Cluster_Loss_ff:
    inner_yuri: callable
    reg_value: float = 1.0
    aux_status: bool = False

    def cluster_loss(self, params, data):

        # Partial Evaluation
        cluster_params, _ = self.inner_yuri.train(params, data)
        a2 = self.inner_yuri.loss_fn(cluster_params, data)
        a1 = self.inner_yuri.loss_fn(params, data)
        return (1 - self.reg_value) * a1 + self.reg_value * a2

    def __call__(self, params, data):
        cluster_losses = jax.tree_util.tree_map(
            partial(self.cluster_loss, params), data
        )
        loss = (1 / (len(data))) * jax.tree_util.tree_reduce(
            lambda a, b: a + b, cluster_losses
        )
        return loss


if __name__ == "__main__":
    from rfp import MLP

    mlp = MLP([32, 32, 1])
    print(type(mlp))

    loss = sqr_error(mlp)
    print(loss.aux_status)
