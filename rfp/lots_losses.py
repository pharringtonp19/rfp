# LOSS FUNCTIONS
from functools import partial

import jax
import jax.numpy as jnp
from rfp.higherOrder import einops_reduce
import optax


def init_feature_loss(linear_model, neural_ode, w1):
    def supervised_loss(params, data):
        """We implement this function as composition of partially evaluated functions"""
        D, X = data

        # Partial Evaluation
        partial_neural_ode = partial(neural_ode, X=X)
        partial_linear = partial(linear_model, D)

        # Composition
        a, a1 = partial_neural_ode(params)
        b, b1 = partial_linear(a)
        return b + w1 * a1 + b1

    return supervised_loss


def init_supervised_loss(linear_layer, feature_map):

    # @jax.jit
    def supervised_loss(params, data):
        """We implement this function as composition of partially evaluated functions"""
        Y, D, X = data

        # Partial Evaluation
        partial_feature_map = partial(feature_map, X=X)
        partial_linear_layer = partial(linear_layer, D)

        # Composition
        a, a1 = partial_feature_map(params)
        b, b1 = partial_linear_layer(a)
        return b, a1 + b1

    return supervised_loss


def init_cluster_loss(supervised_loss, cluster_map):

    # @jax.jit
    @einops_reduce("w -> ()", "mean")
    @partial(jax.vmap, in_axes=(None, 0))
    def cluster_loss(params, data):
        Y, D, X = data

        # Partial Evaluation
        partial_supervised_pass = partial(supervised_loss, data=data)
        partial_cluster_map = partial(cluster_map, data=data)

        # Composition
        a, a1 = partial_cluster_map(params)
        b, b1 = partial_supervised_pass(a)
        return b + a1 + b1  # I don't like this part!

    return cluster_loss


def init_loss_fn(fwd_pass):
    def loss_fn(params, data):
        target, inputs = data
        yhat, _ = fwd_pass(params, inputs)
        return jnp.mean((target - yhat) ** 2)

    return loss_fn

