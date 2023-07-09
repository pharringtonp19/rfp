from dataclasses import dataclass
from functools import partial
from rfp.utils import final_layer
import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Dict
from rfp.utils import Model_Params
from rfp.train import Trainer

def binary_cross_entropy(predict, target, mask):
    return -target * jax.nn.log_sigmoid(predict) - (1 - target) * jax.nn.log_sigmoid(1 - predict) * (1-mask)

@dataclass
class Supervised_Loss:
    loss_fn: Callable[[jnp.array, jnp.array, jnp.array], jnp.array]
    feature_map: Callable[[Dict], Tuple[jnp.array, float]]                                         ### TODO: Is this the correct type?
    reg_value: float = 0.0                                                                         # Regularizing the forward pass
    aux_status: bool = False

    # @jax.jit
    def __call__(self, params: Model_Params, data: Dict) -> jnp.array:
        Y, X, mask = data["Y"], data["X"], data["Mask"]
        # print(Y.shape, X.shape, mask.shape)
        phiX, fwd_pass_penalty = self.feature_map(params.body, X)
        # print(phiX.shape, fwd_pass_penalty)
        Yhat = final_layer(params, phiX)
        empirical_loss = jnp.sum(
            jax.vmap(self.loss_fn)(Yhat, Y, mask)) / jnp.sum(1-mask)
        if self.aux_status:
            return (empirical_loss + self.reg_value * fwd_pass_penalty, fwd_pass_penalty) ### TODO: check this
        return empirical_loss + self.reg_value * fwd_pass_penalty



@dataclass
class Cluster_Loss:
    inner_yuri: Trainer
    reg_value: float = 1.0
    aux_status: bool = False

    def cluster_loss(self, params:Trainer, data: Dict) -> jnp.array:
        cluster_params, _ = self.inner_yuri.train(params, data)
        a2 = self.inner_yuri.loss_fn(cluster_params, data)
        a1 = self.inner_yuri.loss_fn(params, data)
        return (1 - self.reg_value) * a1 + self.reg_value * a2

    def __call__(self, params:Trainer, data: Dict) -> jnp.array:
        losses = jax.vmap(self.cluster_loss, in_axes=(None, 0))(params, data)
        return jnp.mean(losses)


@dataclass
class Sqr_Error:
    """Square Error"""

    mlp: callable
    data_split: callable = lambda x: x
    aux_status: bool = False

    def __call__(self, params, data):
        """compute loss"""
        targets, inputs = self.data_split(data)
        prediction = self.mlp.fwd_pass(params, inputs)
        return jnp.mean((prediction - targets) ** 2)


# @dataclass
# class feature_map_loss:
#     """Computes the Feature Map Loss"""

#     feature_map: callable
#     reg_value: float = 0.0
#     aux_status: bool = True

#     def __call__(self, params, data):
#         target, x = data
#         prediction, penalty = self.feature_map(params, x)
#         prediction_loss = jnp.mean((target - prediction) ** 2)
#         return prediction_loss + self.reg_value * penalty, (prediction_loss, penalty)


# @dataclass
# class Supervised_Loss:
#     linear_layer: callable
#     feature_map: callable

#     # @jax.jit
#     def __call__(self, params, data):
#         """We implement this function as composition of partially evaluated functions"""
#         Y, D, X = data

#         # Partial Evaluation
#         partial_feature_map = partial(self.feature_map, X=X)
#         partial_linear_layer = partial(self.linear_layer, Y, D)

#         # Composition
#         phiX, vector_field_penalty = partial_feature_map(params)
#         prediction_error, prediction_penalty = partial_linear_layer(phiX)
#         return prediction_error, vector_field_penalty + prediction_penalty


# @dataclass
# class Cluster_Loss:
#     supervised_loss: callable
#     inner_trainer: callable
#     reg_value: float = 1.0
#     aux_status: bool = False

#     def cluster_loss(self, params, data):

#         # Partial Evaluation
#         partial_cluster_map = partial(self.inner_trainer.train, data=data)
#         partial_supervised_pass = partial(self.supervised_loss.loss_fn, data=data)

#         # Composition (This is MESSY!)
#         a, _ = partial_cluster_map(params)
#         a1, _ = partial_supervised_pass(params)
#         b, b1 = partial_supervised_pass(a)
#         return b + self.reg_value * a1 + self.supervised_loss.reg_value * b1

#     def __call__(self, params, data):
#         cluster_losses = jax.tree_util.tree_map(
#             partial(self.cluster_loss, params), data
#         )
#         loss = (1 / (len(data))) * jax.tree_util.tree_reduce(
#             lambda a, b: a + b, cluster_losses
#         )
#         return loss


# @dataclass
# class Cluster_Loss_ff:
#     inner_yuri: callable
#     reg_value: float = 1.0
#     aux_status: bool = False

#     def cluster_loss(self, params, data):

#         # Partial Evaluation
#         cluster_params, _ = self.inner_yuri.train(params, data)
#         a2 = self.inner_yuri.loss_fn(cluster_params, data)
#         a1 = self.inner_yuri.loss_fn(params, data)
#         return (1 - self.reg_value) * a1 + self.reg_value * a2

#     def __call__(self, params, data):
#         cluster_losses = jax.tree_util.tree_map(
#             partial(self.cluster_loss, params), data
#         )
#         loss = (1 / (len(data))) * jax.tree_util.tree_reduce(
#             lambda a, b: a + b, cluster_losses
#         )
#         return loss
