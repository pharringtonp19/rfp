from dataclasses import dataclass
from functools import partial
from rfp.utils import final_layer
import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Dict
from rfp.utils import Model_Params
from rfp.train import Trainer


def mse(predict, target, mask):                      ### TODO: Is this the correct type?
    return (predict-target)**2 * mask

def binary_cross_entropy(predict, target, mask):  
    probability_of_outcome = jnp.where(target==1.0, jax.nn.sigmoid(predict), 1.0 - jax.nn.sigmoid(predict))
    return -1*jnp.log(probability_of_outcome) * mask 

def softmax_cross_entropy(predict, target, mask):                     ### TODO: Is this the correct type?
    return -jnp.sum(jax.nn.log_softmax(predict, axis=-1)*target, axis=-1) * mask

@dataclass
class Supervised_Loss:
    loss_fn: Callable 
    feature_map: Callable                                     ### TODO: Is this the correct type?
    reg_value: float = 0.0                                                                         
    aux_status: bool = False

    # @jax.jit
    def __call__(self, params: Model_Params, X, Y, mask) -> jnp.array:
        phiX, fwd_pass_penalty = self.feature_map(params.body, X)
        Yhat = final_layer(params, phiX)
        empirical_loss = jnp.sum(
            jax.vmap(self.loss_fn)(Yhat, Y, mask)) / jnp.sum(mask)
        if self.aux_status:
            return (empirical_loss + self.reg_value * fwd_pass_penalty, (empirical_loss, fwd_pass_penalty)) ### TODO: check this
        return empirical_loss + self.reg_value * fwd_pass_penalty


@dataclass
class Cluster_Loss:
    inner_yuri: Trainer
    reg_value: float = 1.0
    aux_status: bool = False

    def cluster_loss(self, params:Trainer, X, Y, mask) -> jnp.array:
        cluster_params, _ = self.inner_yuri.train(params,  X, Y, mask)
        a2 = self.inner_yuri.loss_fn(cluster_params,  X, Y, mask)
        a1 = self.inner_yuri.loss_fn(params,  X, Y, mask)
        return (1 - self.reg_value) * a1 + self.reg_value * a2

    def __call__(self, params: Trainer,  X, Y, mask) -> jnp.array:
        losses = jax.vmap(self.cluster_loss, in_axes=(None, 0, 0, 0))(params, X, Y, mask)
        weights = jnp.sum(mask, axis=1)
        return (jnp.dot(losses, weights)/jnp.sum(weights)).reshape()
