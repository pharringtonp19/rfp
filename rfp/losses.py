from dataclasses import dataclass
from functools import partial
from rfp.utils import final_layer
import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Dict
from rfp.utils import ModelParams
from rfp.train import Trainer


def mse(predict, target, mask):                      ### TODO: Is this the correct type?
    return (predict-target)**2 * mask

def binary_cross_entropy(logits, target, mask):  
    probability_of_one = jax.nn.sigmoid(logits)
    return -target * jnp.log(probability_of_one) - (1 - target) * jnp.log(1 - probability_of_one) * mask


def softmax_cross_entropy(predict, target, mask):                     ### TODO: Is this the correct type?
    return -jnp.sum(jax.nn.log_softmax(predict, axis=-1)*target, axis=-1) * mask

@dataclass
class Supervised_Loss:
    loss_fn: Callable 
    rfp : Callable                                     ### TODO: Is this the correct type?
    reg_value: float = 0.0                                                                         
    aux_status: bool = False

    # @jax.jit
    def __call__(self, params: ModelParams, X, Y, mask) -> jnp.array:
        Yhat, fwd_pass_penalty = self.rfp(params, X) 
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
