from functools import partial 
import jax.numpy as jnp 
from dataclasses import dataclass

class sqr_error:
    """Square Error """
    
    def __init__(self, mlp):
        self.mlp = mlp 
    
    def apply(self, params, data):
        """compute loss"""
        inputs, targets = data
        prediction = self.mlp.fwd_pass(params, inputs)
        return jnp.mean((prediction-targets)**2)

@dataclass 
class feature_map_loss: 
    """Computes the Feature Map Loss"""
    feature_map: callable
    reg_value: float = 0.0 
    aux_status: bool = True 

    def __call__(self, params, data):
        target, x = data
        prediction, penalty = self.feature_map(params, x)
        prediction_loss = jnp.mean((target-prediction)**2)
        return prediction_loss + self.reg_value*penalty, (prediction_loss, penalty)

        

@dataclass
class supervised_loss:
    linear_layer: callable 
    feature_map: callable 

    # @jax.jit
    def supervised_loss(self, params, data):
        """We implement this function as composition of partially evaluated functions"""
        Y, D, X = data

        # Partial Evaluation
        partial_feature_map = partial(self.feature_map, X=X)
        partial_linear_layer = partial(self.linear_layer, Y, D)

        # Composition
        phiX, vector_field_penalty = partial_feature_map(params)
        prediction_error, prediction_penalty = partial_linear_layer(phiX)
        return prediction_error, vector_field_penalty + prediction_penalty
