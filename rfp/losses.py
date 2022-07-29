import jax.numpy as jnp 

class sqr_error:
    """Square Error """
    
    def __init__(self, mlp):
        self.mlp = mlp 
    
    def apply(self, params, data):
        """compute loss"""
        inputs, targets = data
        prediction = self.mlp.fwd_pass(params, inputs)
        return jnp.mean((prediction-targets)**2)
