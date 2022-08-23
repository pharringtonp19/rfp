"""Public API for rfp."""

# DATA
from rfp._src.data import f1, sample1, sample2, sample3

# FEATURE MAP
from rfp._src.featuremap import neuralODE

# LINEAR MODEL
from rfp._src.linear_model import linear_model_time, linear_model_trainable_time

# LOSSES
from rfp._src.losses import feature_map_loss, sqr_error, supervised_loss_time

# NEURAL NETWORKS
from rfp._src.nn import MLP

# TRAIN
from rfp._src.train import trainer
from rfp._src.types import Data

# UTILS
from rfp._src.utils import (
    Model_Params,
    batch_sample_time,
    batchify,
    split,
    time_grad,
    training_sampler,
)
