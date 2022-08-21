"""Public API for rfp."""

from rfp._src.types import Data

# DATA
from rfp._src.data import f1, sample1, sample2, sample3

# NEURAL NETWORKS
from rfp._src.nn import MLP

# LOSSES
from rfp._src.losses import feature_map_loss, sqr_error, supervised_loss_time

# FEATURE MAP
from rfp._src.featuremap import neuralODE

# TRAIN
from rfp._src.train import trainer

# LINEAR MODEL
from rfp._src.linear_model import linear_model_time

# UTILS
from rfp._src.utils import batch_sample_time
