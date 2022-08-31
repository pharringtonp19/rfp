"""Public API for rfp."""

# Examples
import rfp._src.ff1 as ff1
import rfp._src.ode1 as ode1
import rfp._src.parallel as parallel

from rfp._src.data import f1, sample1, sample2, sample3, sample4

# FEATURE MAP
from rfp._src.featuremap import neuralODE

# LINEAR MODEL
from rfp._src.linear_model import linear_model_time, linear_model_trainable_time

# LOSSES
from rfp._src.losses import (
    Cluster_Loss_ff,
    Supervised_Loss_Time,
    feature_map_loss,
    Sqr_Error
)

# NEURAL NETWORKS
from rfp._src.nn import MLP

# TRAIN
from rfp._src.train import Trainer
from rfp._src.types import Data

# UTILS
from rfp._src.utils import (
    Model_Params,
    batch_sample_time,
    batch_sample_weight,
    init_ode1_model,
    split,
    split_weight,
    time_grad,
    store_time_results, 
    time_grad_pvmap
)

# DATA
