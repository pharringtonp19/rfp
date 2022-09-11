"""Public API for rfp."""

# Examples
import rfp._src.data as data
import rfp._src.ff1 as ff1
import rfp._src.ode1 as ode1
import rfp._src.parallel as parallel

# FEATURE MAP
from rfp._src.featuremap import neuralODE

# LINEAR MODEL
from rfp._src.linear_model import linear_map

# LOSSES
from rfp._src.losses import (
    Cluster_Loss_ff,
    Sqr_Error,
    Supervised_Loss,
    feature_map_loss,
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
    store_time_results,
    time_grad,
    time_grad_pvmap,
)

# DATA
