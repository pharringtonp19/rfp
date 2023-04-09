"""Public API for rfp."""


from rfp._src.featuremap import predict
from rfp._src.losses import Cluster_Loss, Supervised_Loss, loss_fn_real
from rfp._src.nn import MLP
from rfp._src.simulated_data import gp_data
from rfp._src.train import Trainer
from rfp._src.types import Array, Params
from rfp._src.utils import Model_Params

# import rfp._src.simulated_data as simulated_data
# from rfp._src.simulated_data import gp_data

# Examples
# import rfp._src.ff1 as ff1
# import rfp._src.ode1 as ode1
# import rfp._src.parallel as parallel
#

# # FEATURE MAP
# from rfp._src.featuremap import NeuralODE, predict

# # LINEAR MODEL
# from rfp._src.linear_model import linear_map

# # LOSSES
# from rfp._src.losses import (
#     Cluster_Loss,
#     Sqr_Error,
#     Supervised_Loss,
#     loss_fn_binary,
#     loss_fn_real,
# )

# # NEURAL NETWORKS
# from rfp._src.nn import MLP

# # TRAIN
# from rfp._src.train import Trainer
# from rfp._src.types import Data

# # UTILS
# from rfp._src.utils import (
#     Model_Params,
#     batch_sample_weight,
#     compute_cost_analysis,
#     init_ode1_model,
#     split,
#     split_weight,
#     store_time_results,
#     time_grad,
# )

# DATA
