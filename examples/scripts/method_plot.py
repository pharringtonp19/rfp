import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

github_folder = str(Path(os.getcwd()).parent.absolute())
file_link: str = github_folder + "/isga_paper/examples/"
np_file_link: str = os.getcwd() + "/examples/data/"

prediction_loss = np.load(np_file_link + "method_svl_prediction.npy")
regularization = np.load(np_file_link + "method_svl_reg.npy")
fig = plt.figure(dpi=300, tight_layout=True)
plt.plot(prediction_loss, label="Empirical")
plt.plot(regularization, label="Regularization")
plt.legend(frameon=False)
plt.title("Loss", size=14, loc="left")
plt.xlabel("Epochs", size=14)
filename = file_link + "method_svl.png"
fig.savefig(filename, format="png")
plt.show()
