import os 
from pathlib import Path
import jax 
import jax.numpy as jnp 
import numpy as np 
import matplotlib.pyplot as plt 

github_folder = str(Path(os.getcwd()).parent.absolute())
file_link: str = github_folder + "/isga_paper/examples/"
np_file_link : str =  os.getcwd() + "/examples/data/"

prediction_loss = np.load(np_file_link+"method_svl_prediction.npy")
regularization = np.load(np_file_link+"method_svl_reg.npy")
fig = plt.figure(dpi=300, tight_layout=True)
plt.plot(prediction_loss)
plt.plot(regularization)
filename = file_link + "method_svl.png"
fig.savefig(filename, format="png")
plt.show()