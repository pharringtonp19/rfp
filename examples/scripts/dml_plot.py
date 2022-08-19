import os 
from pathlib import Path
import jax 
import jax.numpy as jnp 
import numpy as np 
import matplotlib.pyplot as plt 

github_folder = str(Path(os.getcwd()).parent.absolute())
file_link: str = github_folder + "/isga_paper/examples/"
np_file_link : str =  os.getcwd() + "/examples/data/"

results = np.load(np_file_link+"dml.npy")
div = jnp.max(results)
fig = plt.figure(dpi=300, tight_layout=True)
plt.hist(results, edgecolor='black', density=True, bins=40)
plt.title('Density', loc='left', size=14)
plt.xlabel(r'$(\hat{\theta} - \theta_0)/\sigma(\hat{\theta})$')
plt.xlim(-div-0.5, div+0.5)
plt.axvline(0, linestyle='--', color='black')
filename = file_link + "dml.png"
fig.savefig(filename, format="png")
plt.show()
