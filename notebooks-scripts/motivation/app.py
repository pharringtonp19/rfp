import jax 
import jax.numpy as jnp
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 


import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'viridis'
rcParams['axes.grid'] = False
plt.style.use('seaborn-v0_8-dark-palette')

from matplotlib import font_manager 
locations = './styles/Newsreader'
font_files = font_manager.findSystemFonts(fontpaths=locations)
print(locations)
print(font_files[0])
for f in font_files: 
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Newsreader"
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import matplotlib.ticker as mtick 


def main_effect(x, p):
    x = x*p
    return jnp.log(x**2 + 1.0 + jnp.sin(x*3.0)) + 1.5

def main():
    st.title("Regularizing the Forward Pass")

    """
    In our introductory courses on statistics, we often think of observations as being drawn independently from the same distribution. 
    In practice though, this is may not be the case as observations may be associated with distinct clusters, each with their own distribution.
    """

    """
    A concrete example of this is to consider the situation where the conditional expectation functions differ across clusters. 
    We can represent this difference by assigning each cluster a parameter and constructing a map between the parameter and the 
    conditional expectation function.
    """

    st.latex(r'''
    \theta \longmapsto \Big( x \mapsto \log((x\theta)^2 + 1 + \sin(3x\theta)) + 1.5 \Big)
    ''')

    """
    The following is an interactive plot that allows you to explore the conditional expectation functions for five different clusters.
    """
    col1, col2, col3, col4, col5= st.columns(5)
    c1 = col1.slider('', 0.5, 1.5, .5, step=0.1)  # 👈 this is a widget
    c2 = col2.slider('', 0.5, 1.5, 0.75, step=0.1)  # 👈 this is a widget
    c3 = col3.slider('', 0.5, 1.5, 1., step=0.1)  # 👈 this is a widget
    c4 = col4.slider('', 0.5, 1.5, 1.25, step=0.1)  # 👈 this is a widget
    c5 = col5.slider('', 0.5, 1.5, 1.5, step=0.1)  # 👈 this is a widget

    
    # Display the plot
    fig = plt.figure(dpi=300, tight_layout=True, figsize=(7, 4.5))
    ax = plt.axes(facecolor=(.95, .96, .97))
    ax.xaxis.set_tick_params(length=0, labeltop=False, labelbottom=True)
    for key in 'left', 'right', 'top':
        ax.spines[key].set_visible(False)
    ax.set_title('Conditional Expectation Function', size=16, loc='center', pad=20)
    ax.text(0., 1.02, s='Value', transform=ax.transAxes, size=14)
    ax.yaxis.set_tick_params(length=0)
    ax.yaxis.grid(True, color='white', linewidth=2)
    key = jax.random.PRNGKey(0)
    xs = jnp.linspace(-3., 3., 1000)
    ys = main_effect(xs, c1)
    ax.plot(xs, ys)
    ys = main_effect(xs, c2)
    ax.plot(xs, ys)
    ys = main_effect(xs, c3)
    ax.plot(xs, ys)
    ys = main_effect(xs, c4)
    ax.plot(xs, ys)
    ys = main_effect(xs, c5)
    ax.plot(xs, ys)

    ax.set_xlabel("Feature Space", size=14)
    plt.ylim(0, 6)
    st.pyplot(fig)

if __name__ == "__main__":
    main()