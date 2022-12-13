#! /bin/bash

# https://www.youtube.com/watch?v=vQv4W-JfrmQ&list=PLS1QulWo1RIYmaxcEqw5JhK3b-6rgdWO_&index=2
echo "--Updating Jax!--"
poetry add jax@latest 

echo "--Updating Jaxlib!--"
poetry add jaxlib@latest


### ------------- OUTSIDE OF POETRY 
# Upgrade pyenv -- https://stackoverflow.com/questions/43993714/why-is-python-3-6-1-not-available-in-pyenv
# brew upgrade pyenv