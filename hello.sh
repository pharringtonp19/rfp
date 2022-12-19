#! /bin/bash

# ----- ACTIVATE THE VIRTUAL ENVIRONMNET 
# source $(poetry env info --path)/bin/activate

# ----- Delete pyenv virtual environments
# pyenv virtualenv-delete <venv-name>

# poetry config virtualenvs.in-project true
# pyenv local 3.10.4 
# poetry env use 3.10.4 
# poetry install 
# source $(poetry env info --path)/bin/activate
# export PATH="$HOME/.local/bin:$PATH"

# https://www.youtube.com/watch?v=vQv4W-JfrmQ&list=PLS1QulWo1RIYmaxcEqw5JhK3b-6rgdWO_&index=2
echo "--Updating Jax!--"
poetry add jax@latest 

echo "--Updating Jaxlib!--"
poetry add jaxlib@latest


### ------------- OUTSIDE OF POETRY 
# Homebrew --> Pyenv --> Python --> Poetry 


# Upgrade pyenv -- https://stackoverflow.com/questions/43993714/why-is-python-3-6-1-not-available-in-pyenv
# brew upgrade pyenv

# pyenv local 3.7 followed by poetry env use $(which python)