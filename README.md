# Calibrated Concepts
upenn CIS 6200 FA 24 final project

## Installation
All code was run on Python 3.12.3 with the following setup:
```{bash}
# 1. clone the repo
git clone git@github.com:cassandra-goldberg/Calibrated_Concepts.git
cd Calibrated_Concepts

# 2. make the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# if you don't need to re-generate the embeddings the following are sufficient
pip install numpy pandas matplotlib scipy scikit-learn torch netcal seaborn

# else fixme
```

Notebooks can be run using VSCode, Jupyter Notebook, etc. For example:
```{bash}
pip install notebook ipykernel                     # install (one time)
python3 -m ipykernel install --user --name=.venv   # install (one time)
jupyter-notebook                                   # start session (every time)
```
After opening a notebook in the UI, it may be necessary to manually change the 
kernel (toolbar > `Kernel` > `Change kernel` > `.venv`).

## Usage
The main handler file is `Experiments/Experiment.ipynb`. Further usage 
instructions (e.g. settings) can be found in the inline Markdown documentation 
of individual files.

