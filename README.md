# E3SM-MMF_baseline

### Dataset Information
Variable list can be found here: https://docs.google.com/spreadsheets/d/1ljRfHq6QB36u0TuoxQXcV4_DSQUR0X4UimZ4QHR8f9M/edit#gid=0

1: E3SM-MMF High-Resolution Real Geography
- Two files (one for input; the other for output) are produced at each time step for a 10-year long simulation (with timestep = 20 min.), totaling 525,600 files (= 10 year * 365 days/year * 72 steps/day * 2 files/day)
- Total data volume: 43 TB
- File size:
  - Input: 102 MB/file
  - Output: 61 MB/file
- File format: netcdf
- Dimensions:
  - ncol (horizontal dimension of an unstructured grid): 21600
  - lev (vertical dimension): 60

2: E3SM-MMF Low Resolution Real Geography
- All same as above except for file sizes and dimension sizes.
- Total data volume: 800GB
- File size:
  - Input: 1.9 MB/file
  - Output: 1.1 MB/file
- File format: netcdf
- Dimensions:
  - ncol (horizontal dimension of an unstructured grid): 384
  - lev (vertical dimension): 60

3: E3SM-MMF Low Resolution Aquaplanet
- All same as above except for file sizes and dimension sizes.
- Total data volume: 800GB
- File size:
  - Input: 1.9 MB/file
  - Output: 1.1 MB/file
- File format: netcdf
- Dimensions:
  - ncol (horizontal dimension of an unstructured grid): 384
  - lev (vertical dimension): 60

# Installation
(local machine) in your conda base environment do the following to ensure proper channel management:
```
# Prefer packages in conda-forge
conda config --system --prepend channels conda-forge
# Packages in lower-priority channels not considered if a package with the same
# name exists in a higher priority channel. Can dramatically speed up installations.
# Conda recommends this as a default
# https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html
conda config --set channel_priority strict
conda config --system --set auto_update_conda false
conda config --system --set show_channel_urls true
```

### Install script
todo: move to clean environment.yml
```
# conda deactivate # It's likely not necessary to deactivate base env.
# module load anaconda3/2020.11 # Note my supercomputer loads a base env and this install has only been tested with that. 
conda create -n e3sm python=3.8.5
conda activate e3sm
conda install -c conda-forge numpy 
conda install -c conda-forge pandas
conda install -c conda-forge xarray netCDF4 bottleneck
conda install -c conda-forge scikit-learn statsmodels scipy
conda install -c conda-forge matplotlib seaborn 
conda install -c conda-forge jupyterlab tqdm
conda install -c conda-forge dask hdf5 h5py 
conda install -c conda-forge ipython yaml # All packages already installed
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install --upgrade pip # Make sure this is run inside the conda env.
pip install tensorflow
pip install keras-tuner --upgrade
pip install tensorflow-addons
pip install qhoptim
# Link conda environment to jupyter
python -m ipykernel install --user --name=e3sm
```
