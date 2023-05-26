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
(local machine) in your conda base environment do the following first:
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

```
# e3sm supercloud
conda deactivate
module load cuda/11.2
conda create -n e3sm python=3.8.5
conda activate e3sm
conda install -y -c conda-forge numpy 
conda install -y -c conda-forge pandas
conda install -y -c conda-forge xarray netCDF4 bottleneck
conda install -y -c conda-forge scikit-learn statsmodels scipy
conda install -y -c conda-forge matplotlib seaborn 
conda install -y -c conda-forge jupyterlab tqdm
conda install -y -c conda-forge dask hdf5 h5py 
conda install -y -c conda-forge ipython yaml # All packages already installed
pip install --upgrade pip
mkdir /state/partition1/user/$USER # Only necessary on my supercomputer
export TMPDIR=/state/partition1/user/$USER # Only necessary on my supercomputer
pip install --no-cache-dir tensorflow # --no-cache-dir only necessary on my supercomputer
pip install keras-tuner --upgrade
pip install tensorflow-addons
pip install qhoptim
pip install jupyter
# Launch note book at 
https://txe1-portal.mit.edu/jupyter/jupyter_notebook.php
# todo: test gpu.

# Remote installation: e3sm-python
# From conda (base) env:
module load cudnn/8.2.1_cuda11.2
conda clone --name e3sm-python base
conda activate e3sm-python
conda install -y -c conda-forge xarray netCDF4 bottleneck
# pip install --upgrade pip # Make sure this is run inside the conda env.
pip install tensorflow
pip install keras-tuner --upgrade
pip install tensorflow-addons
pip install qhoptim
--
module deactivate
module unload
conda create -n e3sm-python python=3.8.5
conda activate e3sm-python # recreates error
# From conda (base) evn
module load anaconda3/2020.11
conda create -n e3sm-python python=3.8.5
source activate e3sm-python # recreates error
module load anaconda3/2020.11
conda install -y -c conda-forge python=3.8.5 # recreates error
conda create -n e3sm-python python=3.8.5 # recreates error

# Local Installation
conda activate base
conda create -n e3sm python=3.8.5
conda activate e3sm
conda install -y -c conda-forge numpy pandas 
conda install -y -c conda-forge xarray 
conda deactivate 
conda update -n base conda
conda activate e3sm
conda install -y -c conda-forge netCDF4
conda install -y -c conda-forge bottleneck 
conda install -y -c conda-forge scikit-learn
conda install -y -c conda-forge statsmodels 
conda install -y -c conda-forge scipy
conda install -y -c conda-forge matplotlib
conda install -y -c conda-forge seaborn
conda install -y -c conda-forge jupyterlab
conda install -y -c conda-forge tqdm
conda install -y -c conda-forge dask
conda install -y -c conda-forge hdf5
conda install -y -c conda-forge h5py
conda install -y -c conda-forge ipython
conda install -y -c conda-forge yaml
pip install tensorflow # Throws error.
pip install protobuf==3.20.* # Fixes error
pip install keras-tuner --upgrade
pip install tensorflow-addons
pip install qhoptim

# The following crashes
conda install -c conda-forge numpy pandas xarray scikit-learn statsmodels scipy matplotlib seaborn jupyterlab tqdm dask hdf5 h5py ipython yaml # Install failed.
pip install tensorflow
pip install keras-tuner --upgrade
pip install tensorflow-addons
pip install qhoptim
conda install -c conda-forge netCDF4 bottleneck 
pip install jupyterlab
pip install pandas xarray scikit-learn
pip install statsmodels matplotlib
pip install seaborn tqdm dask
# pip install hdf5 # Throws an error
pip install ipython 
# pip install yaml # Throws an error
python -m ipykernel install --user --name=e3sm
# Call via $jupyter lab
# Error: Could not find TensorRT. Due to version mismatch of protobuf.
conda remove --name e3sm --all


# Remote installation: e3sm-cudnn
conda deactivate
module load cudnn/8.2.1_cuda11.2
conda create -n e3sm-cudnn
conda activate e3sm-cudnn
# already installed: numpy pandas scikit-learn statsmodels scipy matplotlib seaborn jupyterlab tqdm dask h5py yaml
conda install -c conda-forge xarray 
pip install tensorflow
pip install keras-tuner --upgrade
pip install tensorflow-addons
pip install qhoptim
conda install -c conda-forge ipykernel
python -m ipykernel install --user --name=e3sm-cudnn # This through "Debugger warning: It seems that frozen modules are being used"
conda install -c conda-forge netCDF4 bottleneck 
# Error: $python # throws Segmentation fault
conda install -c conda-forge matplotlib


# e3sm
conda deactivate
module load anaconda3/2020.11
conda create -n e3sm python=3.8.5
conda activate e3sm
conda install -y -c conda-forge numpy 
conda install -y -c conda-forge pandas
conda install -y -c conda-forge xarray netCDF4 bottleneck
conda install -y -c conda-forge scikit-learn statsmodels scipy
conda install -y -c conda-forge matplotlib seaborn 
conda install -y -c conda-forge jupyterlab tqdm
conda install -y -c conda-forge dask hdf5 h5py
conda install -y -c conda-forge ipython yaml # All packages already installed
conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install --upgrade pip # Make sure this is run inside the conda env.
pip install tensorflow
pip install keras-tuner --upgrade
pip install tensorflow-addons
pip install qhoptim
# Link conda environment to jupyter
python -m ipykernel install --user --name=e3sm
module load anaconda3/2020.11 # Needed on my supercomp. to run from command line. Gets rid of $ python throwing seg fault.
conda install -y -c conda-forge xarray

```

# todo:
[x] create nersc user account -> https://docs.nersc.gov/accounts/#obtaining-a-user-account -> https://iris.nersc.gov/add-user 
[] test nersc login -> https://iris.nersc.gov/login
[x] install repo locally -> no space left on device. -> delete Downloads -> conda install takes FOREVER -> update conda? -> works now!
[x] install repo on eofe7 -> e3sm-cudnn installed. e3sm now works too; work from there -> python throws SegFault and tensorflow crashes -> also eofe7 jupyter is so slow.
[x] install repo on txe1 -> transfer data
[] run jupyter notebook on eofe7
  [] notebook dies. Write code to run from command line. -> cant call pdb -> need enviornment that doesn't seg fault. -> copy base to e3sm-python
[x] what's the data i have? 384 columns, 124 mappend onto 384 columns, 128
-> indeed it's 2 vectors + 4 scalers -- to -- 2 vectors and 8 scalars. 
[x] change architecture to 1D cnn on 124, 1 to 128, 1 -> ON LOCAL -> sync to github 
  [x] -> change dataloader to output (60,6) and (60,10)
[x] add multiple linear layers in output -- just one layer should be fine.
[] add SE_block? -- what is that even
[x] (opt) correct to 'valid' padding -- how does salva do it?
[] add learning rate scheduler
[] share with Ritwik -- commit to my github 
[] get more data! -> download 6GB from drive. -> upload 6gb onto eofe

architectures:
- is the input 2x60 + 4x1?
- learn projection from 124x1 to 11x11 then do 2D cnn
- do 2x64 do 1D cnn 

