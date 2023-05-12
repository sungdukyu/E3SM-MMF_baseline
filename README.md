# E3SM-MMF_baseline

### Dataset Information
Variable list can be found here: https://docs.google.com/spreadsheets/d/1ljRfHq6QB36u0TuoxQXcV4_DSQUR0X4UimZ4QHR8f9M/edit#gid=0

1: E2SM-MMF High-Resolution Real Geography
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
