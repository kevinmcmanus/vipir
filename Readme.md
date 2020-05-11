# Vipir Object Detection

Custom object detector to extract radar traces from ionograms produced by Vipir instruments.

The main class is the Vipir class which is created from netCDF files.  Methods include finding objects (traces) within ionograms and characterizing the objects thus found.

## Project Directory Structure

```
    ./
    |--->data  #raw and cooked, not saved in repo
    |--->src   #python source and bash scripts
    |--->models # inference graphs from trained models, also tarballs of same; not saved
    |--->notebooks #jupyter notebooks of various project facets
    |--->input_archives #copies of various downloaded file, eg, tensorflow/models
    |--->outreach #output artifacts to be shared with collaborators
    
```
