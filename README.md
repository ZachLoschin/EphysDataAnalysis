# EphysDataAnalysis
Rotation data analysis for Economo lab

Very basic utility functions made so far in zutils.py

### (Susu Chen, Thinh Nguyen, Nuo Li, Karel Svoboda)

## [LINK TO DATASET](https://dandiarchive.org/dandiset/000363?search=susu+chen&pos=1)

## DOWNLOAD INSTRUCTIONS

- you can download from the link above directly
- alternatively, after pip installing `dandi` you can run the following command:
	- `dandi download DANDI:000363 -o path/to/output/folder`

## DANDI - https://www.dandiarchive.org/handbook/12_download/

#### DOWNLOAD DATA 
- `dandi download DANDI:datasetID`

- to specify an output path `dandi download DANDI:000363 -o output/path/here`           (NOTE: downloads all data stored in dandi repository)
	
- use a url to the specific dataset you want to download if you don't want to download entire dataset

- you can also point to the data via a url rather than downloading. See here: https://pynwb.readthedocs.io/en/stable/tutorials/general/read_basics.html

- it seems that you should be able to download one subfolder according to their documention but it doesn't seem to be working. I will look into it - Zach (for now you can manually download trials from the site)
