"""
Import Libraries
"""

"""
Remove units with spike rates less than 1Hz average across the session --> not real units most likely
Remove stimulus areas from the data --> affect neural behavior and cloud our correlation interpretations
Create heat map in bins --> We won't have a unit in every area so bin the data. Matrix should be ML x DV x AP
Try to find way to convert allen coordinates to bregma overlay
"""

import zutils
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import pandas as pd
from time import sleep
import os
from pynwb import NWBHDF5IO
import numpy as np
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from scipy.stats import linregress
from datetime import datetime
from scipy.stats.stats import pearsonr

"""
Load the Data -- Currently includes 1 session of data including recordings from L and R midbrain and L and R ALM
"""
# Data Folder
dataDir = 'C:\\Users\\zachl\\OneDrive\\BU_YEAR1\\Economo_Rotation\\data\\'

# File Parameters
sub = ['441666']
date = ['20190516']

# LOAD DATA
data = zutils.load_data(dataDir, sub, date)

# Get first session from Data
sub1 = data[0]


""" 
Put data in nice format
"""
# Save the Behavioral Data
movementDF = zutils.get_behavioral_timeseries(sub1)

# Get Unit Locations
unitDF = zutils.saveCCFCoordsAndRegion(sub1,'./','./',sub[0],date[0])

# Look at the unique regions that these units span
uniqueRegions = np.unique(unitDF["region"])

# We want to look at the Midbrain reticular nucleus
midbrainDF = unitDF[unitDF['region'].str.contains('.*Midbrain reticular nucleus.*', case=False, regex=True)]
# midbrainDF = unitDF[unitDF['region'].str.contains('.*colliculus.*', case=False, regex=True)]

"""
For a number of units get the behavioral and neural data
"""
# Resample the behavioral data
movementDF.set_index('TimeStampsIndex', inplace=True)
resampled_MovementDF = movementDF.resample('50L').mean()
print(len(movementDF))
print(len(resampled_MovementDF))

data_storage = pd.DataFrame()
data_storage["TimeStamps"] = resampled_MovementDF["TimeStamps"]
data_storage["TongueDist"] = resampled_MovementDF["TongueDist"]
data_storage["TongueDist"] = np.nan_to_num(data_storage["TongueDist"], copy=True, nan=0.0, posinf=None, neginf=None)
data_storage["TongueDist"][np.isneginf(data_storage["TongueDist"])] = 0
data_storage["TongueDist"][np.isinf(data_storage["TongueDist"])] = 0

locations = pd.DataFrame(columns=["ML", "DV", "AP"])
ml = []
dv = []
ap = []
rValue = []
pValue = []

for UoI in range(len(midbrainDF)):

    try:
        unitsRegDF = midbrainDF
        # UoI = 30 # unit of interest

        
        ml.append(unitsRegDF.iloc[UoI]["x"])
        dv.append(unitsRegDF.iloc[UoI]["y"])
        ap.append(unitsRegDF.iloc[UoI]["z"])

        spikeTimes = midbrainDF["spike_times"]

        spikeMatrix = []
        for row in spikeTimes:
                spikeMatrix.append(row)

        # Get Neural spiking data
        spikeDF = zutils.get_firing_rate_from_spikes(spike_vec= spikeMatrix[UoI], binsize=50)  #ms

        """
        Data Resampling
        """
        # Resample the neural data
        # Set the 'timestamp' column as the index
        spikeDF.set_index('TimeStampsIndex', inplace=True)

        # Resample to 50ms intervals
        resampled_df = spikeDF.resample('50L').mean()  # 'L' stands for milliseconds

        """
        Add zeros to the end of the neural data to make length same as behavioral
        """
        # Specify the number of zero rows you want to add
        num_zeros = len(resampled_MovementDF) - len(resampled_df)

        # Generate timestamps for the new rows (increasing from the last timestamp in original_df)
        last_timestamp = resampled_df['TimeStamps'].iloc[-1]
        new_timestamps = np.arange(last_timestamp + 0.05, last_timestamp + (num_zeros * 0.05) + 0.05, 0.05)

        # Create a DataFrame with the specified number of rows of zeros and increasing timestamps
        zero_data = {
            'SpikeRate': np.zeros(len(new_timestamps)),
            'TimeStamps': new_timestamps
        }
        zero_df = pd.DataFrame(zero_data)

        # Concatenate the original DataFrame with the DataFrame containing zero measurements
        neural_extended = pd.concat([resampled_df, zero_df], ignore_index=True)

        # Delete last row of dataframe to correct size
        if len(neural_extended) != len(data_storage):
            neural_extended = neural_extended.drop(neural_extended.index[-1])
        
        neural_extended["SpikeRate"] = np.nan_to_num(neural_extended["SpikeRate"], copy=True, nan=0.0, posinf=None, neginf=None)
        neural_extended["SpikeRate"][np.isneginf(neural_extended["SpikeRate"])] = 0
        neural_extended["SpikeRate"][np.isinf(neural_extended["SpikeRate"])] = 0

        # If the indices don't match, set the index of data_storage to match neural_extended
        data_storage.set_index(neural_extended.index, inplace=True)

        # Add this to the dataframe
        data_storage[UoI] = neural_extended["SpikeRate"]

        # Apply finite mask
        finiteYmask = np.isfinite(data_storage["TongueDist"])

        # Correlate the data
        reg_output = pearsonr(data_storage["TongueDist"][finiteYmask], neural_extended["SpikeRate"][finiteYmask])
        rValue.append(reg_output[0])
        pValue.append(reg_output[1])
    except:
         print("EXCEPTION")

# Save the data

data_storage.to_csv("CorrelationData.csv")

locations["ML"] = ml
locations["DV"] = dv
locations["AP"] = ap
locations["rValue"] = rValue
locations["pValue"] = pValue

locations.to_csv("unit_locations.csv")