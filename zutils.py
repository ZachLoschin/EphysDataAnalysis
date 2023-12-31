"""
Zachary Loschinskey
Dr. Economo
Boston University
Fall 2023

Description: Utility Functions (zutils) for working with Ephys Data
"""


"""
Library Importation
"""
from tqdm import tqdm
from scipy import stats
import pandas as pd
from time import sleep
import os
from pynwb import NWBHDF5IO
import numpy as np
import math
import matplotlib.pyplot as plt
import nrrd # pip install pynrrd
from pathlib import Path
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from datetime import datetime

"""
Plotting Functions
"""
def resample_behavioral_Data(movementDF):

    # Define the desired bin width in seconds (e.g.,100 ms)
    bin_width = 0.1

    # Determine the total duration of the movement data
    total_duration = movementDF["TimeStamps"].max() - movementDF["TimeStamps"].min()

    # Calculate the number of bins needed
    num_bins = int(total_duration / bin_width)

    # Create an empty DataFrame to store the resampled data
    resampledDF = pd.DataFrame()

    # Create an empty array to store the time vector
    time_vector = np.linspace(movementDF["TimeStamps"].min(), movementDF["TimeStamps"].max(), num_bins)

    # Loop through the time bins and calculate the representative "TongueDist" value
    for bin_start, bin_end in zip(time_vector[:-1], time_vector[1:]):
        # Filter the rows in 'movementDF' that fall within the current time bin
        bin_data = movementDF[(movementDF["TimeStamps"] >= bin_start) & (movementDF["TimeStamps"] < bin_end)]
        
        # Calculate the representative "TongueDist" value (e.g., mean)
        if not bin_data.empty:
            representative_value = bin_data["TongueDist"].mean()  # You can use other aggregation methods
            
            # Add the representative value and the bin's midpoint time to 'resampledDF'
            resampledDF = resampledDF.append({"TimeStamps": (bin_start + bin_end) / 2, "TongueDist": representative_value}, ignore_index=True)

    return resampledDF


def get_firing_rate_from_spikes(spike_vec, binsize):
    # spike_vec - np.array - raw spike timings
    # binsize   - int      - binsize for calculation in ms
    # Determine the total duration of the spike data
    # Resolution in s

    total_duration = spike_vec[-1]

    # Define the bin width (50 ms) in seconds
    bin_width = binsize / 1000

    # Calculate the number of bins needed
    num_bins = int(total_duration / bin_width)

    # Create time vector
    time_vector = np.linspace(0, total_duration, num_bins)

    # Create an empty array to store spike rates
    spike_rates = np.zeros(num_bins)

    # Loop through the spike timings and count spikes in each bin
    for spike_time in spike_vec:
        bin_index = int(spike_time / bin_width)
        if bin_index < num_bins:
            spike_rates[bin_index] += 1

    # Divide the spike counts by bin width to get spike rates
    spike_rates /= bin_width

    # Convert to datetime
    time_vector = np.array(time_vector)
    time_vector_index = np.array([datetime.fromtimestamp(timestamp) for timestamp in time_vector])


    spikeDF = pd.DataFrame()
    spikeDF["SpikeRate"] = spike_rates
    spikeDF["TimeStamps"] = time_vector
    spikeDF["TimeStampsIndex"] = time_vector_index
    return spikeDF

def spike_raster_and_behavior_plot(df, unitsRegDF, t0, tf):
    # df         - Pandas.DataFrame - generated by load_data()
    # unitsRegDF - Pandas.DataFrame - generated by saveCCFCoordsAndRegion() and filtering to desired regions

    # midbrainDF = unitDF[unitDF['region'].str.contains('.*Midbrain reticular nucleus.*', case=False, regex=True)]

    """ 
    Prepare space for plotting
    """
    # Clear Existing Figure
    plt.clf()

    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=False)

    """ 
    Plot the Spiking Data
    """
    # Add Rows to the spike matrix

    spikeTimes = unitsRegDF["spike_times"]

    spikeMatrix = []
    for row in spikeTimes:
            spikeMatrix.append(row)
    print(spikeMatrix[0])

    # Loop to create the raster for each row
    for i in range(len(spikeMatrix)):  # len(spikeMatrix)
        # Create Y ticks
        yValue = np.ones(len(spikeMatrix[i])) * i

        # Plot Scatter
        axs[0].scatter(np.array(spikeMatrix[i]), yValue, marker='.', color='b', s=2)


    """
    Plot Movement Data
    """
    # Get Behavioral Data
    movementDF = get_behavioral_timeseries(df)
    print(movementDF)
    # Plot the tongue distance
    axs[1].plot(movementDF["TimeStamps"], movementDF[f'TongueDist'], color='black')

    # Plot the jaw distance
    axs[1].plot(movementDF["TimeStamps"], movementDF[f'JawDist'], color='b')

    # Plot the nose distance
    axs[1].plot(movementDF["TimeStamps"], movementDF[f'NoseDist'], color='r')

    maxVal = max(np.nanmax(movementDF[f'TongueDist']), np.nanmax(movementDF[f'JawDist']), np.nanmax(movementDF[f'NoseDist']))
    minVal = max(np.nanmin(movementDF[f'TongueDist']), np.nanmin(movementDF[f'JawDist']), np.nanmin(movementDF[f'NoseDist']))


    """ 
    Customizing the Plot Settings
    """
    # Remove the x-axis from the first subplot
    axs[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # Set the x-axis limits based on your data
    axs[0].set_xlim(t0, tf)
    axs[0].set_title("Neural Spiking Raster")

    # Add labels and legend
    axs[0].set_ylabel('Neuron')

    # Set xlim and ylim
    axs[1].set_xlim(t0, tf)
    axs[1].set_ylim(0, 500)
    axs[1].legend(['Tongue Distance', 'Jaw Distance', 'Nose Distance'])

    # Add labels
    axs[1].set_xlabel('Time (S)')
    axs[1].set_ylabel('Distance (units)')

    # Add a grid to the second subplot
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # Adjust the spacing between subplots (bottom and top)
    plt.subplots_adjust(hspace=-0)

    # Show the figure
    plt.show()


"""
Utility Functions
"""



def saveCCFCoordsAndRegion(nwbfile,saveDir,ccfDir,sub,date):
    # get x,y,z coords in Allen CCF space for each unit/electrode
    units = nwbfile.units
    unit_id = units.unit.data[:]
    unit_electrodes = units.electrodes.data[:]
    
    # get allen ccf registered electrode coords
    electrodes = nwbfile.electrodes
    x = electrodes.x.data[:].astype(int) # ML in Allen CCF
    y = electrodes.y.data[:].astype(int) # DV in Allen CCF
    z = electrodes.z.data[:].astype(int) # AP in Allen CCF
    
    # get those coordinates for each unit
    x = x[unit_electrodes]
    y = y[unit_electrodes]
    z = z[unit_electrodes]
    
    # positions for electrodes outside brain are encoded as a very negative number. set these to 0
    x[x<0] = 0
    y[y<0] = 0
    z[z<0] = 0
    
    # save to dataframe
    df = pd.DataFrame()
    df['unit'] = unit_id
    df['x'] = x
    df['y'] = y 
    df['z'] = z 
    df['electrodes'] = unit_electrodes
    df['spike_times'] = nwbfile.units["spike_times"][:]
    df["sf"] = nwbfile.units.sampling_rate.data[:]

    # get annotations
    annofile = 'annotation_10.nrrd' # 2017 - can download this using ReferenceSpaceCache.get_annotation_volume() or straight from server
    # http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/
    anno,header = nrrd.read(os.path.join(ccfDir, annofile))   
    # anno[0,:,:] coronal section
    # anno[:,0,:] dv section
    # anno[:,:,0] sagittal section
    
    # get reference space and name map (dictionary from structure id in annotations to name of structure)
    reference_space_key = os.path.join('annotation', 'ccf_2017')
    resolution = 10
    rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest=Path(ccfDir) / 'manifest.json')

    # ID 1 is the adult mouse structure graph
    tree = rspc.get_structure_tree(structure_graph_id=1) 
    name_map = tree.get_name_map()
    
    # get structure ids for each unit/electrode
    struct_id = np.zeros_like(x)
    for i in tqdm(range(len(struct_id))):
        pos = (np.array(df.iloc[i,1:4])[::-1] / resolution).astype(int) # get positions, divide by resolution, reverse to put in ccf coords            
        struct_id[i] = anno[pos[0],pos[1],pos[2]]

    region = [name_map[sid] if sid!=0 else '0' for sid in struct_id]
    
    df['id'] = struct_id
    df['region'] = region
    
    # savedir = os.path.join(saveDir,'sub-'+sub)
    # df.to_csv(os.path.join(savedir,'sub-' + sub + '_ses-' + date + '_ccfcoords.csv'),index=False)
    return df


def find_sessions_by_region(metaData, reg):
    # metaData - pandas.DataFrame - resulting meta data from allSessionsMetaData.csv
    # reg      - string           - string of region to search for in all metadata
    # Returns a list of the sub names and dates of trials that include the brain region of interest
    
    # Storage for subs and dates
    subList = []
    dateList = []

    filteredDF = metaData

    # Enter the metaDF
    for i in range(len(metaData)):
        tempDF = metaData.iloc[i]

        if reg in tempDF["regions"]:
            containsRegion = 1
            subList.append(tempDF['sub'])
            dateList.append(tempDF["date"])
        else:
            filteredDF = filteredDF.drop(index=i)
    
    return subList, dateList, filteredDF

def plot_movement(movementDF, part, range):
    # movementDF - DataFrame - the movement dataframe generated by get_behavioral_timeseries()
    # part       - string    - the part of the body for the plot ["Tongue, Jaw, Nose"]
    # range      - list(2)   - range of the movement in seconds to plot

    # Make a Nice plot of the movement
    plt.plot(movementDF["TimeStamps"][range[0]:range[-1]], movementDF[f'{part}Dist'][range[0]:range[-1]])


def help_find_tongue_basline(ar):
    # ar - array that contains x y z coords
    
    # returns the points distance from the origin
    x2 = ar[0]*ar[0]
    y2 = ar[1]*ar[1]

    # Only save coordinates with a confidence value of >= 0.9
    if ar[2] >=.9:
        return math.sqrt(x2 + y2), x2, y2
    else:
        print(f"WE DONE IT {x2} {y2}")
        return np.nan, x2, y2  # return nan if not high confidence, and return x y coords
    

def get_distance_from_3D_coords(ar, x, y):
    # ar - array that contains x y z coords

    # returns the points distance from the origin
    x2 = ar[0]
    y2 = ar[1]

    # Calculate point distsances squared
    xDistSquared = (x2 - x)*(x2 - x)
    yDistSqaured =(y2 - y)*(y2-y)

    # Only save coordinates with a confidence value of >= 0.9
    if ar[2] >=.9:
        return math.sqrt(xDistSquared + yDistSqaured)
    else:
        return np.nan  # If confidence is low


def get_behavioral_timeseries(df):
    # df - raw import of nwb file
    # given an imported nwb dataframe, returns the tongue, jaw, and nose timeseries data
    # This function assumes the tongue, the nose, and the jaw each have a base coordinate value of 0,0,0 for
    # the resting position. --> Discuss this with Munib and Yujin 

    # Get XYZ coordinates
    tongueCoords = df.acquisition["BehavioralTimeSeries"]["Camera0_side_TongueTracking"].data[:]
    noseCoords = df.acquisition["BehavioralTimeSeries"]["Camera0_side_NoseTracking"].data[:]
    jawCoords = df.acquisition["BehavioralTimeSeries"]["Camera0_side_JawTracking"].data[:]

    # Initialize output df
    BehavioralData = pd.DataFrame()
    tongueDist = []
    noseDist = []
    jawDist = []

    # TimeStamps are in seconds! 294.118 Hz Sampling rate? point every 0.0034 seconds
    BehavioralData["TimeStamps"] = df.acquisition["BehavioralTimeSeries"]["Camera0_side_TongueTracking"].timestamps[:]
    BehavioralData["TimeStamps"]= np.array(BehavioralData["TimeStamps"])
    BehavioralData["TimeStampsIndex"] = np.array([datetime.fromtimestamp(timestamp) for timestamp in BehavioralData["TimeStamps"]])

    # Convert XYZ coordinates to distance from origin
    for i in range(len(BehavioralData["TimeStamps"])):
        # Find Dist from origin in nose and jaw
        noseDist.append(get_distance_from_3D_coords(noseCoords[i],0,0))
        jawDist.append(get_distance_from_3D_coords(jawCoords[i],0,0))
        tongueDist.append(get_distance_from_3D_coords(tongueCoords[i],0,0))

    # Noramlize the tongue data
    print(f"min {np.nanmin(tongueDist)}")
    tongueDist = np.array(tongueDist) - np.nanmin(tongueDist)
    tongueDist = np.nan_to_num(tongueDist)

    BehavioralData["TongueDist"] = tongueDist
    BehavioralData["NoseDist"] = noseDist
    BehavioralData["JawDist"] = jawDist
    
    # Normalize the Data

    # # Take means ignoring nan values
    # meanJaw = np.nanmean(jawDist)
    # meanTongue = np.nanmean(tongueDist)
    # meanNose = np.nanmean(noseDist)

    # # Create a copy of jawDist to avoid modifying the original data
    # BehavioralData["JawDist_Normalized"] = jawDist.copy()
    # BehavioralData["TongueDist_Normalized"] = tongueDist.copy()
    # BehavioralData["NoseDist_Normalized"] = noseDist.copy()

    # # Create a mask to identify non-nan values in the 'jawDist' column
    # nonNanMaskJaw = ~np.isnan(BehavioralData['JawDist'])
    # # Normalize the data, taking care of np.nan values
    # BehavioralData.loc[nonNanMaskJaw, 'JawDist_Normalized'] = BehavioralData['JawDist'][nonNanMaskJaw] - meanJaw
    # BehavioralData['JawDist_Normalized'] /= np.nanstd(BehavioralData['JawDist'])
    # # Handle np.nan values in the resulting column
    # BehavioralData.loc[~nonNanMaskJaw, 'JawDist_Normalized'] = np.nan


    # # Create a mask to identify non-nan values in the 'NoseDist' column
    # nonNanMaskNose = ~np.isnan(BehavioralData['NoseDist'])
    # # Normalize the data, taking care of np.nan values
    # BehavioralData.loc[nonNanMaskNose, 'NoseDist_Normalized'] = BehavioralData['NoseDist'][nonNanMaskNose] - meanNose
    # BehavioralData['NoseDist_Normalized'] /= np.nanstd(BehavioralData['NoseDist'])
    # # Handle np.nan values in the resulting column
    # BehavioralData.loc[~nonNanMaskNose, 'NoseDist_Normalized'] = np.nan

    # # Create a mask to identify non-nan values in the 'TongueDist' column
    # nonNanMaskTongue = ~np.isnan(BehavioralData['TongueDist'])
    # # Normalize the data, taking care of np.nan values
    # BehavioralData.loc[nonNanMaskTongue, 'TongueDist_Normalized'] = BehavioralData['TongueDist'][nonNanMaskTongue] - meanTongue
    # BehavioralData['TongueDist_Normalized'] /= np.nanstd(BehavioralData['TongueDist'])
    # # Handle np.nan values in the resulting column
    # BehavioralData.loc[~nonNanMaskJaw, 'TongueDist_Normalized'] = np.nan

    return BehavioralData


def find_nwb(dataDir,sub,date):
    # dataDir - directory where data lives, should be where subject directories live
    # sub     - list of subjects
    # date    - list of dates for each subject
    # given dates corresponding to a session, returns the full name of the .nwb session files
    
    sessionListTemp = os.listdir(os.path.join(dataDir,"sub-"+sub))
    sessionList = []

    for i in range(len(sessionListTemp)):
        if date[i] in sessionListTemp[i]:
            sessionList.append(sessionListTemp[i])

    return sessionList


def load_data(dataDir, sub, date):
    # datadir - Directory storing all data regarding the project, contains subdirectories named as subnumbers
    # subdir  - List of the subdirectory(s) to enter
    # date    - List of dates to enter for each subdirectory, one row for each subdir

    # Indicate to user
    print(f"Loading data for sub-{sub}:{date} from {os.linesep}Data directory: {dataDir}")

    # Find NWB files in each subdirectory
    sessionList = []

    for i in range(len(sub)):
        # This works for multiple sub folders, but it uses the dates to look at every subdirectory, that needs to be fixed.
        sessionList.append(find_nwb(dataDir, sub[i], date))
    
    sessionList = np.array(sessionList)
    sessionList = sessionList.flatten()
    print(sessionList)
    data = []

    # sub[0] is hard coded. Thus this only works for one session folder with multiple dates...
    for i in range(len(sessionList)):
        file = (f'{dataDir}\\sub-{sub[0]}\\{sessionList[i]}')
        data.append(NWBHDF5IO(file).read())

    return data