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


"""
Utility Functions
"""
def prep_data():
    # I want this to be a master function that can prepare raw imports
    # for multiple types of analysis given kwargs
    return 0

def get_electrode_locations(df):
    # df - pandas.DataFrame - DataFrame generated by load_data()
    # Pass in df and get another column in unit electrodes
    return 0

def coord_to_region(coord):
    # Pass in a set of electrode coordinates and get its brain region
    # Using the mouse common coordinate framework
    return 0


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



def get_distance_from_3D_coords(ar):
    # ar - array that contains x y z coords
    # returns the points distance from the origin
    x2 = ar[0]*ar[0]
    y2 = ar[1]*ar[1]

    # Only save coordinates with a confidence value of >= 0.9
    if ar[2] >=.09:
        return math.sqrt(x2 + y2)
    else:
        return np.nan


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

    # Convert XYZ coordinates to distance from origin
    for i in range(len(BehavioralData["TimeStamps"])):
        tongueDist.append(get_distance_from_3D_coords(tongueCoords[i]))
        noseDist.append(get_distance_from_3D_coords(noseCoords[i]))
        jawDist.append(get_distance_from_3D_coords(jawCoords[i]))
    
    BehavioralData["TongueDist"] = tongueDist
    BehavioralData["NoseDist"] = noseDist
    BehavioralData["JawDist"] = jawDist
    BehavioralData["TongueDist_Norm"] = np.mean(tongueDist)
    BehavioralData["NoseDist_Norm"] = np.mean(noseDist)
    BehavioralData["JawDist_Norm"] = np.mean(jawDist)
    
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