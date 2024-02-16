import numpy as np
import copy
from types import SimpleNamespace

#This is the object of data
DATA = {
    #NEED TO BE LOADED IN
    'trajectories': None,       #coordinates of the data
    'trajectoriesIndex': None,  #index of trajectory number
    'deltaT': None,             #time passed in between each frame of data

    #INITIALIZED INDEPENDENTLY BASED ON PARAMATERS ABOVE
    'nData': None,              #number of datapoints (set to len(nData))
    'nTrajectories': None,      #number of trajectories (set to len(np.unique(data.trajectoriesIndex)))
}

def data_reader(path, scale=1, deltaT = 1/30):
    # Read the CSV file, considering the header
    data = np.genfromtxt(path, delimiter=', ', skip_header=1)

    # Separate columns into individual arrays
    dataVectIndex = data[:, 0]
    dataVect = data[:, 1:]
    
    #localization adjustment to nanometers
    dataVect = dataVect[::]*scale
    dataVectIndex = dataVectIndex[::]
    
    #put time step manually as unavailable from data file
    deltaT = deltaT

    # return dataVect, dataVectIndex, deltaT

    data = SimpleNamespace(**copy.deepcopy(DATA))
    data.trajectoriesIndex = dataVectIndex
    data.trajectories = dataVect
    data.deltaT = deltaT
    data.nData = len(data.trajectoriesIndex)
    data.nTrajectories = len(np.unique(data.trajectoriesIndex))

    return data
