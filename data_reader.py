import numpy as np
import copy
from types import SimpleNamespace

# CAPITALIZED COMMENTS FOR CLARITY

# THIS IS THE OBJECT OF DATA
DATA = {
    # NEED TO BE LOADED IN
    'trajectories': None,       # Coordinates of the data
    'trajectoriesIndex': None,  # Index of trajectory number
    'deltaT': None,             # Time passed in between each frame of data

    # INITIALIZED INDEPENDENTLY BASED ON PARAMETERS ABOVE
    'nData': None,              # Number of data points (set to len(nData))
    'nTrajectories': None,      # Number of trajectories (set to len(np.unique(data.trajectoriesIndex)))
}

def data_reader(path, scale=1, deltaT = 1/30):
    """
    Read data from a CSV file and prepare it for further processing.

    Args:
        path (str): Path to the CSV file.
        scale (float): Scaling factor to transform data to nm. Defaults to 1.
        deltaT (float): Time interval between each frame of data. Defaults to 1/30.

    Returns:
        types.SimpleNamespace: An object containing trajectory data and related information.
    """

    # Read the CSV file, considering the header
    data = np.genfromtxt(path, delimiter=', ', skip_header=1)

    # Separate columns into individual arrays
    dataVectIndex = data[:, 0]
    dataVect = data[:, 1:]
    
    # Localization adjustment to nanometers
    dataVect = dataVect[::]*scale
    dataVectIndex = dataVectIndex[::]
    
    # Put time step manually as unavailable from data file
    deltaT = deltaT

    # Return dataVect, dataVectIndex, deltaT
    data = SimpleNamespace(**copy.deepcopy(DATA))
    data.trajectoriesIndex = dataVectIndex
    data.trajectories = dataVect
    data.deltaT = deltaT
    data.nData = len(data.trajectoriesIndex)
    data.nTrajectories = len(np.unique(data.trajectoriesIndex))

    return data