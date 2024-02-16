#imports
import model
import data_reader
import numpy as np
import os
import plotting

# load real data from csv File
dataPath = "data/simulated_flat_Dmap.csv"
data = data_reader.data_reader(dataPath)

#generate samples
variables, dVect, pVect = model.analyze(data)

# Plot Results
plotting.plot_surface(variables, dVect, data)