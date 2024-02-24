
# Imports
import os
import numpy as np
import matplotlib.pyplot as plt

# Local imports
import data_reader
import model
import plotting

# Load real data from csv File
dataPath = "data/simulated_flat_Dmap.csv"
data = data_reader.data_reader(dataPath)

# Generate samples
variables, dVect, pVect = model.analyze(data, nIter=50)

# Plot Results
plotting.plot(variables, dVect, pVect, data)
plt.savefig('output/InferredMAP.png')