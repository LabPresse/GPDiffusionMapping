
# Imports
import os
import numpy as np
import matplotlib.pyplot as plt

# Local imports
import data_reader
import model
import plotting

# Load data from csv File
dataPath = "data/synthetic/simulated_flat_Dmap.csv" # REQUIRED: Change path to data file you would like to analyze
data = data_reader.data_reader(dataPath)

# Generate samples
variables, dVect, pVect = model.analyze(data) # OPTIONAL: Edit any hyperparameters as keyword arguements here

# Plot Results
plotting.plot(variables, dVect, pVect, data)
plt.savefig('output/InferredMAP.png')