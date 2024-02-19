import csv
import numpy as np
import matplotlib.pyplot as plt
import functions
from types import SimpleNamespace
from matplotlib import cm
import ast
import pickle
import matplotlib

def plot_contour_map(variables, dVect, data):
    """This function plots the mean of all dVect samples as a contour map."""

    #necassary variables
    nFineX = variables.nFineX
    nFineY = variables.nFineY
    cInduFine = variables.cInduFine
    cInduInduInv = variables.cInduInduInv
    fineCoordinates = variables.fineCoordinates
    trajectories = data.trajectories

    #shape for plot
    shape = (nFineX, nFineY)

    #take mean of all samples
    unshapedMap = cInduFine.T @ (cInduInduInv @ np.mean(dVect, 0))
    
    #reshape variables to make plotting easy
    shapedMap = np.reshape(unshapedMap, shape)
    shapedX = np.reshape(fineCoordinates[:,0], shape)
    shapedY = np.reshape(fineCoordinates[:,1], shape)

    #generate contour plot
    fig = plt.figure()
    mapPlot = plt.contour(shapedX, shapedY, shapedMap, levels = 25, cmap = cm.autumn)
    plt.clabel(mapPlot, inline=1, fontsize=10)
    plt.scatter(trajectories[:,0], trajectories[:,1], alpha = 0.01, c = "black")

    return fig

def plot_surface(variables, dVect, data):
    """This function plots the mean of all dVect samples as a surface plot."""

    #necassary variables
    nFineX = variables.nFineX
    nFineY = variables.nFineY
    cInduFine = variables.cInduFine
    cInduInduInv = variables.cInduInduInv
    fineCoordinates = variables.fineCoordinates
    trajectories = data.trajectories

    #shape for plot
    shape = (nFineX, nFineY)

    #take mean of all samples
    unshapedMap = cInduFine.T @ (cInduInduInv @ np.mean(dVect, 0))
    
    #reshape variables to make plotting easy
    shapedMap = np.reshape(unshapedMap, shape)
    shapedX = np.reshape(fineCoordinates[:,0], shape)
    shapedY = np.reshape(fineCoordinates[:,1], shape)

    # Initialize figure
    fig = plt.figure()
    plt.ion()
    plt.show()
    ax = fig.add_subplot(111, projection='3d')

    # Generate contour plot
    ax = plt.axes(projection='3d')
    ax.plot_surface(shapedX, shapedY, shapedMap, cmap=cm.coolwarm)
    ax.scatter3D(trajectories[:,0], trajectories[:,1], 0, color = "green", alpha = 0.1, label = "Particle Data")
    
    # Finalize figure
    ax.set_xlabel(r"X ($\mu m$)")
    ax.set_ylabel(r"Y ($\mu m$)")
    ax.set_zlabel(r"Diff. Coefficient ($\mu m$)")
    ax.set_title('Learned Diffusion Map')
    ax.legend()
    plt.tight_layout()

    # Return figure
    return fig, ax