import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib

def plot(variables, dVect, pVect, data):
    """
    Plot the maximum a posteriori (MAP) estimate of the diffusion coefficient 
    as a 3D surface plot, along with particle trajectory data.

    Args:
        variables (SimpleNamespace): Object containing necessary variables.
        dVect (numpy.ndarray): Array of diffusion coefficient samples.
        pVect (numpy.ndarray): Array of probabilities for each sample.
        data (SimpleNamespace): Object containing trajectory data.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    
    # Necessary variables
    nFineX = variables.nFineX
    nFineY = variables.nFineY
    cInduFine = variables.cInduFine
    cInduInduInv = variables.cInduInduInv
    fineCoordinates = variables.fineCoordinates
    trajectories = data.trajectories

    # Calculate MAP estimate
    index_of_max = np.argmax(pVect)
    dMAP = dVect[index_of_max]
    unshapedMap = (cInduFine.T @ (cInduInduInv @ dMAP))

    # Reshape variables for plotting
    shape = (nFineX, nFineY)
    shapedMap = np.reshape(unshapedMap, shape) / 1e6  # Convert to micron^2/s
    shapedX = np.reshape(fineCoordinates[:, 0], shape)
    shapedY = np.reshape(fineCoordinates[:, 1], shape)

    # Set up plotting parameters
    vmax = np.ceil(np.max(shapedMap) * 100) / 100
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)

    # Generate 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(shapedX / 1000, shapedY / 1000, shapedMap, cmap=cm.coolwarm, norm=norm)
    ax.scatter(trajectories[:, 0] / 1000, trajectories[:, 1] / 1000, np.zeros_like(trajectories[:, 0]),
               color='green', alpha=0.01, label='Particle Data', s=1)
    ax.set_xlabel(r"X ($\mu m$)")
    ax.set_ylabel(r"Y ($\mu m$)")
    ax.set_zlabel(r"Diffusion Coeff. ($\mu m^2/s$)")
    ax.set_title("Inferred Diffusion Map")

    # Create colorbar
    m = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    cbar = fig.colorbar(m, ax=ax, shrink=0.5, anchor=(0.5, 0.5))
    cbar_ticks = np.linspace(0, vmax, 3)
    cbar.set_ticks(ticks=cbar_ticks)

    # Set the same ticks for the z-axis
    ax.set_zticks(cbar_ticks)

    return fig
