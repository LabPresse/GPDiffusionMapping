import numpy as np
from types import SimpleNamespace
import functions
import copy
import time
import pickle

# NECESSARY VARIABLES TO RUN THE GIBBS SAMPLER
PARAMETERS = {

    # General Knowns
    'nInduX': 30,   # Number of x inducing points
    'nInduY': 30,   # Number of y grid points (total inducing points is nInduY*nInduX)
    'nFineX': 100,  # Fine grid points in x
    'nFineY': 100,  # Fine grid points in y (total fine points is nFineY*nFineX)
    'nIndu': 0,     # Number of inducing points after trimming

    # Knowns to be evaluated
    'dataCoordinates': None,    # All points of each trajectory except final location
    'sampleCoordinates': None,  # All points of each trajectory except initial location
    'induCoordinates': None,    # Coordinates of the inducing points
    'fineCoordinates': None,    # Coordinates of fine grid points
    'cInduIndu': None,          # Covariance matrix between inducing points
    'cInduData': None,          # Covariance matrix between inducing points and data points
    'cInduFine': None,          # Covariance matrix between inducing points and fine grid points
    'cInduInduChol': None,      # Cholesky decomposition of cInduIndu
    'cInduInduInv': None,       # Inverse of cInduIndu
    'cDataIndu': None,          # Product of cInduData*cInduInduInv

    # Variables
    'P': float,     # Probability of each sample
    'dIndu': None,  # Diffusion coefficient sample map at inducing points
    'dData': None,  # Interpolated diffusion from induCoordinates to dataCoordinates

    # Priors
    'covLambda': None,      # Coefficient of covariance square exponential kernel (1 only used if hyper parameters on specified)
    'covL': None,           # Lengths parameter of covariance square exponential kernel (20 only used if hyper parameters on specified)
    'mle': None,            # Prior on Inducing point MAP (set to MLE in init)
    'priorMean': None,      # Flat surface at MLE to be used as mean of GPP

    # Sampler parameters
    'epsilon': 1e-2,        # Perturbation parameter to keep matrix decomposition numerically stable and sample magnitude
    'temperature': 10,      # Temperature for simulated annealing will decay to one over time
}

# THIS FUNCTION RUNS THE GIBBS SAMPLER
def analyze(data, nIter=1000, **kwargs):
    """
    Run the Gibbs sampler.

    Args:
        data (Data): Object containing trajectory data.
        nIter (int, optional): Number of iterations. Defaults to 1000.
        **kwargs: Additional keyword arguments.

    Returns:
        SimpleNamespace: Object containing the results.
        numpy.ndarray: Diffusion coefficient sample map at inducing points.
        numpy.ndarray: Probability of each sample.
    """

    # Progress Marker
    print('Initialization Started')
    startTime = time.time()
    
    # Initialize data and variables and time them
    parameters = {**copy.deepcopy(PARAMETERS), **kwargs}
    variables = functions.initialization(parameters, data)
    endTime = time.time()

    # Progress Marker
    print("Initialization Successful: " + str(endTime - startTime))
    print("The flat MLE is: " + str(variables.mle))

    # Get number of iterations
    burnIter = 2 * nIter
    stabilizeIter = 200

    # Redefine perturbation magnitude for samples
    variables.epsilon = np.ones(np.shape(variables.dIndu))
    startTime = time.time()

    # Initial temp and cooling rate for targeted annealing
    initTemp = 7.5
    coolRate = np.log(initTemp) / burnIter

    ## BURN IN 1 ##
    # Burn in super aggressively but do not save the samples, 
    # updating proposal size to maintain healthy acceptance rate
    accVect = np.zeros(np.shape(variables.dIndu))
    for i in range(burnIter):
        variables.temperature = functions.expCooling(i, initTemp, coolRate)
        print(f"Annealing Iteration {i+1}/{burnIter}", end=" ")
        t = time.time()
        variables, accCount = functions.diffusionPointSampler(variables, data)

        # Update proposal size for next iteration
        accVect += accCount
        accVectNorm = 100 * accVect / (i + 1)
        variables.epsilon = np.where(
            accVectNorm > 40, variables.epsilon * 1.5, 
            np.where(accVectNorm < 20, variables.epsilon * 0.5, variables.epsilon)
        )

        print(f"({time.time()-t:.3f}s)")

    ## Stabilize proposal Magnitude ##
    # Set temperature back to 1 and acceptance to 0, and iterate 
    # a few times to find ideal magnitude of proposal to maintain healthy acceptance
    variables.temperature = 1
    accVect = np.zeros(np.shape(variables.dIndu))
    for i in range(stabilizeIter):
        print(f"Stabilization Iteration {i+1}/{stabilizeIter}", end=" ")
        t = time.time()
        variables, accCount = functions.diffusionPointSampler(variables, data)

        # Update proposal size for next iteration
        accVect += accCount
        if i > 10:
            accVectNorm = 100 * accVect / (i + 1)
            variables.epsilon = np.where(accVectNorm > 27.5, variables.epsilon * 1.25, np.where(accVectNorm < 22.5, variables.epsilon * 0.75, variables.epsilon))

        print(f"({time.time()-t:.3f}s)")

    ## MCMC SAMPLING ##
        
    # Initialize outputs
    dVect = np.zeros((nIter, variables.nIndu))
    pVect = np.zeros(nIter)
    map_variables = copy.deepcopy(variables)
    map_variables.P = -np.inf

    # Gibbs Sampler
    for i in range(nIter):
        print(f"MCMC Iteration {i+1}/{nIter}", end=" ")
        t = time.time()

        # Sample diffusion map
        variables, accCount = functions.diffusionPointSampler(variables, data)
        accVect += accCount

        # Save the sample
        dVect[i] = variables.dIndu
        pVect[i] = variables.P
        if variables.P >= map_variables.P:
            map_variables = copy.deepcopy(variables)

        print(f"({time.time()-t:.3f}s)")
        
    endTime = time.time()
        
    print(str(nIter) + " iterations in " + str(endTime-startTime) + " seconds." )

    return map_variables, dVect, pVect