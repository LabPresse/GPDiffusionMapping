import numpy as np
from types import SimpleNamespace
import functions
import copy
import time
import pickle

#These are the necassary variables to run the Gibbs Sampler
PARAMETERS = {

    #General Knowns
    'nInduX': 30,   #number of x inducing points
    'nInduY': 30,   #number of y grid points (total inducing points is nInduY*nInduX)
    'nFineX': 100,  #fine grid points in x
    'nFineY': 100,  #fine grid points in y (total fine points is nFineY*nFineX)
    'nIndu': 0,     # of inducingpoints after trimming

    #Knowns to be evaluated
    'dataCoordinates': None,    #all points of each trajectory exepts final location
    'sampleCoordinates': None,  #all points of each trajectory exepts initial location
    'induCoordinates': None,    #coordinates of the inducing points
    'fineCoordinates': None,    #coordinates of fine grid points
    'cInduIndu': None,          #covariance matrix between inducing points
    'cInduData': None,          #covariance matrix between inducing points and data points
    'cInduFine': None,          #covariance matrix between inducing points and fine grid points
    'cInduInduChol': None,      #cholesky decomposition of cInduIndu
    'cInduInduInv': None,       #inverse of cInduIndu
    'cDataIndu': None,          #product of cInduData*cInduInduInv

    # Variables
    'P': float,     #probability of each sample
    'dIndu': None,  #diffusion coefficient sample map at inducing points
    'dData': None,  #interpolated diffusion from induCoordinates to dataCoordinates

    # Priors
    'covLambda': None,      #coefficient of covariance square exponential kernal (1 only used if hyper parameters on specified)
    'covL': None,           #lenghts parameter of covariance square exponential kernal (20 only used if hyper parameters on specified)
    'mle': None,            #Prior on Inducing point MAP (set to MLE in init)
    'priorMean': None,      #Flat surface at MLE to be used as mean of GPP

    # Sampler parameters
    'epsilon': 1e-2,        #perturbation parameter to keep matrix decomp numerically stable and sample magnitude
    'temperature': 10,      #temperature for simulated annealing will decay to one over time
}

# This function runs the Gibbs Sampler
def analyze(data, nIter=1000, **kwargs):
    
    # Progress Marker
    print('Inititalization Started')
    startTime = time.time()
    
    # Initialize data and variables and time them
    parameters = {**copy.deepcopy(PARAMETERS), **kwargs}
    variables = functions.initialization(parameters, data)
    endTime = time.time()

    # #save variables and data dictionaries to pickle files for easy access when plotting
    # file = open(str(nIter) + " " + str(variables.covLambda) + " " + str(variables.covL) + "variables.pkl","wb")
    # pickle.dump(variables, file) 
    # file.close()
    # file = open(str(nIter) + " " + str(variables.covLambda) + " " + str(variables.covL) + "data.pkl","wb")
    # pickle.dump(data, file) 
    # file.close()

    #Progress Marker
    print("Initialization Sucessful: " + str(endTime - startTime))
    print("The flat MLE is: " + str(variables.mle))

    # Get number of iterations
    burnIter = 2*nIter
    stabalizeIter = 200

    #vectors to store diffusion samples and their probabilities
    # h5 = h5py.File('Results.h5', 'w')
    # h5.create_dataset(name='P', shape=(numTot,1), chunks=(1,1), dtype='f')
    # h5.create_dataset(name='d', shape=(numTot,variables.nIndu), chunks=(1,variables.nIndu), dtype='f')
    # totIter = 0

    # Redefine perturbation magnitude for samples
    variables.epsilon = np.ones(np.shape(variables.dIndu))
    startTime = time.time()

    # Initial temp and cooling rate for targeted annealing
    initTemp = 7.5
    coolRate = np.log(initTemp)/burnIter

    ## BURN IN 1 ##
    # Burn in super aggressively but do not save the samples, 
    # updating proposal size to maintain healthy acceptance rate
    accVect = np.zeros(np.shape(variables.dIndu))
    for i in range(burnIter):
        variables.temperature = functions.expCooling(i, initTemp, coolRate)
        print(f"Iteration {i+1}/{burnIter}", end=" ")
        t = time.time()
        variables, accCount = functions.diffusionPointSampler(variables, data)

        #update proposal size for next iteration
        accVect += accCount
        accVectNorm = 100*accVect/(i+1)
        variables.epsilon = np.where(
            accVectNorm > 40, variables.epsilon * 1.5, 
            np.where(accVectNorm < 20, variables.epsilon * 0.5, variables.epsilon)
        )

        print(f"({time.time()-t:.3f}s)")

    ## Stabalize proposal Magnitude ##
    # Set temperature back to 1 and acceptance to 0, and iterate 
    # a few times to find ideal magnitude of proposal to maintain healthy acceptance
    variables.temperature = 1
    accVect = np.zeros(np.shape(variables.dIndu))
    for i in range(stabalizeIter):
        print(f"Iteration {i+1}/{stabalizeIter}", end=" ")
        t = time.time()
        variables, accCount = functions.diffusionPointSampler(variables, data)

        #update proposal size for next iteration
        accVect += accCount
        if i>10:
            accVectNorm = 100*accVect/(i+1)
            variables.epsilon = np.where(accVectNorm > 27.5, variables.epsilon * 1.25, np.where(accVectNorm < 22.5, variables.epsilon * 0.75, variables.epsilon))

        print(f"({time.time()-t:.3f}s)")

    ## SAMPLING ##
        
    # Initialize outputs
    dVect = np.zeros((nIter, variables.nIndu))
    pVect = np.zeros(nIter)
    map_variables = copy.deepcopy(variables)
    map_variables.P = -np.inf

    # Gibbs Sampler
    for i in range(nIter):
        print(f"Iteration {i+1}/{nIter}", end=" ")
        t = time.time()

        # Sample diffusion map
        variables, accCount= functions.diffusionPointSampler(variables, data)
        accVect += accCount

        # Save the sample
        dVect[i] = variables.dIndu
        pVect[i] = variables.P
        if variables.P >= map_variables.P:
            map_variables = copy.deepcopy(variables)

        print(f"({time.time()-t:.3f}s)")
        
    endTime = time.time()
        
    print(str(nIter) + " iterations in " + str(endTime-startTime) + " seconds." )

    # #Save Samples as h5 files and time
    # h5.close()

    return map_variables, dVect, pVect
