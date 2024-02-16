#Neceassary Imports
import numpy as np
from types import SimpleNamespace
from scipy import stats
import numba as nb

#setseed
np.random.seed(42)

#function that determines temperature
def expCooling(iteration, initialTemp, coolRate):
    return initialTemp*np.exp(-coolRate*iteration)

#function that calculates unnormalized logpdf of Normal distribution asumming n=1
@nb.njit(cache=True)
def logNormpdf(diff, sigma):
    return -np.log(np.abs(sigma))-0.5*(diff/sigma)**2

#function that randomizes index sequence for point sampler
@nb.njit(cache=True)
def indexShuffler(length):
    indices = np.arange(length)
    for i in range(length):
        j = int(np.random.random() * (length - i)) + i
        indices[i], indices[j] = indices[j], indices[i]
    return indices

#create a covariance matrix based on data at hand
@nb.njit(cache=True)
def covMat(coordinates1, coordinates2, covLambda, covL):

    #Create empty matrix for covariance
    C = np.zeros((len(coordinates1), len(coordinates2)))
    
    #loop over all indices in covariance matrix
    for i in range(len(coordinates1)):
        for j in range(len(coordinates2)):
            #Calculate distance between points
            dist = np.sqrt((coordinates1[i,0] - coordinates2[j,0])**2 + (coordinates1[i,1] - coordinates2[j,1])**2)
            #Determine each element of covariance matrix
            C[i, j] = (covLambda**2)*(np.exp(((-0.5)*((dist)**2))/(covL**2)))

    #Return Covariance Matrix
    return C

#initialize sampler parameters
def initialization(variables, data):

    #declare variables as instance of SimpleNamespace
    variables = SimpleNamespace(**variables)

    #pull necassary variables
    trajectories = data.trajectories
    nData = data.nData
    trajectoriesIndex = data.trajectoriesIndex
    nInduX = variables.nInduX
    nInduY = variables.nInduY
    nFineX = variables.nFineX
    nFineY = variables.nFineY
    epsilon = variables.epsilon
    deltaT = data.deltaT
    dataX = trajectories[:,0]
    dataY = trajectories[:,1]
    minX = min(dataX)
    minY = min(dataY)
    maxX = max(dataX)
    maxY = max(dataY)
    covL = variables.covL
    covLambda = variables.covLambda

    #Points of trajectory where learning is possible
    dataCoordinates = np.empty((0,2))
    for i in range(nData-1):
        if (trajectoriesIndex[i] == trajectoriesIndex[i+1]):
            dataCoordinates = np.vstack((dataCoordinates, trajectories[i]))

    #Points of trajectory that are "sampled"
    sampleCoordinates = np.empty((0,2))
    for i in range(1,nData):
        if (trajectoriesIndex[i] == trajectoriesIndex[i-1]):
            sampleCoordinates = np.vstack((sampleCoordinates, trajectories[i]))
    
    #find MLE assuming flat map
    diff = sampleCoordinates - dataCoordinates
    num = np.sum(diff * diff)
    den = 4*deltaT*len(diff)
    mle = num/den

    #Estimate Hyperparameters if not chosen by user
    if covL == None:
        covL = np.max([maxX-minX, maxY-minY])*0.2
    if covLambda == None:
        covLambda = mle * 0.1

    #define coordinates for Inducing points
    xIndu = np.linspace(minX, maxX, nInduX)
    yIndu = np.linspace(minY, maxY, nInduY)
    xTemp, yTemp = np.meshgrid(xIndu, yIndu)
    X = np.reshape(xTemp, -1)
    Y = np.reshape(yTemp, -1)
    induCoordinates = np.vstack((X, Y)).T
    nIndu = len(induCoordinates)
    
    #define coordinates for Fine points
    x = np.linspace(minX, maxX, nFineX)
    y = np.linspace(minY, maxY, nFineY)
    xTemp, yTemp = np.meshgrid(x, y)
    X = np.reshape(xTemp, -1)
    Y = np.reshape(yTemp, -1)
    fineCoordinates = np.vstack((X, Y)).T

    #set up initial sample and mean of prior
    dIndu = mle * np.ones(nIndu)
    priorMean = dIndu.copy()
    
    #determine Covarince matrices
    cInduIndu = covMat(induCoordinates, induCoordinates, covLambda, covL)
    cInduData = covMat(induCoordinates, dataCoordinates, covLambda, covL)
    cInduFine = covMat(induCoordinates, fineCoordinates, covLambda, covL)
    cInduInduInv = np.linalg.inv(cInduIndu + epsilon*np.mean(cInduIndu)*np.eye(nIndu))
    cDataIndu = cInduData.T @ cInduInduInv
    cInduInduChol = np.linalg.cholesky(cInduIndu + epsilon*np.mean(cInduIndu)*np.eye(nIndu))

    #determine the mle at each individual data point
    diff = sampleCoordinates - dataCoordinates
    dMleData = np.sum(diff*diff, axis = 1)/(4*deltaT)

    @nb.jit(cache = True, nopython = True)
    def compute_vector(induCoordinates, dataCoordinates, covLambda, covL, dMleData, dIndu, nIndu, cDataIndu, smoother=1):

        #set up matrrix for interpolation
        cInduDataInterpolate = covMat(induCoordinates, dataCoordinates, covLambda, covL/smoother)

        #find smoothing
        for i in range(nIndu):
            dIndu[i] = np.sum(dMleData*cInduDataInterpolate[i]) / np.sum(cInduDataInterpolate[i])
        
        dData = cDataIndu @ dIndu

        return dIndu, dData

    def calculate_log_posterior(dataCoordinates, sampleCoordinates, dData, deltaT, dIndu, priorMean, cInduInduInv):
        # Likelihood of the data
        lhood = np.sum(
            stats.norm.logpdf(
                sampleCoordinates,
                loc=dataCoordinates,
                scale=np.sqrt(2*np.vstack((dData, dData)).T * deltaT)
            )
        )

        # Prior of the Data ignoring normalization
        diff = dIndu - priorMean
        prior = -0.5 * (diff.T @ (cInduInduInv @ diff))
        pTemp = lhood + prior

        return pTemp

    P = -np.inf     # Set to negative infinity initially or any other large negative value to ensure the loop runs at least once

    #compute the initialization based on the value found for smoother
    dIndu, dData = compute_vector(induCoordinates, dataCoordinates, covLambda, covL, dMleData, dIndu, nIndu, cDataIndu)
    P = calculate_log_posterior(dataCoordinates, sampleCoordinates, dData, deltaT, dIndu, priorMean, cInduInduInv)

    #make sure interpolated estimate is positive
    if np.any(dData < 0):
        print("Increase the length scale, the # of inducing points, or set the initial Sample to a flat plane")
        exit()

    print(f"The initial probability is {P} and there are {nIndu} inducing points.")

    #save all variable parameters
    variables.nIndu = nIndu
    variables.sampleCoordinates = sampleCoordinates
    variables.dataCoordinates = dataCoordinates
    variables.induCoordinates = induCoordinates
    variables.fineCoordinates = fineCoordinates
    variables.cInduIndu = cInduIndu
    variables.cInduData = cInduData
    variables.cInduFine = cInduFine
    variables.cInduInduChol = cInduInduChol
    variables.cInduInduInv = cInduInduInv
    variables.dIndu = dIndu
    variables.P = P
    variables.mle = mle
    variables.priorMean = priorMean
    variables.cDataIndu = cDataIndu
    variables.dData = dData
    variables.covLambda = covLambda
    variables.covL = covL

    return variables


#jitted sampler that proposes surface perturbations in inverse space
@nb.jit(nopython=True, cache = True)
def diffusionPointSampler_nb(nIndu, cInduIndu, cInduData, cInduInduInv, cDataIndu, deltaT, means, samples, data, chol, dInduOld, pOld, dDataOld, priorMean, covLambda, epsilon):

    pVectTemp = np.zeros((nIndu))
    dVectTemp = np.zeros((nIndu, nIndu))

    # Calculate probabilities of induced samples
    def probability(dIndu_, dData_):
        
        # Prior ignoring normalization
        diff = dIndu_ - priorMean
        prior = -0.5*(diff.T @ (cInduInduInv @ diff))
        
        #Likelihood of that data
        lhood = 0
        for i in range(samples.shape[0]):
            for j in range(samples.shape[1]):
                lhood += (
                    -.5 * (samples[i, j] - means[i, j])**2 / (2*dData_[i]*deltaT)
                    -.5 * np.log(2*np.pi*2*dData_[i]*deltaT)
                )
        prob = lhood + prior

        return prob
    
    #initialize inverse space
    alphaVect = cInduInduInv @ dInduOld

    #Counter for acceptances and iterations
    accCounter = 0
    iter = 0

    #shuffle the index to sample through alpha vect randomly
    shuffledIndex = indexShuffler(nIndu)

    #Propose new dIndu by sampling random points in inverse space
    for pointIndex in shuffledIndex:

        #Propose new alpha point
        oldAlphaPoint = alphaVect[pointIndex]
        a = np.random.exponential(epsilon)
        alphaDiff = a * oldAlphaPoint * np.random.randn()
        newAlphaPoint = oldAlphaPoint + alphaDiff
        
        #Incorporate new point into dInduNew and dDataNew
        dInduNew = dInduOld + cInduIndu[:, pointIndex] * alphaDiff
        dDataNew = dDataOld + cInduData[pointIndex, :] * alphaDiff

        # Make sure sampled diffusion vallues are all positive
        if (np.all(dDataNew > 0) and np.all(dInduNew > 0)):
                
            #Probability of old and new function
            pOld = pOld
            pNew = probability(dInduNew, dDataNew)

            #Compute acceptance ratio
            acc_prob = (
                pNew - pOld
                +logNormpdf(diff = alphaDiff, sigma = a * newAlphaPoint)
                -logNormpdf(diff = alphaDiff, sigma = a * oldAlphaPoint)
                )
            
            #Accept or Reject
            if acc_prob > np.log(np.random.rand()):
                accCounter += 1
                dInduOld = dInduNew
                dVectTemp[iter] = dInduNew
                pVectTemp[iter] = pNew
                dDataOld = dDataNew
                pOld = pNew
            else:
                dVectTemp[iter] = dInduOld
                pVectTemp[iter] = pOld
        else:
            dVectTemp[iter] = dInduOld
            pVectTemp[iter] = pOld
        iter += 1
    return dInduOld, pOld, dDataOld, dVectTemp, pVectTemp


@nb.jit(nopython=True, cache=True)
def diffusionPointSamplerSA_nb(nIndu, cInduIndu, cInduData, cInduInduInv, cDataIndu, deltaT, means, samples, data, chol, dInduOld, pOld, dDataOld, priorMean, covLambda, epsilon, temperature):

    # Calculate probabilities of induced samples
    def probability(dIndu_, dData_):
        # Prior ignoring normalization
        diff = dIndu_ - priorMean
        prior = -0.5 * (diff.T @ (cInduInduInv @ diff))

        # Likelihood of the data
        lhood = 0
        for i in range(samples.shape[0]):
            for j in range(samples.shape[1]):
                lhood += (
                    -0.5 * (samples[i, j] - means[i, j]) ** 2 / (2 * dData_[i] * deltaT)
                    -0.5 * np.log(2 * np.pi * 2 * dData_[i] * deltaT)
                )
        prob = lhood + prior

        return prob

    # Initialize inverse space
    alphaVect = cInduInduInv @ dInduOld

    # Counter for acceptances
    accCounter = np.zeros(np.shape(alphaVect))

    # Shuffle the index to sample through alpha vect randomly
    shuffledIndex = indexShuffler(nIndu)

    # Propose new dIndu by sampling random points in inverse space
    for pointIndex in shuffledIndex:

        # Propose new alpha point
        oldAlphaPoint = alphaVect[pointIndex]
        a = np.random.exponential(epsilon[pointIndex])
        alphaDiff = a * oldAlphaPoint * np.random.randn()
        newAlphaPoint = oldAlphaPoint + alphaDiff

        # Incorporate new point into dInduNew and dDataNew
        dInduNew = dInduOld + cInduIndu[:, pointIndex] * alphaDiff
        dDataNew = dDataOld + cInduData[pointIndex, :] * alphaDiff

        # Make sure sampled diffusion values are all positive
        if np.all(dDataNew > 0) and np.all(dInduNew > 0):

            # Probability of old and new function
            pOld = pOld
            pNew = probability(dInduNew, dDataNew)

            # Compute acceptance probability based on energy difference
            energy_diff = pNew - pOld

            # Apply simulated annealing by incorporating temperature
            if energy_diff/temperature > np.log(np.random.rand()):
                accCounter[pointIndex] = 1
                dInduOld = dInduNew
                dDataOld = dDataNew
                pOld = pNew
    return dInduOld, pOld, dDataOld, accCounter


#This function is a Metropolis sampler that samples from inverse gaussian process space 
def diffusionPointSampler(variables, data):
    
    #necassary variables
    cInduIndu = variables.cInduIndu
    cInduData = variables.cInduData
    cInduInduInv = variables.cInduInduInv
    deltaT = data.deltaT
    means = variables.dataCoordinates
    samples = variables.sampleCoordinates
    data = data.trajectories
    chol = variables.cInduInduChol
    dInduOld = variables.dIndu
    cDataIndu = variables.cDataIndu
    P = variables.P
    dData = variables.dData
    priorMean = variables.priorMean
    covLambda = variables.covLambda
    epsilon = variables.epsilon
    nIndu = variables.nIndu
    temperature = variables.temperature
    
    # Run numba version
    #dIndu, P , dData, dVect, pVect = diffusionPointSampler_nb(nIndu, cInduIndu, cInduData, cInduInduInv, cDataIndu, deltaT, means, samples, data, chol, dInduOld, P, dData, priorMean, covLambda, epsilon)
    dIndu, P , dData, accCount = diffusionPointSamplerSA_nb(nIndu, cInduIndu, cInduData, cInduInduInv, cDataIndu, deltaT, means, samples, data, chol, dInduOld, P, dData, priorMean, covLambda, epsilon, temperature)
    variables.dIndu = dIndu
    variables.P = P
    variables.dData = dData

    accRate = 100*np.sum(accCount)/nIndu

    print(f"{accRate:.2f}%", end=" ")

    return variables, accCount