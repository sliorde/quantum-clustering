import numpy as np
from scipy.spatial import  distance

def VGradient(data,sigma,x=None):
    if x is None:
        x = data.copy()
        
    twoSigmaSquared = 2*sigma**2
        
    data = data[np.newaxis,:,:]
    x = x[:,np.newaxis,:]
    differences = x-data
    squaredDifferences = np.sum(np.square(differences),axis=2)
    gaussian = np.exp(-(1/twoSigmaSquared)*squaredDifferences)
    laplacian = np.sum(gaussian*squaredDifferences,axis=1)
    parzen = np.sum(gaussian,axis=1)
    v = 1 + (1/twoSigmaSquared)*laplacian/parzen
    
    
    
    dv = -1*(1/parzen[:,np.newaxis])*np.sum(differences*(gaussian[:,:,np.newaxis])*(twoSigmaSquared*(v[:,np.newaxis,np.newaxis])-(squaredDifferences[:,:,np.newaxis])),axis=1)
    
    v = v-1
    
    return v, dv

def SGradient(data, sigma, x=None):
    if x is None:
        x = data.copy()
    
    twoSigmaSquared = 2 * sigma ** 2
    
    data = data[np.newaxis, :, :]
    x = x[:, np.newaxis, :]
    differences = x - data
    squaredDifferences = np.sum(np.square(differences), axis=2)
    gaussian = np.exp(-(1 / twoSigmaSquared) * squaredDifferences)
    laplacian = np.sum(gaussian * squaredDifferences, axis=1)
    parzen = np.sum(gaussian, axis=1)
    v = (1 / twoSigmaSquared) * laplacian / parzen
    s = v + np.log(np.abs(parzen))
    
    ds = (1 / parzen[:, np.newaxis]) * np.sum(differences * (gaussian[:, :, np.newaxis]) * (
    twoSigmaSquared * (v[:, np.newaxis, np.newaxis]) - (squaredDifferences[:, :, np.newaxis])), axis=1)
    
    return s, ds

def PGradient(data, sigma, x=None):
    if x is None:
        x = data.copy()
    
    twoSigmaSquared = 2 * sigma ** 2
    
    data = data[np.newaxis, :, :]
    x = x[:, np.newaxis, :]
    differences = x - data
    squaredDifferences = np.sum(np.square(differences), axis=2)
    gaussian = np.exp(-(1 / twoSigmaSquared) * squaredDifferences)
    p = np.sum(gaussian,axis=1)
    
    dp = -1*np.sum(differences * (gaussian[:, :, np.newaxis]) * twoSigmaSquared,axis=1)
    
    return p, dp

def GradientDescent(data,sigma,repetitions=1,stepSize=None,clusteringType='v',recalculate=False,returnHistory=False,stopCondition=True):
    if stepSize is None:
        stepSize = sigma/10
    
    if clusteringType == 'v':
        gradientFunction = VGradient
    elif clusteringType == 's':
        gradientFunction = SGradient
    else:
        gradientFunction = PGradient
        
    if recalculate:
        x = data
    else:
        x = data.copy()
        
    if returnHistory:
        xHistory = np.zeros(x.shape+(repetitions+1,))
        xHistory[:,:,0] = x.copy()
        
    if stopCondition:
        prevX = x.copy()

    for i in range(repetitions):
        if ((i>0) and (i%10==0)):
            if stopCondition:
                if np.all(np.linalg.norm(x-prevX,axis=1) < np.sqrt(3*stepSize**2)):
                    i = i-1
                    break
                prevX = x.copy()
            
        f,df = gradientFunction(data,sigma,x)
        df = df/np.linalg.norm(df,axis=1)[:,np.newaxis]
        x[:] = x + stepSize*df

        if returnHistory:
            xHistory[:, :, i+1] = x.copy()

    if returnHistory:
        xHistory = xHistory[:,:,:(i+2)]
        return x,xHistory
    else:
        return x
        
def PerformFinalClustering(data,stepSize):
    clusters = np.zeros((data.shape[0]))
    i = np.array([0])
    c = 0
    distances = distance.squareform(distance.pdist(data))
    while i.shape[0]>0:
        i = i[0]
        inds = np.argwhere(clusters==0)
        clusters[inds[distances[i,inds] <= 3*stepSize]] = c
        c += 1
        i = np.argwhere(clusters==0)
    return clusters
        
        
    