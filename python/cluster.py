import numpy as np
from scipy.spatial import  distance

def VGradient(data,sigma,x=None,coeffs=None):
    if x is None:
        x = data.copy()
    
    if coeffs is None:
        coeffs = np.ones((data.shape[0],))
        
    twoSigmaSquared = 2*sigma**2
        
    data = data[np.newaxis,:,:]
    x = x[:,np.newaxis,:]
    differences = x-data
    squaredDifferences = np.sum(np.square(differences),axis=2)
    gaussian = np.exp(-(1/twoSigmaSquared)*squaredDifferences)
    laplacian = np.sum(coeffs*gaussian*squaredDifferences,axis=1)
    parzen = np.sum(coeffs*gaussian,axis=1)
    v = 1 + (1/twoSigmaSquared)*laplacian/parzen

    dv = -1*(1/parzen[:,np.newaxis])*np.sum(differences*((coeffs*gaussian)[:,:,np.newaxis])*(twoSigmaSquared*(v[:,np.newaxis,np.newaxis])-(squaredDifferences[:,:,np.newaxis])),axis=1)
    
    v = v-1
    
    return v, dv

def SGradient(data, sigma, x=None,coeffs=None):
    if x is None:
        x = data.copy()
        
    if coeffs is None:
        coeffs = np.ones((data.shape[0],))
    
    twoSigmaSquared = 2 * sigma ** 2
    
    data = data[np.newaxis, :, :]
    x = x[:, np.newaxis, :]
    differences = x - data
    squaredDifferences = np.sum(np.square(differences), axis=2)
    gaussian = np.exp(-(1 / twoSigmaSquared) * squaredDifferences)
    laplacian = np.sum(coeffs*gaussian * squaredDifferences, axis=1)
    parzen = np.sum(coeffs*gaussian, axis=1)
    v = (1 / twoSigmaSquared) * laplacian / parzen
    s = v + np.log(np.abs(parzen))
    
    ds = (1 / parzen[:, np.newaxis]) * np.sum(differences * ((coeffs*gaussian)[:, :, np.newaxis]) * (
    twoSigmaSquared * (v[:, np.newaxis, np.newaxis]) - (squaredDifferences[:, :, np.newaxis])), axis=1)
    
    return s, ds

def PGradient(data, sigma, x=None,coeffs=None):
    if x is None:
        x = data.copy()
        
    if coeffs is None:
        coeffs = np.ones((data.shape[0],))
    
    twoSigmaSquared = 2 * sigma ** 2
    
    data = data[np.newaxis, :, :]
    x = x[:, np.newaxis, :]
    differences = x - data
    squaredDifferences = np.sum(np.square(differences), axis=2)
    gaussian = np.exp(-(1 / twoSigmaSquared) * squaredDifferences)
    p = np.sum(coeffs*gaussian,axis=1)
    
    dp = -1*np.sum(differences * ((coeffs*gaussian)[:, :, np.newaxis]) * twoSigmaSquared,axis=1)
    
    return p, dp

def getApproximateParzen(data,sigma,voxelSize):
    newData = uniqueRows(np.floor(data/voxelSize)*voxelSize+voxelSize/2)[0]
    
    nMat = np.exp(-1*distance.squareform(np.square(distance.pdist(newData)))/(4*sigma**2))
    mMat = np.exp(-1 * np.square(distance.cdist(newData,data)) / (4 * sigma ** 2))
    cMat = np.linalg.solve(nMat,mMat)
    coeffs = np.sum(cMat,axis=1)
    coeffs = data.shape[0]*coeffs/sum(coeffs)
    
    return newData,coeffs

def uniqueRows(x):
    y = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, inds,indsInverse,counts = np.unique(y, return_index=True,return_inverse=True,return_counts=True)

    xUnique = x[inds]
    return xUnique,inds,indsInverse,counts

def GradientDescent(data,sigma,repetitions=1,stepSize=None,clusteringType='v',recalculate=False,returnHistory=False,stopCondition=True,voxelSize=None):
    
    n = data.shape[0]

    useApproximation = ~(voxelSize is None)
    
    if stepSize is None:
        stepSize = sigma/10
    
    if clusteringType == 'v':
        gradientFunction = VGradient
    elif clusteringType == 's':
        gradientFunction = SGradient
    else:
        gradientFunction = PGradient

    if useApproximation:
        newData, coeffs = getApproximateParzen(data, sigma, voxelSize)
    else:
        coeffs = None

    if recalculate:
        if useApproximation:
            x = np.vstack((data,newData))
            data = x[data.shape[0]:]
        else:
            x = data
    else:
        if useApproximation:
            x = data
            data = newData
        else:
            x = data.copy()
        
        
    if returnHistory:
        xHistory = np.zeros((n,x.shape[1],repetitions+1))
        xHistory[:,:,0] = x[:n,:].copy()
        
    if stopCondition:
        prevX = x[:n].copy()

    for i in range(repetitions):
        if ((i>0) and (i%10==0)):
            if stopCondition:
                if np.all(np.linalg.norm(x[:n]-prevX,axis=1) < np.sqrt(3*stepSize**2)):
                    i = i-1
                    break
                prevX = x[:n].copy()
            
        f,df = gradientFunction(data,sigma,x,coeffs)
        df = df/np.linalg.norm(df,axis=1)[:,np.newaxis]
        x[:] = x + stepSize*df

        if returnHistory:
            xHistory[:, :, i+1] = x[:n].copy()
            
    x = x[:n]

    if returnHistory:
        xHistory = xHistory[:,:,:(i+2)]
        return x,xHistory
    else:
        return x

def GradientDescentApprox(data,sigma,repetitions=1,stepSize=None,clusteringType='v',recalculate=False,returnHistory=False,stopCondition=True,voxelSize=1):
    newData, coeffs = getApproximateParzen(data,sigma,voxelSize)
    
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