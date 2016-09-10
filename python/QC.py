import numpy as np
import cluster

data = np.loadtxt('Iris.csv',delimiter=',')
data = data[:,:4]

sigma=0.55
repetitions=100
stepSize=0.1
clusteringType='v'
recalculate=False
returnHistory=True
stopCondition=True
voxelSize = None

x,xHistory = cluster.GradientDescent(data,sigma=sigma,repetitions=repetitions,stepSize=stepSize,clusteringType=clusteringType,recalculate=recalculate,returnHistory=returnHistory,stopCondition=stopCondition,voxelSize=voxelSize)

clusters = cluster.PerformFinalClustering(x,stepSize)

cluster.displayClustering(xHistory,clusters)