import numpy as np
import cluster
# import ipdb
# ipdb.set_trace()


data = np.loadtxt('Iris.csv',delimiter=',')
data = data[:,:4]

sigma=0.5
repetitions=50
stepSize=0.15
clusteringType='s'
recalculate=False
returnHistory=True
stopCondition=True

x,xHistory = cluster.GradientDescent(data,sigma=sigma,repetitions=repetitions,stepSize=stepSize,clusteringType=clusteringType,recalculate=recalculate,returnHistory=returnHistory,stopCondition=stopCondition)

clusters = cluster.PerformFinalClustering(x,stepSize)