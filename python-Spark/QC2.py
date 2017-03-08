import numpy as np
from pyspark import SparkContext,SparkConf
from pyspark import mllib
from scipy.optimize import minimize
import shutil

data_source = 'Iris.csv' # path must be accessible from all nodes, so either copy file to all nodes or make sure path is on shared filesystem
out_file = 'clusters.pickle'
number_of_header_lines = 0
delimiter = ','
sigma = 0.8
iterations=400
min_gradient=1e-5
voxel_size_for_final_clustering = sigma / 10
voxel_size_for_approximation = sigma / 7

minimal_number_of_partitions = None

def Potential(x,data,weights,sigma):
    q = 1 / (2 * sigma ** 2)
    differences = x-data
    squared_distances = np.linalg.norm(differences,axis=1,keepdims=1)**2
    gaussians = weights*np.exp(-q*squared_distances)
    parzen = np.sum(gaussians)
    probabilities = gaussians/parzen
    potential = q*np.sum(probabilities*squared_distances)
    gradient = 2*q*np.sum(differences*probabilities*(1+potential-q*squared_distances),axis=0)

    return potential,gradient

def MinimizePotential(x,data,weights,sigma,iterations=400,min_gradient=0.1):
    x = minimize(Potential,x,args=(data,weights,sigma),method='BFGS',jac=True,options={'max_iter':iterations,'gtol':min_gradient}).x
    return x

conf = SparkConf()

sc = SparkContext()
text_data = sc.textFile(data_source,minPartitions=minimal_number_of_partitions).\
    zipWithIndex().\
    filter(lambda x: x[1]>=number_of_header_lines)

data = text_data.map(lambda x: (x[1]-number_of_header_lines, np.fromstring(x[0],sep=delimiter))).persist()
data_in_voxels = data.map(lambda x: tuple(np.round(x[1]/voxel_size_for_approximation)*voxel_size_for_approximation)).countByValue()
data_points = sc.broadcast(np.array(list(data_in_voxels.keys())))
weights = np.array(list(data_in_voxels.values()))
weights = weights/np.sum(weights)
weights = sc.broadcast(weights)

replicas = data

data.unpersist()

replicas_final_locations = replicas.mapValues(lambda x: MinimizePotential(x[np.newaxis,:],data_points.value,weights.value[:,np.newaxis],sigma,iterations,min_gradient))
replicas_snapped_to_grid = replicas_final_locations.mapValues(lambda x: np.round(x / voxel_size_for_final_clustering) * voxel_size_for_final_clustering)
clusters = replicas_snapped_to_grid.groupBy(lambda x: tuple(x[1])).\
    zipWithUniqueId().\
    map(lambda x: (x[1],[y[0] for y in x[0][1]]))

shutil.rmtree(out_file,ignore_errors=True)
clusters.saveAsPickleFile(out_file)



#
#
#
# inds = np.arange(data.count())
# replicas = sc.parallelize(np.hstack((inds[:, np.newaxis], data))).persist()
# replicas =  replicas.mapValues(lambda x: (x[0],(x[1:],np.inf))) # (ind, (r,v))
# n = len(data_points.value)
#
# i = 0
# while i < max_iterations:
#
#     differences = replicas.mapValues(lambda x: (x[0],x[1],[(x[0]-data_points.value[j],) for j in range(n)])) # (ind, (r,v,[(diff1,),...]))
#     squared_distances = differences.mapValues(lambda x: (x[0],x[1],[x[2][j]+(np.linalg.norm(x[2][j][0]) ** 2,) for j in range(n)])) # (ind, (r,v,[(diff1,d1),...]))
#     gaussians = squared_distances.mapValues(lambda x: (x[0],[x[1][j]+(np.exp((-q)*x[1][j][1]),) for j in range(n)])) # (ind,(r,[(diff1,d1,gauss1),...]))
#     parzens = gaussians.mapValues(lambda x: (x[0],np.sum([x[1][j][2] for j in range(n)]),x[1])) # (ind,(r,parz,[(diff1,d1,gauss1),...]))
#     probabilities = parzens.mapValues(lambda x: (x[0],x[1],[(x[2][j][0],x[2][j][1])+(x[2][j][2]/x[1],) for j in range(n)])) # (ind,(r,parz,[(diff1,d1,p1),...]))
#     potential = probabilities.mapValues(lambda x: (x[0],q*np.sum([x[2][j][1]*x[2][j][3] for j in range(n)]),x[2])) # (ind,(r,v,[(diff1,d1,p1)...]))
#     gradient = potential.mapValues(lambda x: (x[0],x[1],2*q*np.sum([x[2][j][0]*x[2][j][2]*(1+x[1]-q*x[2][j][1]) for j in range(n)]))) # (ind,(r,v,grad))
#     replicas =
#
#     # dv = 2 * q * sum(bsxfun( @ times, differences, (1 + v - q * squared_differences). * probabilities));
#
#
#     pairs = replicas.flatMap(lambda x: [((x[0],j),(x[1:],data_points.value[j])) for j in range(len(data_points.value))]) # ((ind,ind2),(r,x))
#     differences = pairs.mapValues(lambda x: x[0]-x[1])  # ((ind,ind2),diff)
#     squared_distances = differences.mapValues(lambda x: (x, np.linalg.norm(x) ** 2))  # ((ind,ind2),(diff,sqrd_dist))
#     gaussians = squared_distances.mapValues(lambda x: (x[0],x[1],np.exp((-q) * x[2]))).persist() # ((ind,ind2),(differences,sqrd_dist,gaussian)
#     parzens = gaussians.groupBy(lambda x: x[0][0])\
#         .reduceByKey(lambda x, y: x[1][2] + y[1][2])  #(ind,[((ind,ind2),(differences,sqrd_dist,gaussian))]) -> (ind,parzen)
#     probabilities = parzens.join(gaussians.map(lambda x: (x[0][0],(x[0][1],x[1][0],x[1][1],x[1][2]))))\
#         .map(lambda x: ((x[0],x[1][1][0]),(x[1][1][1],x[1][1][2],x[1][1][3],x[1][0],x[1][1][3]/x[1][0]))) # (ind,(parzen,(ind2,difference,dist,gaussian))) -> ((ind,ind2),(differences,dist,gaussian,parzen,p))
#     gaussians.unpersist()
#     potential = probabilities.groupBy(lambda x: x[0][0])\
#         .reduceByKey(lambda x)

    # differences = pairs.mapValues(lambda x: x[0]-x[1]).persist() # ((ind,ind2),diff)
    # squared_distances = differences.mapValues(lambda x: (x,np.linalg.norm(x) ** 2)) # ((ind,ind2),(diff,sqrd_dist))
    # gaussians = squared_distances.mapValues(lambda x: (x[0],x[1],np.exp((-q) * x[2])))# ((ind,ind2),(differences,sqrd_dist,gaussian)
    # parzens = gaussians.groupBy(lambda x: x[0][0])\
    #     .reducebyKey(lambda x,y: x[1]+y[1]) # (ind,parzen)
    # probabilities = parzens.join(gaussians.map(lambda x: (x[0][0],(x[0][1],x[1]))))\ # (ind,(parzen,(ind2,gaussian)))
    #     .map(lambda x: ((x[0],x[1][1][0]),x[1][1][1]/x[1][0]))  # ((ind,ind2),p)
    # gaussians.unpersist()
    # probabilities_times_squared_distances = squared_distances.join(probabilities)\
    #     .mapValues(lambda x,y: q*x*y).persist() # ((ind,ind2),q*p*sqrd_dist)
    # potential = probabilities_times_squared_distances.groupBy(lambda x:x[0][0])\
    #     .reduceByKey(lambda x,y:(x+y)) # (ind,v)
    # differences.join(probabilities).
    # potential.join(squared_distances.map(lambda x: (x[0][0],(x[0][1],x[1]))))\
    #     .mapValues(lambda x: (x[1][0],1+x[0]-q*x[1][1]))\
    #     .map(lambda x: ((x[0],x[1][0]),x[1][1]))\
    #     .join(probabilities)
    # squared_distances.unpersist()
    # probabilities.unpersist()
    # differences.unpersist()




    #
    #
    # v = q * sum(squared_differences. * probabilities);
    #
    # dv = 2 * q * sum(bsxfun( @ times, differences, (1 + v - q * squared_differences). * probabilities));
    #









