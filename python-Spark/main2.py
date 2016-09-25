import numpy as np
from pyspark import SparkContext
import ipdb

# ipdb.set_trace()


data = np.loadtxt('Iris.csv', delimiter=',')
data = data[:, :4]

sc = SparkContext()

sigma = 0.5
stepSize = 0.02

ind = np.arange(data.shape[0])
initialData = sc.broadcast(data)
sigmaSquared = sc.broadcast(sigma ** 2)
stepSize = sc.broadcast(stepSize)
q = sc.broadcast(data.shape[1])
currentData = []
currentData.append(sc.parallelize(np.hstack((ind[:, np.newaxis], data))))

iterations = 100

for i in range(iterations):
    dataPairs = currentData[-1].flatMap(lambda x: [(x,initialData.value[i]) for i in range(len(initialData.value))])
    differences = dataPairs.map(lambda x: (x[0], x[0][1:] - x[1]))
    squaredDistances = differences.map(lambda x: (x[0], x[1], np.linalg.norm(x[1]) ** 2))
    gaussians = squaredDistances.map(
        lambda x: (x[0], x[1], x[2], np.exp((-1 / (2 * sigmaSquared.value)) * x[2])))
    laplaciansAndParzens = gaussians.map(lambda x: (tuple(x[0]), (x[1], x[2], x[3], x[2] * x[3]))).aggregateByKey(
        (np.zeros((0, q.value)), np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 1))), combFunc=(lambda x, y: (
        np.vstack((x[0], y[0])), np.vstack((x[1], y[1])), np.vstack((x[2], y[2])), np.vstack((x[3], y[3])))), seqFunc=(
        lambda x, y: (
        np.vstack((x[0], y[0])), np.vstack((x[1], y[1])), np.vstack((x[2], y[2])), np.vstack((x[3], y[3]))))).map(
        lambda x: (x[0], (x[1][0], x[1][1], x[1][2], np.sum(x[1][3]), np.sum(x[1][2]))))
    potential = laplaciansAndParzens.map(
        lambda x: (x[0], (x[1][0], x[1][1], x[1][2], x[1][4], 1 + (1 / (2 * sigmaSquared.value)) * x[1][3] / x[1][4])))
    gradient = potential.map(
        lambda x: (x[0], np.sum(x[1][0] * x[1][2] * (2 * sigmaSquared.value * x[1][4] - x[1][1]), axis=0) / x[1][3]))
    
    currentData.append(gradient.map(lambda x: (np.hstack((x[0][0], np.array(x[0][1:]) + stepSize.value * x[1])))))
    currentData[-1].persist()
    currentData[-2].unpersist()