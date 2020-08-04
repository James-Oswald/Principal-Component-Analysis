import csv
import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

#From Chapter 2

#Univariate Sample Mean
def sampleMean(attrArray):
    n = len(attrArray)
    counter = 0
    for i in range(n):
        counter += attrArray[i]
    return counter / n

#multivariate sample mean
def mvSampleMean(dataMatrix):
    attribs = dataMatrix.transpose()
    rv = np.zeros(len(attribs))
    for i in range(len(attribs)):
        rv[i] = sampleMean(attribs[i])
    return rv


#Centered Data Matrix
def centered(dataMatrix):
    n = dataMatrix.shape[0]
    u = mvSampleMean(dataMatrix)
    #centered data matrix
    z = np.zeros(dataMatrix.shape)
    for i in range(n):
        z[i] = np.subtract(dataMatrix[i], u)
    return z

#Multivariate sample covariance matrix
def mvscvm(dataMatrix):
    n = dataMatrix.shape[0]
    u = mvSampleMean(dataMatrix)
    #centered data matrix
    z = np.zeros(dataMatrix.shape)
    for i in range(n):
        z[i] = np.subtract(dataMatrix[i], u)
    return (1 / n) * np.dot(z.transpose(), z)

#PCA chapter 7
def PCAMax(dataMatrix, dimensionality):
    covMat = mvscvm(dataMatrix)
    eigenVal, eigenVec = la.eig(covMat)
    eigenVec = np.transpose(eigenVec)
    print(sorted(zip(eigenVal, eigenVec), reverse=True))
    eigenVec = np.array([vec for _, vec in sorted(zip(eigenVal, eigenVec), reverse=True)])
    eigenVal = np.flip(np.sort(eigenVal))
    print(eigenVal)
    print(eigenVec)
    n = dataMatrix.shape[0]
    d = dataMatrix.shape[1] #original dimensionality
    f = lambda r: sum(eigenVal[:r]) / sum(eigenVal[:d])
    dimDict = {}
    for r in range(1, d+1):
        f_r = f(r)
        if f_r >= dimensionality:
            dimDict[r] = f_r
    r = min(dimDict.keys())
    reducedBasis = eigenVec[:r]
    A = np.zeros([n, r])
    for i in range(0, n):
        A[i] = np.dot(reducedBasis, dataMatrix[i])
    return A

#Principal component analysis 
def PCA(D, alpha):
    n, d = D.shape      #Shape of the original data matrix
    sigma = mvscvm(D)   #Covariance Matrix
    Lambda, U = (lambda eVal, eVec: (np.flip(np.sort(eVal)), np.array([vec for (_, vec) in sorted(zip(eVal, np.transpose(eVec)), reverse=True)])))(*la.eig(sigma)) #EigenValues, Eigen Vectors
    f = lambda r: sum(Lambda[:r]) / sum(Lambda[:d])         #fraction of total variance function
    r = min([r for r in range(1, d+1) if f(r) >= alpha])    #reduced dimention 
    A = [np.dot(U[:r], D[i]) for i in range(0, n)]          #Reduced dimention data matrix
    return A

#smallest I could get the PCA algorithem
def PCAMin(D, alpha):
    Lambda, U = (lambda eVal, eVec: (np.flip(np.sort(eVal)), np.array([vec for (_, vec) in sorted(zip(eVal, np.transpose(eVec)), reverse=True)])))(*la.eig(mvscvm(D)))
    return [np.dot(U[:min([r for r in range(1, D.shape[1]+1) if sum(Lambda[:r]) / sum(Lambda[:D.shape[1]]) >= alpha])], D[i]) for i in range(0, D.shape[0])]

#the homogeneous quadratic polynomial kernel
def hqpk(xi, xj):
    return np.dot(np.transpose(xi), xj)**2

def KernelPCA(D, Kf, alpha):
    n, d = D.shape 
    K = np.array([[Kf(D[i], D[j]) for i in range(0, n)] for j in range(0, n)])
    print(K)
    O = np.subtract(np.identity(n), 1/n * np.ones((n,n)))   #Centering Matrix
    K = np.dot(np.dot(O, K), O)
    eta, C = (lambda eVal, eVec: ([v.real for v in eVal if v.imag == 0 and v.real > 0.001], eVec))(*la.eig(K))
    print(eta)


with open('iris.txt') as irisFile:
    data = list(csv.reader(irisFile))

#Limit the number of data attributes down to the first three
dataMatrix = np.array(data)[:,:3].astype(np.float)

#Preform PCA 
#reducedDimentionData = np.transpose(PCAMin(dataMatrix, 0.95))
#plt.scatter(reducedDimentionData[0], reducedDimentionData[1])
#plt.show()

#modify the data matrix as done by example 7.7 to prepare for Kernal PCA, display it as shown in fig 7.6
dataMatrix2 = centered(dataMatrix)
dataMatrix2[:, 0] = np.add(np.add(0.2 * np.square(dataMatrix2[:, 0]), np.square(dataMatrix2[:, 1])), 0.1 * np.dot(dataMatrix2[:, 0], dataMatrix2[:, 1]))
plt.scatter(dataMatrix2[:, 0], dataMatrix2[:, 1])
plt.show()

KernelPCA(dataMatrix2, hqpk, 0.95)

#print("Multivariate sample covariance matrix: \n" + str(mvscvm(dataMatrix)))