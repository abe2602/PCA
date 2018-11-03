from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from numpy.linalg import eig

from matplotlib import pyplot as plt
import numpy as np


#0: setosa, 1: versicolor, 2: Virginica
def biblePCA():
	iris = load_iris()

	#x = matriz, y = classe
	x, y = iris.data, iris.target 
	print("PRONTO")
	pca = PCA(1)
	pca.fit(x)
	B = pca.transform(x)
	print(B)

def plotPCA(y, x, newMatrix):
	"""feature_vector = x.dot(newMatrix)
	print (feature_vector.T)
	plt.scatter(feature_vector[0], feature_vector[1], color='r', marker='x')
	plt.show()"""
	plt.plot([1, 2, 3, 4])
	plt.ylabel('some numbers')
	plt.show()

def myPCA():
	iris = load_iris()

	#x = matriz, y = classe
	x, y = iris.data, iris.target 	
	
	#Matriz de covariança
	cov = np.cov(x.T)

	#Covalor e Covetor
	val, vec = eig(cov)

	#Covalor e covetor equivalentes juntos em pares, para ordenar
	pairs = [(np.abs(val[i]), vec[:,i]) for i in range(len(val))]
	pairs.sort()
	pairs.reverse()

	#Escolhe os dois primeiros(são os que mais "ajudam") e cria a matriz nova
	newMatrix = np.hstack((pairs[0][1].reshape(4, 1), pairs[1][1].reshape(4, 1)))
	print(newMatrix)

	#plotPCA(y, x, newMatrix)

#biblePCA()
myPCA()
