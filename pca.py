from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eig

from matplotlib import pyplot as plt
import numpy as np


#0: setosa, 1: versicolor, 2: Virginica

#Usamos a função na biblioteca pronta para validar os resultados, pode ser ignorado
def biblePCA():
	iris = load_iris()

	#x = matriz, y = classe
	x, y = iris.data, iris.target 
	pca = PCA(2)
	pca.fit(x)
	B = pca.transform(x)

def myPCA():
	iris = load_iris()

	#x = matriz, y = classe
	x, y = iris.data, iris.target 		

	#print(x)

	#Normaliza
	x = StandardScaler().fit_transform(x)
	
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
	newSamples = x.dot(newMatrix)
	print("\n", newSamples)

#biblePCA()
myPCA()
