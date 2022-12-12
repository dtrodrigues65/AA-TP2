# -*- coding: utf-8 -*-
"""

@author: Diogo Rodrigues 56153 && Jose Murta 55226
"""

import tp2_aux as aux
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
import numpy as np

##Functions

def pca (matrix):
    pca = PCA(n_components=6)
    pca.fit(matrix)
    t_data = pca.transform(matrix)
    return t_data

def kernel_pca (matrix):
    kernel_pca = KernelPCA(n_components=6, kernel = 'rbf')
    kernel_pca.fit(matrix)
    t_data = kernel_pca.transform(matrix)
    return t_data

def iso (matrix):
    iso = Isomap(n_components=6)
    iso.fit(matrix)
    t_data = iso.transform(matrix)
    return t_data

def stds (d):
    data = (d - np.mean(d, 0)) / np.std(d, 0)
    return data


##Main code
labels = np.loadtxt("labels.txt", delimiter= ",")
labels = labels [:,1]
image_matrix = aux.images_as_matrix()

PCA_features = pca(image_matrix)
kernelPCA_features = kernel_pca(image_matrix)
iso_features = iso(image_matrix)
image_features = np.hstack((PCA_features, np.hstack((kernelPCA_features,iso_features))))

image_features =stds(image_features) ##can be or cannot be

###Clustering















