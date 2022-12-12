# -*- coding: utf-8 -*-
"""

@author: Diogo Rodrigues 56153 && Jose Murta 55226
"""

import tp2_aux as aux
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans

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

def agg_clust (n_clusters, matrix):
    #para efeitos de analise pode ser metido connectivity constraints
    ward = AgglomerativeClustering(n_clusters=n_clusters)
    pred = ward.fit_predict(matrix)
    return pred

def spectral_clust(n_clusters, matrix):
    clustering = SpectralClustering(n_clusters=n_clusters,assign_labels='cluster_qr')
    pred = clustering.fit_predict(matrix)
    return pred

def k_means_clust(n_clusters, matrix):
    kmeans = KMeans(n_clusters=n_clusters)
    pred = kmeans.fit_predict(matrix)
    return pred
    
    


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
#Agglomerative
agg_clust_pred = agg_clust(6, image_features)
aux.report_clusters(np.array(range(image_features.shape[0])), agg_clust_pred, "test.html")

#Spectral
spectral_clust_pred = spectral_clust(3, image_features)
aux.report_clusters(np.array(range(image_features.shape[0])), spectral_clust_pred, "test2.html")

#K-means
kmeans_clustPred = k_means_clust(6, image_features)
aux.report_clusters(np.array(range(image_features.shape[0])), kmeans_clustPred, "test3.html")












