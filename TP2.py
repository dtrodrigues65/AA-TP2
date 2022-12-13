# -*- coding: utf-8 -*-
"""

@author: Diogo Rodrigues 56153 && Jose Murta 55226
"""

import tp2_aux as aux
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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
    d = (d - np.mean(d)) / np.std(d)
    return d

def agg_clust (n_clusters, matrix):
    #para efeitos de analise pode ser metido connectivity constraints
    #connectivity = kneighbors_graph(matrix, n_neighbors=20, include_self=False)
    #ward = AgglomerativeClustering(n_clusters=n_clusters, connectivity = connectivity)
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
    sse = kmeans.inertia_
    return pred, sse
    
def confusion_matrix(clusters, groups):
    tPositive = 0
    tNegative = 0
    fPositive = 0
    fNegative = 0
    counter = 0
    for i in range(len(clusters)):
        if(groups[i] != 0):
            for j in range(i+1,len(clusters)):
                if(groups[j] != 0):
                    counter+=1
                    if groups[i] == groups[j]:
                        if clusters[i] == clusters[j]:
                            tPositive+=1
                        else:
                            fNegative+=1
                    else:
                        if clusters[i] == clusters[j]:
                            fPositive+=1
                        else:
                            tNegative+=1
    return tPositive, tNegative, fPositive, fNegative, counter

def precision(tPositives, fPositives):
    return tPositives / (tPositives+fPositives)

def recall(tPositives, fNegatives):
    return tPositives / (tPositives+fNegatives)

def rand(tPositives, tNegatives, total):
    numerator =tPositives + tNegatives
    return numerator / total

def f1(precision, recall):
    return 2* ((precision*recall)/(precision+recall))

def purity(n_clusters, clusters, labelsLabeled):
    total = 0
    for i in range(n_clusters):
        nclass1 = 0
        nclass2 = 0
        nclass3 = 0
        for j in range(len(clusters)):
            if (clusters[j] == i):
                if (labels[j] == 1):
                    nclass1+=1
                elif (labels[j] == 2):
                    nclass2+=1
                elif (labels[j] == 3):
                    nclass3+=1
        total += max (nclass1,nclass2,nclass3)
    return total/labelsLabeled
        
def returnExternalIndexes(clust_pred, labels, n_clusters, labelsLabeled):
        tPositive, tNegative, fPositive, fNegative, counter = confusion_matrix(clust_pred, labels)
        precision_aux = precision(tPositive, fPositive)
        recall_aux = recall (tPositive,fNegative)
        rand_aux = rand (tPositive,tNegative,counter)
        f1_aux = f1 (precision_aux,recall_aux)
        purity_aux = purity(n_clusters, clust_pred, labelsLabeled)
        return precision_aux, recall_aux, rand_aux, f1_aux, purity_aux
    

##Main code
labels = np.loadtxt("labels.txt", delimiter= ",")
labels = labels [:,1]
image_matrix = aux.images_as_matrix()
labelsLabeled = len(labels[labels[:]>0]) #pode ser 81

PCA_features = pca(image_matrix)
kernelPCA_features = kernel_pca(image_matrix)
iso_features = iso(image_matrix)
image_features = np.hstack((PCA_features, np.hstack((kernelPCA_features,iso_features))))

image_features =stds(image_features)
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(0.7))
image_features = sel.fit_transform(image_features)

agg_matrix = np.zeros((5, 9))
spectral_matrix = np.zeros((5, 9))
kmeans_matrix = np.zeros((5, 9))

###Clustering
clusterArray = np.arange(2, 11, 1)
for n in clusterArray:
    print("--------------------CLUSTERS =", n, "--------------------")
    #Agglomerative
    agg_clust_pred = agg_clust(n, image_features)
    precision_agg, recall_agg, rand_agg, f1_agg, purity_agg = returnExternalIndexes(agg_clust_pred, labels, n, labelsLabeled)
    print("-----Agglomoretive with", n, "clusters-------")
    print("Precision:", precision_agg)
    agg_matrix[0][n-2] = precision_agg
    print("Recall:", recall_agg)
    agg_matrix[1][n-2] = recall_agg
    print("Rand:", rand_agg)
    agg_matrix[2][n-2] = rand_agg
    print("F1:", f1_agg)
    agg_matrix[3][n-2] = f1_agg
    print("Purity:", purity_agg)
    agg_matrix[4][n-2] = purity_agg
    
    if n == 6:
    	aux.report_clusters(np.array(range(image_features.shape[0])), agg_clust_pred, "test.html")
    
    #Spectral
    spectral_clust_pred = spectral_clust(n, image_features)
    precision_spe, recall_spe, rand_spe, f1_spe, purity_spe = returnExternalIndexes(spectral_clust_pred, labels, n, labelsLabeled)
    print("-----Spectral with", n, "clusters-------")
    print("Precision:", precision_spe)
    spectral_matrix[0][n-2] = precision_spe
    print("Recall:", recall_spe)
    spectral_matrix[1][n-2] = recall_spe
    print("Rand:", rand_spe)
    spectral_matrix[2][n-2] = rand_spe
    print("F1:", f1_spe)
    spectral_matrix[3][n-2] = f1_spe
    print("Purity:", purity_spe)
    spectral_matrix[4][n-2] = purity_spe
    
    if n == 10:
        aux.report_clusters(np.array(range(image_features.shape[0])), spectral_clust_pred, "test2.html")
    
    #K-means
    kmeans_clust_pred, sse_kmeans = k_means_clust(n, image_features)
    precision_kmeans, recall_kmeans, rand_kmeans, f1_kmeans, purity_kmeans = returnExternalIndexes(kmeans_clust_pred, labels, n, labelsLabeled)
    print("-----K-means with", n, "clusters-------")
    print("Precision:", precision_kmeans)
    kmeans_matrix[0][n-2] = precision_kmeans
    print("Recall:", recall_kmeans)
    kmeans_matrix[1][n-2] = recall_kmeans
    print("Rand:", rand_kmeans)
    kmeans_matrix[2][n-2] = rand_kmeans
    print("F1:", f1_kmeans)
    kmeans_matrix[3][n-2] = f1_kmeans
    print("Purity:", purity_kmeans)
    kmeans_matrix[4][n-2] = purity_kmeans
    print("K-means loss / SSE:", sse_kmeans)
    
    if n==8:
        aux.report_clusters(np.array(range(image_features.shape[0])), kmeans_clust_pred, "test3.html")
    

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10,8))
fig.subplots_adjust(right=1.5, hspace=0.4)
titleArray = ["Precision", "Recall", "Rand", "F1", "Purity"]
counter = 0

for i in range(0,2):
    for j in range(0,3):
        axs[i,j].set_title(titleArray[counter])
        axs[i,j].set_xlabel('Number of clusters')
        axs[i,j].set_ylabel(titleArray[counter])
        agg_legend = mlines.Line2D([], [], color='red', marker='_', label='Agglomerative', linestyle ='None')
        spectral_legend = mlines.Line2D([], [], color='blue', marker='_', label='Spectral', linestyle ='None')
        kmeans_legend = mlines.Line2D([], [], color='green', marker='_', label='K-means', linestyle ='None')
        axs[i,j].legend(handles=[agg_legend, spectral_legend, kmeans_legend])
        
        axs[i,j].plot(clusterArray, agg_matrix[counter], '-r') 
        axs[i,j].plot(clusterArray, spectral_matrix[counter], '-b')
        axs[i,j].plot(clusterArray, kmeans_matrix[counter], '-g')
        
        
        counter+=1
        if counter == 5:
            break















