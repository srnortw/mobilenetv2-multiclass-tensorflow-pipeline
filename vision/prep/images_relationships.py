# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:14:34 2025

@author: Serkan
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import pdb

class images_relationships_c():
    def __init__(self):
        pass
        
    
    def compute_histogram_correlation(self,sconcat_rgbh):
        
        correlation_matrix = 1 - squareform(pdist(sconcat_rgbh, metric='correlation'))
        return correlation_matrix


    def anomaly_detection(self,correlation_matrix):
        # Fitting the model
        clf = IsolationForest(contamination=0.1)

        #mask=(np.mean(x[:, :3], axis=1) >= 40) & (np.mean(x[:, :3], axis=1) <= 215) & (np.mean(x[:, 3::3], axis=1) >= 40)
        
        #xf = x[mask]

        #print(xf.shape)
            
        
        clf.fit(correlation_matrix)
        
        # Predicting anomalies
        predictions = clf.predict(correlation_matrix)
        
        # -1 for anomalies, 1 for normal points
        print(predictions)
        
        # predictions[np.where(predictions==1)]=True
        # predictions[np.where(predictions==-1)]=False

        #indexes = np.where(predictions == 1)
        # print(indexes[0])
        # print(x)
        
        indexes=predictions==1
        #pdb.set_trace()
        
        return indexes

    def cluster_quantity_test(self,X):
        
        #pdb.set_trace()
        
        # Compute the within-cluster sum of squares for different number of clusters
        wcss = []
        
        
        for i in range(1,100):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        # Plot the results
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 100), wcss, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        
        # big leap changes like elbow thinks us this quantity clusters is such a good choice to pick because of distances between points and its cluster.(elbow method)
        

    # Function to cluster images using K-means
    def cluster_images_kmeans(self,X, num_clusters):
        
        num_clusters = int(num_clusters)
        
    
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(X)
        clusters =kmeans.labels_
        #print(f"{clusters.shape}")
        #print(clusters)
            
        centroids =kmeans.cluster_centers_
        
        print(f"x x {centroids.shape}")
        print(f"x x {centroids}")
            
        wcss=kmeans.inertia_
        print(f"x x {wcss}")
        
        
        #print(clusters.shape,'sx',centroids.shape,'s',wcss,'ss')

        return clusters,centroids,wcss
        

    def pca(self,f,s,wcss,num_clusters=1):#(f,s,wcss)
        
        
        
        fig = plt.figure(figsize=(10, 8))
        
        ax = fig.add_subplot(111)#,projection='3d'
        
        
        ax.set_title(f'2D PCA of Images Correlations also Wcss is {wcss}')

        print(f.shape, s.shape)

        m=(f,s)
        
        cluster_size=m[1].shape[0]#f[1]

        data_with_centroid=np.vstack(m)#f

        
        pca = PCA(n_components=2)
        
        principal_components = pca.fit_transform(data_with_centroid)
        
        datas=principal_components[:-cluster_size,:]
        
        centroid=principal_components[-cluster_size:,:]
        
        
        pc_df = pd.DataFrame(data=datas, columns=['PC1', 'PC2'])
        
        
        t=['Image Samples Correlations',f'Its {num_clusters} Clusters Centroids']
        
        ax.scatter(pc_df['PC1'], pc_df['PC2'],c='blue',label=f'{t[0]}')#s
        
        
        #ax.legend()
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        
        
        pc_df = pd.DataFrame(data=centroid, columns=['PC1', 'PC2'])
        
        ax.scatter(pc_df['PC1'], pc_df['PC2'],c='red',label=f'{t[1]}')#s
        
        
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        plt.show()
        
        # for i,j in enumerate(f):
            
        #     # # # Standardize the data
        #     # scaler = StandardScaler()
        #     data_std =j#scaler.fit_transform(j)
            
            
        #     # Apply PCA
        #     pca = PCA(n_components=2)#3
            
        #     pdb.set_trace()
        #     principal_components = pca.fit_transform(data_std)
                
        #     # Create a DataFrame with the principal components
        #     pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2']) #, 'PC3'
            
        #     ax.scatter(pc_df['PC1'], pc_df['PC2'],c=c[i],label=f'{s[i]}')#, pc_df['PC3']
                
        #     # Plot the principal components in 3D
            
        #     #pdb.set_trace()

        #     #ax.axis((-5,5,-5,5,-5,5))
                
        #     ax.legend()
        #     ax.set_xlabel('x')
        #     ax.set_ylabel('y')
        #     #ax.set_ylabel('z')
        
        # plt.show()

        return f