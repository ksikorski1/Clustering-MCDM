import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import rand_score, fowlkes_mallows_score, davies_bouldin_score, adjusted_rand_score, jaccard_score, silhouette_score
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

from .preprocessing import Preprocessor

class Clustering():

    @staticmethod
    def kmeans(dataframe, K):

        kmeans = KMeans(n_clusters = K, n_init=100)
        kmeans.fit(dataframe)
        clustering_res = kmeans.predict(dataframe)
        return clustering_res
    
    @staticmethod
    def dbscan(dataframe, true, min_pts = 5, max_cluster_count = 10):
        i = 0.5
        score = 100
        score2 = -1
        clustering_res = []
        epsilon = 0
        temp_score = 99
        temp_score2 = 99
        while  (i < 12.0):
            clustering_temp = DBSCAN(eps = i, min_samples= min_pts).fit_predict(dataframe)
            uniq_labels = np.unique(clustering_temp)
            if len(uniq_labels) > 2 and len(uniq_labels) < max_cluster_count:
                temp_score = Clustering.compute_davies_bouldin(dataframe, clustering_temp)
                temp_score2 = Clustering.compute_silhouette_score(dataframe, clustering_temp)
                if temp_score2 > score2:
                    if temp_score < score:
                        clustering_res = clustering_temp
                        score = temp_score
                        score2 = temp_score2
                        epsilon = i
            i+=0.1
        
        uniq_labels = np.unique(clustering_res)
        print("Epsilon = %.2f" %epsilon)
        print("DBSCAN Number of clusters: " + str(len(uniq_labels)))
        print("Davies Boulding score: %.4f" %score)
        print("Silhouette score: %.4f" %score2)
        print(uniq_labels)
        return clustering_res
    
    @staticmethod
    def wards_method(data, n):
        clustering_res = AgglomerativeClustering(n_clusters=n, linkage='ward').fit_predict(data)
        return clustering_res
    
    @staticmethod
    def draw_dendrogram(data):
        dend = shc.dendrogram(shc.linkage(data, method='ward'))
        return dend

    @staticmethod
    def compute_rand_index(true, pred):
        return (rand_score(true, pred))
    
    @staticmethod
    def compute_fowlkes_mallows(true, pred):
        return (fowlkes_mallows_score(true, pred))

    @staticmethod
    def compute_davies_bouldin(dataset, clustering):
        return (davies_bouldin_score(dataset, clustering))
    
    @staticmethod
    def compute_silhouette_score(dataset, clustering):
        return (silhouette_score(dataset, clustering))
    
    @staticmethod
    def compute_adjusted_rand_score(true, pred):
        return (adjusted_rand_score(true, pred))
    
    @staticmethod
    def compute_jaccard_score(true, pred):
        return (jaccard_score(true, pred, average='weighted'))
    
    @staticmethod
    def compute_all_external_metrics(data, true, pred):
        result = []

        result.append(Clustering.compute_rand_index(true, pred))
        result.append(Clustering.compute_fowlkes_mallows(true, pred))            
        result.append(Clustering.compute_adjusted_rand_score(true, pred))
        result.append(Clustering.compute_jaccard_score(true, pred))

        return result
    
    @staticmethod
    def compute_best_clustering(data):
        kMeansResult_temp =  0
        wardResult_temp = 0
        prev_kmeans = 0
        prev_ward = 0
        k1 = -1
        k2 = -1
        for i in range(2, 8):

            kMeansResult_temp = DataClustering.kmeans(data, i)
            wardResult_temp = DataClustering.wards_method(data, i)

            kmeans_temp = DataClustering.compute_silhouette_score(data, kMeansResult_temp)
            ward_temp = DataClustering.compute_silhouette_score(data, wardResult_temp)

            if (kmeans_temp) > (prev_kmeans):
                kMeansResult = kMeansResult_temp
                prev_kmeans = kmeans_temp
                k1 = i
            if (ward_temp) > (prev_ward):
                wardResult = wardResult_temp
                prev_ward = ward_temp
                k2 = i
        print("KMeans clusters: ", k1)
        print("Ward clusters:", k2)

        return kMeansResult, wardResult


class DataClustering(Clustering):

    @staticmethod
    def breast_cancer_clustering(df):
        measures = []
        data = df
        data = data.drop('id', axis=1)
        breastcancer = data['class']
        data = data.drop('class', axis=1)

        kMeansResult, wardResult = DataClustering.compute_best_clustering(data)

        dbscanResult = DataClustering.dbscan(data, breastcancer)

        measures.append(DataClustering.compute_all_external_metrics(data, breastcancer, kMeansResult))
        measures.append(DataClustering.compute_all_external_metrics(data, breastcancer, dbscanResult))
        measures.append(DataClustering.compute_all_external_metrics(data, breastcancer, wardResult))
        
        return measures
        
    @staticmethod
    def german_credit_clustering(df):
        measures = []
        data = df
        creditability = data['Creditability']
        data = data.drop('Creditability', axis=1)

        kMeansResult, wardResult = DataClustering.compute_best_clustering(data)

        dbscanResult = DataClustering.dbscan(data, creditability, min_pts = 3)

        measures.append(DataClustering.compute_all_external_metrics(data, creditability, kMeansResult))
        measures.append(DataClustering.compute_all_external_metrics(data, creditability, dbscanResult))
        measures.append(DataClustering.compute_all_external_metrics(data, creditability, wardResult))

        return measures

    @staticmethod
    def glass_clustering(df):
        measures = []
        data = df
        glass_type = data['Type']
        glass_type = Preprocessor.glass_type_encode(glass_type)
        data = data.drop('Type', axis=1)
        #print(glass_type)

        dbscanResult = DataClustering.dbscan(data, glass_type, min_pts = 5)
        
        kMeansResult, wardResult = DataClustering.compute_best_clustering(data)

        measures.append(DataClustering.compute_all_external_metrics(data, glass_type, kMeansResult))
        measures.append(DataClustering.compute_all_external_metrics(data, glass_type, dbscanResult))
        measures.append(DataClustering.compute_all_external_metrics(data, glass_type, wardResult))

        return measures