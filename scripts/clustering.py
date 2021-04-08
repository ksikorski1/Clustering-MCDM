import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import rand_score, fowlkes_mallows_score, davies_bouldin_score, adjusted_rand_score, jaccard_score
from .preprocessing import Preprocessor

class Clustering():

    @staticmethod
    def kmeans(dataframe, K):

        kmeans = KMeans(n_clusters = K, n_init=100)
        kmeans.fit(dataframe)
        clustering_res = kmeans.predict(dataframe)
        return clustering_res
    
    @staticmethod
    def dbscan(dataframe, true, min_pts = 5):
        i = 0.5
        score = 0
        clustering_res = []
        epsilon = 0
        while  (i < 12.0):
            clustering_temp = DBSCAN(eps = i, min_samples= min_pts).fit_predict(dataframe)
            temp_score = Clustering.compute_adjusted_rand_score(true, clustering_temp)
            uniq_labels = np.unique(clustering_temp)
            if temp_score > score and len(uniq_labels) > 2:
                clustering_res = clustering_temp
                score = temp_score
                epsilon = i
            i+=0.1
        
        uniq_labels = np.unique(clustering_res)
        print("Epsilon = %.2f" %epsilon)
        print("DBSCAN Number of clusters: " + str(len(uniq_labels)))
        print(uniq_labels)
        return clustering_res
    
    @staticmethod
    def wards_method(data, n):
        clustering_res = AgglomerativeClustering(n_clusters=n, linkage='ward').fit_predict(data)
        return clustering_res

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
        #result.append(Clustering.compute_davies_bouldin(data, pred))

        return result

class DataClustering(Clustering):

    @staticmethod
    def breast_cancer_clustering(df):
        measures = []
        data = df
        data = data.drop('id', axis=1)
        breastcancer = data['class']
        data = data.drop('class', axis=1)

        kMeansResult_temp =  0
        wardResult_temp = 0
        prev_kmeans = []
        prev_ward = []
        k1 = 0
        k2 = 0
        for i in range(2, 8):

            kMeansResult_temp = DataClustering.kmeans(data, i)
            wardResult_temp = DataClustering.wards_method(data, i)

            kmeans_temp = DataClustering.compute_all_external_metrics(data, breastcancer, kMeansResult_temp)
            ward_temp = DataClustering.compute_all_external_metrics(data, breastcancer, wardResult_temp)

            if (sum(kmeans_temp) > sum(prev_kmeans)):
                kMeansResult = kMeansResult_temp
                prev_kmeans = kmeans_temp
                k1 = i
            if (sum(ward_temp) > sum(prev_ward)):
                wardResult = wardResult_temp
                prev_ward = ward_temp
                k2 = i
        print("klastrow: ", k1, k2)

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

        kMeansResult_temp =  0
        wardResult_temp = 0
        prev_kmeans = []
        prev_ward = []
        k1 = 0
        k2 = 0
        for i in range(2, 8):

            kMeansResult_temp = DataClustering.kmeans(data, i)
            wardResult_temp = DataClustering.wards_method(data, i)

            kmeans_temp = DataClustering.compute_all_external_metrics(data, creditability, kMeansResult_temp)
            ward_temp = DataClustering.compute_all_external_metrics(data, creditability, wardResult_temp)

            if (sum(kmeans_temp) > sum(prev_kmeans)):
                kMeansResult = kMeansResult_temp
                prev_kmeans = kmeans_temp
                k1 = i
            if (sum(ward_temp) > sum(prev_ward)):
                wardResult = wardResult_temp
                prev_ward = ward_temp
                k2 = i
        print("klastrow: ", k1, k2)

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
        
        kMeansResult_temp =  0
        wardResult_temp = 0
        prev_kmeans = []
        prev_ward = []
        k1 = 0
        k2 = 0
        for i in range(2, 8):

            kMeansResult_temp = DataClustering.kmeans(data, i)
            wardResult_temp = DataClustering.wards_method(data, i)

            kmeans_temp = DataClustering.compute_all_external_metrics(data, glass_type, kMeansResult_temp)
            ward_temp = DataClustering.compute_all_external_metrics(data, glass_type, wardResult_temp)

            if (sum(kmeans_temp) > sum(prev_kmeans)):
                kMeansResult = kMeansResult_temp
                prev_kmeans = kmeans_temp
                k1 = i
            if (sum(ward_temp) > sum(prev_ward)):
                #print(ward_temp, prev_ward)
                wardResult = wardResult_temp
                prev_ward = ward_temp
                k2 = i
        print("klastrow: ", k1, k2)
        measures.append(DataClustering.compute_all_external_metrics(data, glass_type, kMeansResult))
        measures.append(DataClustering.compute_all_external_metrics(data, glass_type, dbscanResult))
        measures.append(DataClustering.compute_all_external_metrics(data, glass_type, wardResult))

        return measures