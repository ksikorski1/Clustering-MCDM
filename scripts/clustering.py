import pandas as pd
import numpy as np
from scipy.sparse import data
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import rand_score, fowlkes_mallows_score, davies_bouldin_score, adjusted_rand_score, jaccard_score, silhouette_score
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, Normalizer
from fcmeans import FCM

from .preprocessing import Preprocessor

class Clustering():

    @staticmethod
    def kmeanspp(dataframe, K):

        kmeans = KMeans(init="k-means++", n_clusters = K, n_init=50)
        kmeans.fit(dataframe)
        clustering_res = kmeans.predict(dataframe)
        return clustering_res

    @staticmethod
    def kmeans_lloyd(dataframe, K):

        kmeans = KMeans(init="random", n_clusters = K, n_init=50, max_iter=100, algorithm="full")
        kmeans.fit(dataframe)
        clustering_res = kmeans.predict(dataframe)
        return clustering_res
    
    @staticmethod
    def dbscan(dataframe, start, stop, gap, min_pts = 4, max_cluster_count = 10):
        i = start
        score = 100
        score2 = -1
        clustering_res = []
        epsilon = 0
        while  (i < stop):
            clustering_temp = DBSCAN(eps = i, min_samples= min_pts).fit_predict(dataframe)
            uniq_labels = np.unique(clustering_temp)
            if (len(uniq_labels) > 2) and (len(uniq_labels) < max_cluster_count):
                temp_score = Clustering.compute_davies_bouldin(dataframe, clustering_temp)
                temp_score2 = Clustering.compute_silhouette_score(dataframe, clustering_temp)
                if temp_score2 >= score2:
                    if temp_score <= score:
                        clustering_res = clustering_temp
                        score = temp_score
                        score2 = temp_score2
                        epsilon = i
            i+=gap
        
        uniq_labels = np.unique(clustering_res)
        print("Epsilon = %.4f" %epsilon)
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
    def fuzzy_cmeans(data, n):
        fcm = FCM(n_clusters=n)
        fcm.fit(data)
        fcm_labels = fcm.predict(data)
        return fcm_labels
    
    @staticmethod
    def optics(dataframe, start = 0.1, stop = 6.0, gap = 0.2, min_pts = 5, max_cluster_count = 10):
        i = start
        score = 100
        score2 = -1
        clustering_res = []
        epsilon = 0
        while  (i < stop):
            clustering_temp = None
            clustering_temp = OPTICS(eps = i, min_samples=min_pts, cluster_method="dbscan").fit_predict(dataframe)
            uniq_labels = np.unique(clustering_temp)
            if (len(uniq_labels) > 2) and (len(uniq_labels) < max_cluster_count):
                temp_score = Clustering.compute_davies_bouldin(dataframe, clustering_temp)
                temp_score2 = Clustering.compute_silhouette_score(dataframe, clustering_temp)
                if temp_score2 >= score2:
                    if temp_score <= score:
                        clustering_res = clustering_temp
                        score = temp_score
                        score2 = temp_score2
                        epsilon = i
            i+=gap
        uniq_labels = np.unique(clustering_res)
        print("Epsilon = %.4f" %epsilon)
        print("OPTICS Number of clusters: " + str(len(uniq_labels)))
        print("Davies Boulding score: %.4f" %score)
        print("Silhouette score: %.4f" %score2)
        print(uniq_labels)
        return clustering_res

    @staticmethod
    def optics_auto(dataframe):
        return OPTICS().fit_predict(dataframe)

    @staticmethod
    def gaussian_mixture(dataframe, components = 2):
        return GaussianMixture(n_components=components, n_init=10).fit_predict(dataframe)
    
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
        prev_kmeans = -1
        prev_lloyd = -1
        prev_ward = -1
        prev_fuzzy = -1

        prev_kmeans1 = 50
        prev_lloyd1 = 50
        prev_ward1 = 50
        prev_fuzzy1 = 50
        for i in range(2, 10):

            kMeansResult_temp = DataClustering.kmeanspp(data, i)
            lloydResult_temp  = DataClustering.kmeans_lloyd(data, i)
            wardResult_temp = DataClustering.wards_method(data, i)
            fuzzyResult_temp = DataClustering.fuzzy_cmeans(data, i)

            kmeans_temp = DataClustering.compute_silhouette_score(data, kMeansResult_temp)
            lloyd_temp = DataClustering.compute_silhouette_score(data, lloydResult_temp)
            ward_temp = DataClustering.compute_silhouette_score(data, wardResult_temp)
            fuzzy_temp = DataClustering.compute_silhouette_score(data, fuzzyResult_temp)

            kmeans_temp1 = DataClustering.compute_davies_bouldin(data, kMeansResult_temp)
            lloyd_temp1= DataClustering.compute_davies_bouldin(data, lloydResult_temp)
            ward_temp1 = DataClustering.compute_davies_bouldin(data, wardResult_temp)
            fuzzy_temp1 = DataClustering.compute_davies_bouldin(data, fuzzyResult_temp)

            if ((kmeans_temp > prev_kmeans) and (kmeans_temp1 < prev_kmeans1)):
                kMeansResult = kMeansResult_temp
                prev_kmeans = kmeans_temp
                prev_kmeans1 = kmeans_temp1
                k1 = i

            if ((lloyd_temp > prev_lloyd) and (lloyd_temp1 < prev_lloyd1)):
                lloydResult = lloydResult_temp
                prev_lloyd = lloyd_temp
                prev_lloyd1 = lloyd_temp1
                k2 = i

            if ((ward_temp > prev_ward) and (ward_temp1 < prev_ward1)):
                wardResult = wardResult_temp
                prev_ward = ward_temp
                prev_ward1 = ward_temp1
                k3 = i

            if ((fuzzy_temp > prev_fuzzy) and (fuzzy_temp1 < prev_fuzzy1)):
                fuzzyResult = fuzzyResult_temp
                prev_fuzzy = fuzzy_temp
                prev_fuzzy1 = fuzzy_temp1
                k4 = i

        print("KMeans clusters: ", k1)
        print("Lloyd clusters: ", k2)
        print("Ward clusters:", k3)
        print("Fuzzy clustering:", k4)

        return kMeansResult, lloydResult,  wardResult, fuzzyResult

    @staticmethod
    def do_all_clusterings(data, classes, min_pts, gaussian_components, start, stop, gap, scaler = False):
        measures = []
        if scaler:
            scaler = Normalizer()
            data = scaler.fit_transform(data)
        kMeansResult, lloydResult,  wardResult, fuzzyResult = DataClustering.compute_best_clustering(data)

        dbscanResult = DataClustering.dbscan(data, start, stop, gap, min_pts)
        opticsResult = DataClustering.optics(data, start, stop, gap, min_pts)

        gaussianResult = DataClustering.gaussian_mixture(data, gaussian_components)


        measures.append(DataClustering.compute_all_external_metrics(data, classes, kMeansResult))
        measures.append(DataClustering.compute_all_external_metrics(data, classes, lloydResult))
        measures.append(DataClustering.compute_all_external_metrics(data, classes, dbscanResult))
        measures.append(DataClustering.compute_all_external_metrics(data, classes, wardResult))
        measures.append(DataClustering.compute_all_external_metrics(data, classes, fuzzyResult))
        measures.append(DataClustering.compute_all_external_metrics(data, classes, opticsResult))
        measures.append(DataClustering.compute_all_external_metrics(data, classes, gaussianResult))

        return measures


class DataClustering(Clustering):

    @staticmethod
    def breast_cancer_clustering(df):
        data = df
        data = data.drop('id', axis=1)
        breastcancer = data['class']
        data = data.drop('class', axis=1)

        measures = DataClustering.do_all_clusterings(data, breastcancer, 4, 2, start=0.05, stop=3.0, gap=0.05)

        return measures
        
    @staticmethod
    def german_credit_clustering(df):
        measures = []
        data = df
        creditability = data['Creditability']
        data = data.drop('Creditability', axis=1)

        measures = DataClustering.do_all_clusterings(data, creditability, 5, 2, start=7.05, stop=11.0, gap=0.05)

        return measures

    @staticmethod
    def glass_clustering(df):
        measures = []
        data = df
        glass_type = data['Type']
        glass_type = Preprocessor.glass_type_encode(glass_type)
        data = data.drop('Type', axis=1)
        #print(glass_type)

        measures = DataClustering.do_all_clusterings(data, glass_type, 5, 6, start=0.001, stop=0.1, gap=0.002, scaler=True)

        return measures

    @staticmethod
    def zoo_clustering(df):
        measures = []
        data = df
        data = data.drop('animal_name', axis=1)
        animals = data['class_type']
        data = data.drop('class_type', axis=1)

        measures = DataClustering.do_all_clusterings(data, animals, 5, 7, start=0.05, stop=5.0, gap=0.1)

        return measures

    @staticmethod
    def wine_clustering(df):
        measures = []
        data = df
        wine = data['class']
        data = data.drop('class', axis=1)

        measures = DataClustering.do_all_clusterings(data, wine, 4, 3, start=0.001, stop=0.1, gap=0.002, scaler=True)

        return measures

    @staticmethod
    def people_clustering(df):
        measures = []
        data = df
        group = data['group']
        data = data.drop('group', axis=1)

        measures = DataClustering.do_all_clusterings(data, group, min_pts=12, gaussian_components=4, start=0.88, stop=1.02, gap=0.02, scaler=False)
        #measures = DataClustering.dbscan(data, min_pts=12, start=0.01, stop=1.1, gap=0.05)

        return measures