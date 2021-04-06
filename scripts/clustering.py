import pandas as pd
import numpy as np
from sklearn.cluster import KMeans 
from sklearn.metrics import rand_score, fowlkes_mallows_score, davies_bouldin_score

class clustering():

    @staticmethod
    def kSrednich(dataframe, K):
        clustering = KMeans(n_clusters = K).fit_predict(dataframe)

        return clustering

    @staticmethod
    def compute_rand_index(true, pred):
        return (rand_score(true, pred))
    
    @staticmethod
    def compute_fowlkes_mallows(true, pred):
        return (fowlkes_mallows_score(true, pred))

    @staticmethod
    def compute_davies_bouldin(dataset, clustering):
        return (davies_bouldin_score(dataset, clustering))