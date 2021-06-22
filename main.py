from timeit import timeit

import pandas
import numpy as np

from scripts.dataLoader import DataLoader
from scripts.clustering import Clustering, DataClustering
from scripts.preprocessing import Preprocessor
from scripts.mcdm import MCDM

def main():
    path = './data/test'
    data_loader = DataLoader(path)
    files = data_loader.getAllFiles()

    results = []

    alternative_names = ['KMeans', 'Lloyd', 'DBScan', "Ward", "fuzzy", "optics", "gaussian mixture"]
    index = ['Rand Index', 'Fowlkes Mallows', 'Adjusted Rand Index HA', 'Jaccard Index']

    for file in files:
        data = data_loader.readFile(file)
        res = []
        df = pandas.DataFrame()
        
        if file == 'breast-cancer-wisconsin.data':
            data = Preprocessor.breast_cancer_classes(data)
            res = DataClustering.breast_cancer_clustering(data)
            df = pandas.DataFrame(res)
            results.append(res)

        if file == 'german_credit_data.csv':
            res = DataClustering.german_credit_clustering(data)
            df = pandas.DataFrame(res)
            results.append(res)
        
        if file == 'glass.csv':
            res = DataClustering.glass_clustering(data)
            df = pandas.DataFrame(res)
            results.append(res)
        
        if file == 'zoo.csv':
            res = DataClustering.zoo_clustering(data)
            df = pandas.DataFrame(res)
            results.append(res)
        
        if file == 'wine.data':
            res = DataClustering.wine_clustering(data)
            df = pandas.DataFrame(res)
            results.append(res)
        
        if file == 'kaggle_Interests_group.csv':
            data = Preprocessor.clean_people_interest(data)
            res = DataClustering.people_clustering(data)
            df = pandas.DataFrame(res)
            results.append(res)
        
        filename = str(file) + "_results"
        if not df.empty:
            df.to_csv(path_or_buf=filename,  header=index)
        
        print("done " + file + "\n")

    alternative_names = ['KMeans', 'Lloyd', 'DBScan', "Ward", "fuzzy", "optics", "gaussian mixture"]
    benefit_list = [True, True, True, True]

    #1. Rand index 2. Fowlkes Mallows 3. Adjusted Rand Index (HA) 4. Jaccard

    
    print("ari Rand index breast cancer KMeans: %.2f" %results[0][0][2])
    print("Rand index breast cancer lloyd: %.2f" %results[0][1][2])
    print("Rand index breast cancer dbscan: %.2f" %results[0][2][2])
    print("Rand index breast cancer ward: %.2f" %results[0][3][2])
    print("Rand index breast cancer fuzzy: %.2f" %results[0][4][2])
    print("Rand index breast cancer optics: %.2f" %results[0][5][2])
    print("Rand index breast cancer gaussian: %.2f" %results[0][6][2])
    
    print("\n\n")
    for i in range(len(results)):
        for j in range(len(results[i])):
            for k in range(len(results[i][j])):
                if results[i][j][k] < 0.0:
                    results[i][j][k] = 0.0

    types = np.array([-1, 1, 1, 1])

    topsis = MCDM.group_topsis(results, alternative_names, benefit_list)
    wsm = MCDM.group_wsm(results, alternative_names)
    copras = MCDM.group_copras(results, alternative_names, types)

    for res in results:
        print(res)
        print("\n")

    print("\nGROUP TOPSIS:")
    print(topsis)
    print("\nGROUP WSM:")
    print(wsm)
    print("\nCOPRAS Mcdm Method")
    print(copras)

    print('\n\n')
    print("TOPSIS Borda Count Method")
    MCDM.borda_count(topsis)
    print("WSM Borda Count Method")
    MCDM.borda_count(wsm)
    print("COPRAS Borda Count Method")
    MCDM.borda_count(copras)
    
    

if __name__ == '__main__':
    main()