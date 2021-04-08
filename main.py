from timeit import timeit

from scripts.dataLoader import DataLoader
from scripts.clustering import Clustering, DataClustering
from scripts.preprocessing import Preprocessor
from scripts.mcdm import MCDM

def main():
    path = './data/test'
    data_loader = DataLoader(path)
    files = data_loader.getAllFiles()

    results = []

    for file in files:
        data = data_loader.readFile(file)

        if file == 'breast-cancer-wisconsin.data':
            data = Preprocessor.breast_cancer_classes(data)
            results.append(DataClustering.breast_cancer_clustering(data))

        if file == 'german_credit_data.csv':
            results.append(DataClustering.german_credit_clustering(data))

        if file == 'glass.csv':
            results.append(DataClustering.glass_clustering(data))

        print("done " + file + "\n")

    alternative_names = ['KMeans', 'DBScan', "Ward"]
    benefit_list = [True, True, True, True]

    #print(results[2])
    
    topsis = MCDM.group_topsis(results, alternative_names, benefit_list)
    wsm = MCDM.group_wsm(results, alternative_names)

    print("\nGROUP TOPSIS:")
    print(topsis)
    print("GROUP WSM:")
    print(wsm)

    print("TOPSIS Borda Count Method")
    MCDM.borda_count(topsis)
    print("WSM Borda Count Method")
    MCDM.borda_count(wsm)
    

if __name__ == '__main__':
    main()