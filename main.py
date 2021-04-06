from timeit import timeit

from scripts.dataLoader import dataLoader
from scripts.clustering import clustering

def main():
    path = './data/test'
    DataLoader = dataLoader(path)

    files = DataLoader.getAllFiles()
    print(files)
    for file in files:
        data = DataLoader.readFile(file)
        if file == 'breast-cancer-wisconsin.data':
            data = data.drop('id', axis=1)
            breastcancer = data['class']
            data = data.drop('class', axis=1)
        wynik = clustering.kSrednich(data, 2)
        rand_index = clustering.compute_rand_index(breastcancer, wynik)
        print(rand_index)

        fowlkes = clustering.compute_fowlkes_mallows(breastcancer, wynik)
        print(fowlkes)
        
        davies_bouldin_sco = clustering.compute_davies_bouldin(data, wynik)
        print(davies_bouldin_sco)
    
    

if __name__ == '__main__':
    main()