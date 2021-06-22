from pandas.core import indexing
from scripts.dataLoader import DataLoader
import pandas
import numpy

def main():
    path = './results/'
    data_loader = DataLoader(path)
    files = data_loader.getAllFiles()

    for file in files:
        data = data_loader.readFile(file)
        data = data.round(4)

        data.to_csv(file, index=False)
    

if __name__ == "__main__":
    main()