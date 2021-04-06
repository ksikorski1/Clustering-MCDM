from os import listdir
from os.path import isfile, join
import pandas as pd

class dataLoader():
    def __init__(self, path):
        self.path = path

    def getAllFiles(self):
        onlyfiles = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        return onlyfiles

    def readFile(self, file):    
        missing_values = ["n/a", "na", "--", "?", "NaN"]
        filepath = self.path + "/" + file
        df = pd.read_csv(filepath, na_values=missing_values)
        if (file == 'breast-cancer-wisconsin.data'):
            print("Breast cancer")
            df = dataLoader.breastCancer(df)
        df = df.dropna()
        return df

    @staticmethod
    def breastCancer(df):
        dataframe = df
        for index, row in enumerate(dataframe['class']):
            if row == 2:
                dataframe.at[index, 'class'] = 0
            if row == 4:
                dataframe.at[index, 'class'] = 1
        return dataframe
        