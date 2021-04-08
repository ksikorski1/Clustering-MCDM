from os import listdir
from os.path import isfile, join
import pandas as pd

from .preprocessing import Preprocessor

class DataLoader():
    def __init__(self, path):
        self.path = path

    def getAllFiles(self):
        onlyfiles = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        return onlyfiles

    def readFile(self, file):    
        missing_values = ["n/a", "na", "--", "?", "NaN"]
        filepath = self.path + "/" + file
        df = pd.read_csv(filepath, na_values=missing_values)
        return df
        