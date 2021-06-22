import numpy as np
import mcdm
import operator
from pymcdm.methods import COPRAS
from pymcdm import weights as mcdm_weights
from pymcdm.methods.topsis import TOPSIS
from pymcdm.helpers import rrankdata

class MCDM():

    @staticmethod
    def calculate_topsis(array, alternative_names=None, benefit_list=None):
        
        mcdm_rank = mcdm.rank(array, alt_names=alternative_names, is_benefit_x=benefit_list, s_method="TOPSIS")
        return mcdm_rank
    
    @staticmethod
    def calculate_wsm(array, alternative_names):
        mcdm_rank = mcdm.rank(array, alt_names=alternative_names, s_method="SAW")
        
        return mcdm_rank
    
    @staticmethod
    def group_topsis(arrays, alternative_names=None, benefit_list=None):
        topsis = 0
        result = {}
        for dataset in arrays:
            topsis = MCDM.calculate_topsis(dataset, alternative_names, benefit_list)
            print(topsis)
            for alternative in topsis:
                try:
                    result[alternative[0]] += alternative[1]
                    print(result[alternative[0]], alternative[1])
                except KeyError:
                    result[alternative[0]] = alternative[1]
        number_of_summation = len(arrays)
        #print(number_of_summation)
        for key in result.keys():
            result[key] = result[key]/number_of_summation
        return result
    
    @staticmethod
    def group_wsm(arrays, alternative_names):
        wsm = 0
        result = {}
        for dataset in arrays:
            wsm = MCDM.calculate_wsm(dataset, alternative_names)
            #print(wsm)
            for alternative in wsm:
                try:
                    result[alternative[0]] += alternative[1]
                except KeyError:
                    result[alternative[0]] = alternative[1]
        number_of_summation = len(arrays)
        print("ile zestawow", number_of_summation)
        for key in result.keys():
            result[key] = result[key] / number_of_summation
        return result

    @staticmethod    
    def reverse_results(arrays):
        arr = arrays
        for array in arr:
            for method in array:
                for value in method:
                    value = 1 - value
        return arr 
    
    @staticmethod
    def group_copras(arrays, alternative_names, types):
        copras = COPRAS()
        np.seterr(divide='ignore', invalid='ignore')
        weights = mcdm_weights.equal(np.array(arrays[0]))
        result = {}
        for dataset in arrays:
            data = np.array(dataset)
            pref = copras(data, weights, types)
            for i in range(len(pref)):
                try:   
                    result[alternative_names[i]] += pref[i]
                except KeyError:
                    result[alternative_names[i]] = pref[i]        
        number_of_summation = len(arrays)
        for key in result.keys():
            result[key] = result[key] / number_of_summation
        
        return result

    @staticmethod
    def borda_count(dictionary):
        result = {}
        max_pts = len(dictionary.keys())
        sorted_dict = sorted(dictionary, key=dictionary.get, reverse=True)
        for i, key in enumerate(sorted_dict):
            result[key] = max_pts - i

        print(result)