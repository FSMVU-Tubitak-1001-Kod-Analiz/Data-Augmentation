import numpy
import pandas as pd

class AugmentationUtils:
    def __init__(self,filename):
        self.frame = pd.read_csv(filename).groupby("smellKey").count().sort_values(by="function")
    def get_smells_count(self,selected_amount = -5,function_column_name = "function"):
        selected_amount = selected_amount * -1 if selected_amount > 0 else selected_amount
        index = self.frame.columns.get_loc(function_column_name)
        return_list = [element[index] for element in self.frame.values[selected_amount:]]
        return return_list
    def calculate_coefficient(self,smells_count,smells_increment,step=0,max_coef = 1):
        return_list = []
        for i in range(len(smells_count)):
            return_list.append((smells_increment[i] - smells_count[i]) / smells_count[i])
        return return_list

    def increase_smells_amount(self,smells_count,smells_increment,iter_step=0):
        return_list = []
        for i in range(len(smells_count)):
            return_list.append(smells_count[i] + (smells_increment[i] * iter_step))
        return return_list
    def calculate_smells_increment(self,smells_count,step,max_coef = 1,shift = 0):
        maximum = smells_count[-1:] * max_coef
        return_list = [(((shift + (maximum - element))  / step) )[0]
                       if type(((shift + (maximum - element)) / step) ) == numpy.ndarray
                       else ((shift+ ( maximum - element)) / step)
                       for element in smells_count]

        return [int(element) for element in return_list]
    def sort_coeffs_by_amount_of_smells(self,coeffs,smells_increments):
        smells_increment = smells_increments.copy()
        print(smells_increment)
        print(coeffs)
        for i in range(len(coeffs)-1):
            for j in range(len(coeffs)-1-i):
                if smells_increment[j] > smells_increment[j+1]:
                    temp = smells_increment[j]
                    smells_increment[j] = smells_increment[j+1]
                    smells_increment[j+1] = temp
                    temp = coeffs[j]
                    coeffs[j] = coeffs[j+1]
                    coeffs[j+1] = temp
        return coeffs
    def get_coeff_list(self,smells_count,smells_increment,step):
        return_list = []
        for x in range(step):
            temp_smells_increment = self.increase_smells_amount(smells_count, smells_increment, x)
            coeff = self.calculate_coefficient(smells_count, temp_smells_increment)
            # coeff = self.sort_coeffs_by_amount_of_smells(coeff,temp_smells_increment)
            return_list.append(coeff)
        return return_list

