import pandas as pd
import random
from refactoring_methods import *
import csv
import shutil

class Augmentor:

    def __init__(self, file_name,smells : int|tuple|None ,step=None, path_updated_smellKey = None, only_refactored_file = None,path_test_element = None):
        self.refactors_list = [
                  return_optimal,
                  rename_api,
                  rename_local_variable,#???
                  rename_method_name,
                  enhance_if,
                  add_print,
                  duplication,
                  apply_plus_zero_math,
                  dead_branch_if_else,
                  dead_branch_if,
                  dead_branch_while,
                  dead_branch_for,
                  dead_branch_switch
                ]#
        self.file_name = file_name
        self.path_updated_smellKey= path_updated_smellKey if path_updated_smellKey is not None else 'new_data_set.csv'
        self.only_refactored_file = only_refactored_file if only_refactored_file is not None else 'refactored.csv'
        self.path_test_element = path_test_element if path_test_element is not None else 'test_elements.csv'
        self.step = step if step is not None else 500
        self.smells = smells if step is not None else None

    def get_smellKeys_amount_and_percent(self,path,custom_filter =[]):
        df = pd.read_csv(path)
        grouped = df.groupby("smellKey")
        if custom_filter ==[]:
            elements = [{"smellKey":group,"count": count} for group, count in zip(grouped.groups.keys(), grouped.size())]
        else:
            elements = [{"smellKey":group,"count": count} for group, count in zip(grouped.groups.keys(), grouped.size()) if group in custom_filter]
    
        #print(elements)
        sum = 0
        for element in elements:
            sum = element["count"]+ sum
        for element in elements:
            percent = (element["count"]/sum)*100
            element["percent"] = percent

        #print(elements)
        return elements
   
    def filter_smells(self,custom_filter):
        df = pd.read_csv(self.file_name)
        grouped = df.groupby("smellKey")
        elements = [{"smellKey":group,"count": count} for group, count in zip(grouped.groups.keys(), grouped.size())]
        if custom_filter is not None:
            elements = sorted(elements, key=lambda k: k['count'],reverse=True)
            new_elements =[]
            if type(custom_filter) is int:
                for index in range(custom_filter):
                    if (index < custom_filter and len(elements) > index):
                        new_elements.append(elements[index])

            elif type(custom_filter) is tuple:
                for index in range(len(elements)):
                    if (index >=custom_filter[0] and index <= custom_filter[1]
                        and (custom_filter[0] < len(elements) and custom_filter[1] < len(elements))
                        and (custom_filter[0] >0 and custom_filter[1] >0)):
                        new_elements.append(elements[index])         
            elements = new_elements

        elements = [element["smellKey"] for element in elements]
        return elements

    def smell_keys_with_less_percent(self,elements):

        max_percent = 100/len(elements)
        #print(max_percent)
        refactor_list =[]
        for element in elements:
            if(element["percent"] < max_percent):
                refactor_list.append(element["smellKey"])

        return refactor_list

    # dışarıdan bir for ile smeel keyler gezilecek bu fonksiyon çağırılacak
    def refactor_specified_smell_key_with_filter(self,specified_smeels,data,filtered_smells,deneme_obje,deneme_obje2,deneme_obje3,mycount):
        new_data=[]
        refactored_elements = []
        filtered_elements_without_augmentation = []
        counter = 0
        counter2=0
        for element in data:
            deneme_obje[element[1]] = deneme_obje[element[1]] + 1            
                #print(element[1], specified_smeels["smeel"])
            if element[1] in specified_smeels:
                deneme_obje2[element[1]] = deneme_obje2[element[1]] + 1
                new_data = new_data + self.refactor_input(element)
                refactored_elements.append(element)
                if("java:S1161" == element[1] ):
                    counter = counter +1
                #print(element[1],specified_smeel["smellKey"])
            if element[1] in filtered_smells and element[1] not in specified_smeels:
                deneme_obje3[element[1]] = deneme_obje3[element[1]] + 1
                filtered_elements_without_augmentation.append(element)
            if("java:S3776" == element[1] ):
                mycount["mycount"] = mycount["mycount"] +1
                counter2=counter2+1
        #print(f"deneme {counter}")
        #print(f"my count {mycount['mycount']}, counter :{counter2}")
        return new_data,refactored_elements,filtered_elements_without_augmentation
    
    def refactor_specified_smell_key(self,specified_smeels,data,deneme_obje,deneme_obje2):
        new_data=[]
        refactored_elements = []
        filtered_elements_without_augmentation = []
        counter = 0
        for element in data:
            deneme_obje[element[1]] = deneme_obje[element[1]] + 1            
            for specified_smeel  in specified_smeels:
                #print(element[1], specified_smeels["smeel"])

                if element[1] == specified_smeel["smellKey"]:
                    deneme_obje2[element[1]] = deneme_obje2[element[1]] + 1
                    new_data = new_data + self.refactor_input(element)
                    refactored_elements.append(element)
                    if("java:S1161" == element[1] ):
                        counter = counter +1
                    #print(element[1],specified_smeel["smellKey"])

            filtered_elements_without_augmentation.append(element)
        print(f"deneme {counter}")
        return new_data,refactored_elements,filtered_elements_without_augmentation
    def refactor_input(self,element):
        #function = refactors_list[0]
        #refactored_rows= [function(rows[i][j]) if j==0 else rows[i][j] for j in range(rows[0])for i in range(len(rows))]
        #Sadece bir refactor işlemi için oluşan liste
        new_list=[]
        for refactor_function in self.refactors_list:
            try:
                new_row_element= refactor_function(element[0])
                if(new_row_element == element[0] or new_row_element==""):
                    continue
                new_list_row = [new_row_element,element[1],element[2]]
                new_list.append(new_list_row)
            except Exception:
                self.log_refactor_error(element)
        return new_list

    def log_refactor_error(self,data):
        path="log_refactor_error.csv"
        check_file = os.path.isfile(path)
        
        if(check_file is False):
            with open(path, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["function","smellKey","smellId"])

        with open(path, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
            
    def read_csv_file(self,file_name,read_amount=50, start =-1):
        # read_amount'u tek seferde okunacak miktar olarak düşündüm.
        file = open(file_name,encoding="utf8")
        csvreader = csv.reader(file)

        header = []
        header = next(csvreader)
        rows = []
        
        indexer = 0
        counter = 0
        for row in csvreader:
            if indexer >= start:
                if read_amount == counter:
                    break
                else:
                    rows.append(row)
                counter = counter +1
            indexer = indexer + 1
            
        file.close()
        return header,rows

    def read_csv_file_as_bulk(self,file_name):
        # read_amount'u tek seferde okunacak miktar olarak düşündüm.
        try:
            file = open(file_name,encoding="UTF-8")
            csvreader = csv.reader(file)

            header = []
            header = next(csvreader)
            rows = []
            
            for row in csvreader:
                rows.append(row)
                
            file.close()
        except Exception:
            
            print(f"{file_name} is not existed")
            return [],[]
        return header,rows

    def write_csv_file(self,file_name,header,rows,mode):

        check_file = os.path.isfile(file_name)
        
        if(check_file is False):
            with open(self.file_name, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["function","smellKey","smellId"])

        with open(file_name, mode, encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(header)
            for row in rows:
                writer.writerow(row)

    def write_cvs_file_only_rows(self,file_name,rows,mode):

        check_file = os.path.isfile(file_name)
        
        if(check_file is False):
            print(f"{file_name} is not created. It will be created now.")
            with open(file_name, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["function","smellKey","smellId"])

        with open(file_name, mode, encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)


    def find_max(self,elements):
        max =0
        for element in elements:
            current = element["count"]
            if(max< current):
                max = current
        return max
    
    def discard_existed_refactored_smells(self,file_name, refactored_data):
        headers, rows = self.read_csv_file_as_bulk(file_name)

        new_data =[]
        control =True

        if(len(rows)==0):
            return refactored_data
        
        #print(rows)
        for data in refactored_data:
            control =True
            for element in rows:
                #print(element)
                #print(data)
                if(element[0] == data[0]):
                    control =False
                    break 
            if control:
                new_data.append(data)

        return new_data
    def run_program(self):
        shutil.copy2(self.file_name,self.path_updated_smellKey )

        step = self.step
        counter = step
        max_step =  pd.read_csv(self.file_name)
        max_step = max_step.shape[0]

        #print(max_step)

        headers, rows = self.read_csv_file(self.file_name,counter)
        custom_filter= []
        if self.smells is not None:
            custom_filter = self.filter_smells(self.smells)

        #print(custom_filter)
        smells = self.get_smellKeys_amount_and_percent(self.file_name,custom_filter)
        total_elements = [ element["count"] for element in smells]
        #print(total_elements)
        total_smells = sum(total_elements)
        train_amount = total_smells * 0.8
        total_filtered_element = 0

        #print(f"total element count : {total_smells} and the train amount : {train_amount}")
        
        deneme_obje ={}
        deneme_obje2={}
        deneme_obje3={}
        mycount={"mycount":0}
        df = pd.read_csv(self.file_name)
        grouped = df.groupby("smellKey")
        elementler = [{"smellKey":group,"count": count} for group, count in zip(grouped.groups.keys(), grouped.size())]
        #print(f"toplam smell: {len(elementler)}")
        for element in elementler:
            deneme_obje[element["smellKey"]] =0
            deneme_obje2[element["smellKey"]] =0
            deneme_obje3[element["smellKey"]] =0

        while len(smells) > 1 and len(rows) > 0:
            filtered_smeels = self.smell_keys_with_less_percent(smells)
            
            refactored_smeels_as_function, old_refactored_elements,not_refactored_elements = self.refactor_specified_smell_key_with_filter(filtered_smeels,rows,custom_filter,deneme_obje,deneme_obje2,deneme_obje3,mycount)
            
            refactored_smeels_as_function = self.discard_existed_refactored_smells(self.only_refactored_file,refactored_smeels_as_function)

            total_filtered_element = total_filtered_element + len(old_refactored_elements) + len(not_refactored_elements)
            #print(f"total_filtered_element: {total_filtered_element}")
            
            if(train_amount > total_filtered_element):

                #rows = rows + refactored_smeels_as_function

                self.write_cvs_file_only_rows(self.path_updated_smellKey,refactored_smeels_as_function,"a")
                
                old_refactored_elements_with_new_ones = old_refactored_elements + refactored_smeels_as_function + not_refactored_elements
                self.write_cvs_file_only_rows(self.only_refactored_file,old_refactored_elements_with_new_ones,"a")

                #self.write_cvs_file_only_rows(self.only_refactored_file,refactored_smeels_as_function,"a")
            else:
                old_refactored_elements_with_new_ones = old_refactored_elements + not_refactored_elements               
                self.write_cvs_file_only_rows(self.path_test_element,old_refactored_elements_with_new_ones,"a")

            smells = self.get_smellKeys_amount_and_percent(self.path_updated_smellKey,custom_filter)
            counter = counter+step
            headers, rows = self.read_csv_file(self.file_name,step,counter)        
            print(counter)
        #print(deneme_obje)
        #print(deneme_obje2)
        #print(deneme_obje3)
        #print(total_filtered_element-train_amount)
        #print(f"eleman : {mycount}")
#