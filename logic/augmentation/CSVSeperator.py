import pandas as pd
import numpy as np
import random
class CSVSeperator:
    def __init__(self,file_name):
        self.file_name = file_name

    #should be between 0 and 1
    def split_csv_as_percentage(self,percentage,train_file_name="train.csv",test_file_name="test.csv",shuffle = False):
        if not (percentage >= 0 and percentage <= 1):
            print("parameter should be between 0 and 1")

        file = pd.read_csv(self.file_name)
        size = file.shape[0]
        seperator = int(size * percentage)
        if shuffle:
            file = file.sample(frac=1).reset_index(drop=True)

        df = file[0:seperator]
        df2= file[seperator:]
        print(df)
        print(df2)

        df.to_csv(test_file_name, sep=',', index=False, encoding='utf-8')
        df2.to_csv(train_file_name, sep=',', index=False, encoding='utf-8')

