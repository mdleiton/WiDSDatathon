import pandas
import csv
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import StratifiedKFold

data = pandas.read_csv("menor30.csv", decimal=".")
data_total = data.drop(columns=["patient_id","encounter_id", "readmission_status"])
data_training_total =data_total.drop(columns=["hospital_death"])
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(data_training_total, data_total["hospital_death"])
print(skf)
train_x  = pandas.DataFrame()
test_x = pandas.DataFrame()
train_y = pandas.DataFrame()
test_y =pandas.DataFrame()
contador = 0

for train_index, test_index in skf.split(data_training_total, data_total["hospital_death"]):
    train_x = data_total.iloc[train_index] 
    test_x = data_total.iloc[test_index]
    train_x.to_csv("train_"+ str(contador) + ".csv",sep=",", header=True, index=False)
    test_x.to_csv("test_"+ str(contador) + ".csv",sep=",", header=True, index=False)
    contador = contador + 1

