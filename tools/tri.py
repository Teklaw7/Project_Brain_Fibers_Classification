import pandas
import numpy as np


path ="/home/timtey/Documents/Projet/dataset4/tracts_filtered_train_test_label_to_number_nb_cells_without_missing.csv"

df = pandas.read_csv(path)
print(len(df))
# df.drop(df.columns[0])
for i in range(len(df)):
    df.loc[i][0] = i
    df.loc[i][1] = i
    df.loc[i][2] = i
print(df)
df.to_csv(path)