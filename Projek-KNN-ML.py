from tkinter import *
from tkinter import messagebox

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as pltNew
import pandas as pd
import seaborn as sn

from sklearn import preprocessing
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix

#Inisialisasi dataset
link = "dataset\\seattleWeather_1948-2017.csv"
dataset = pd.read_csv('dataset\\seattleWeather_1948-2017.csv')

#Preprocessing data
data = dataset
data = data.drop('DATE', axis=1)
data["RAINTOMORROW"] = None
for i in range(1,len(data)):
  data.loc[i, 'RAINTOMORROW'] = data.loc[i-1, 'RAIN']
data = data.dropna()

#Atribut
X = data.iloc[:, :-1].values
#Label
y = data.iloc[:, -1].values

#Pembagian data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#Inisialisasi model
knn = KNeighborsClassifier(n_neighbors=9)

# Inisialisasi K-Fold cross validator
kfold = KFold(n_splits=10, shuffle=True, random_state=0)

for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    lab = preprocessing.LabelEncoder()
    y_train = lab.fit_transform(y_train)
    y_test = lab.fit_transform(y_test)

    #Training model
    knn.fit(X_train, y_train)

def prediksi():
    
    
    curah_hujan = float(curah_hujan_entry.get())
    suhu_maks = int(suhu_maks_entry.get())
    suhu_min = int(suhu_min_entry.get())
    hujan = hujan_entry.get()
    atribut = [[curah_hujan, suhu_maks, suhu_min, hujan]]
    prediksi = knn.predict(atribut)

    if prediksi[0] == True:
       hasil_prediksi = "Besok akan hujan."
    elif prediksi[0] == False:
       hasil_prediksi = "Besok tidak akan hujan."
    else:
       hasil_prediksi = "-"

    messagebox.showinfo("Hasil Prediksi", hasil_prediksi)


root = Tk()
root.title("Prediksi Hujan")
root.geometry('450x200')

header = Label(root, text="Input Data untuk Prediksi Hujan", anchor="w", width=30, height=2, font="Arial, 10")
header.grid(row=0, column=0)

curah_hujan_label = Label(root, text="Curah Hujan Hari ini (inch)", anchor="w", width=30)
curah_hujan_label.grid(row=2, column=0)
curah_hujan_entry = Entry(root)
curah_hujan_entry.grid(row=2, column=1)

suhu_maks_label = Label(root, text="Suhu Maksimum Hari ini (Fahrenheit)", anchor="w", width=30)
suhu_maks_label.grid(row=3, column=0)
suhu_maks_entry = Entry(root)
suhu_maks_entry.grid(row=3, column=1)

suhu_min_label = Label(root, text="Suhu Minimum Hari ini (Fahrenheit)", anchor="w", width=30)
suhu_min_label.grid(row=4, column=0)
suhu_min_entry = Entry(root)
suhu_min_entry.grid(row=4, column=1)

hujan_label = Label(root, text="Apakah hari ini hujan ?", anchor="w", width=30)
hujan_label.grid(row=5, column=0)
hujan_entry = BooleanVar()
R1 = Radiobutton(root, text="Ya", variable=hujan_entry, value=True)
R1.grid(row=5, column=1)
R2 = Radiobutton(root, text="Tidak", variable=hujan_entry, value=False)
R2.grid(row=5, column=2)

predict_button = Button(root, text="Prediksi", command=prediksi)
predict_button.grid(row=7, column=1)

root.mainloop()