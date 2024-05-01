import random

import numpy as np
import pandas as pd
import csv
x=1000
x_train=0
x_test=0
select_train=[]
select_test=[]
data_train=[]
data_test=[]
def size_x():
    global x
    data = pd.read_csv("./datasets/winequality-white.csv", sep=";" , on_bad_lines='skip')
    x=len(data)-2
def num_rows(x):
    global x_train
    global x_test
    x_train=int(2/3*x)
    x_test=int(1/3*x)
    while (x_train+x_test)!=x:
        x_train=x_train+1

def select_rows(x):
    global select_train
    global select_test
    row=[]
    for i in range(0,x):
        row.append(i)
    i=-1

    while i< x_train:
        i=i+1
        rnd=random.randrange(len(row))
        #print(type(rnd),rnd)
        select_train.append(row[rnd])
        row.pop(rnd)
        print(len(row))
    select_test=row
    select_train.sort()

def generate_train():
    data = pd.read_csv("./datasets/winequality-white.csv", sep=";" ,  on_bad_lines='skip')
    global data_train
    global data_test
    fields=data.keys()
    print(fields)
    print(type(fields))
    data_train.append(data.keys())
    data_test.append(data.keys())

    for i in select_train:
        data_train.append(data.loc[i])
    for i in select_test:
        data_test.append(data.loc[i])

size_x()
print("size_x= ",x)
num_rows(x)
select_rows(x)
generate_train()
print(x_train,x_test,(x_train+x_test)==x)
print(select_train)
print(select_test)
print(type(data_train))
#print(data_train)

with open('./datasets/wine_dataset_train.csv', 'w',newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f,delimiter=';')
    write.writerows(data_train)
f.close()
with open('./datasets/wine_dataset_test.csv', 'w',newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f,delimiter=';')
    write.writerows(data_test)
f.close()