import os
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import lfilter
from pyexcel.cookbook import merge_all_to_a_book
import matplotlib.pyplot as plt
import glob

USER_INPUT = False

if(USER_INPUT):
    margin_sell = float(input("Margem para a venda (%): "))/100
    margin_buy = float(input("Margem para a compra (%): "))/100
    days = int(input("Dias: "))
else:
    margin_sell = 0.10
    margin_buy = 0.10
    days = 10

def labelPoints(indexes,allPoints):
    label = []
    for i in range(len(allPoints) - days):
        j = 1
        labeled = 0
        while(j < days):
            if((1 + margin_buy)*allPoints[i] < allPoints[i + j]):
                label.append(((indexes[i],allPoints[i]),'BUY'))
                labeled = 1
                break
            elif((1 - margin_sell)*allPoints[i] > allPoints[i + j]):
                label.append(((indexes[i],allPoints[i]),'SELL'))
                labeled = 1
                break
            else:
                pass
            j += 1
        
        if(not labeled):
            label.append(((indexes[i],allPoints[i]),'HOLD'))
               
    for i in range(len(allPoints) - days,len(allPoints)):
        label.append(((indexes[i],allPoints[i]),'HOLD'))
        
    return label

OUTDIR = "./raw_data"

if not os.path.exists(OUTDIR):
    os.mkdir(OUTDIR)

raw_data = pd.read_csv("Ibovespa_index_composition.csv")
stocks = raw_data["Código"].tolist()    

for stock in stocks[0:1]:
    historical_data = yf.Ticker(stock + ".SA").history(period="max")
    if not historical_data.empty:
        historical_data.to_csv(os.path.join(OUTDIR, stock + "_raw.csv"))
        historical_data.to_excel(os.path.join(OUTDIR, stock + "_raw.xlsx"))

        n = 15             # larger n gives smoother curves
        b = [1.0 / n] * n  # numerator coefficients
        a = 1              # denominator coefficient
        y_lf = lfilter(b, a, historical_data['Close'])

        label = labelPoints(historical_data.index,y_lf)
        
tup = [tpl[1] for tpl in label]
alldata =  pd.read_excel('raw_data/VALE3_raw.xlsx')
k = 0

plt.plot(historical_data.index, y_lf)
for i in range(len(label)-1):
    if(label[i][1] == 'BUY'):
        plt.scatter(label[i][0][0],label[i][0][1],color = 'hotpink')
    if(label[i][1] == 'SELL'):
        plt.scatter(label[i][0][0],label[i][0][1],color = 'blue')
plt.show()

def allData():
    return alldata

def get_input():
    return y_lf

def get_dataframe_input():
    return historical_data

def get_output():
    convert_dict = {'HOLD':0,'BUY':1,'SELL':2}
    return [convert_dict[i[1]] for i in label]

def days_number():
    return days

def csv2xlsx(csvfile):
    merge_all_to_a_book(glob.glob("raw_data/VALE3_raw.csv"), "raw_data/VALE3_raw.xlsx")
    
    