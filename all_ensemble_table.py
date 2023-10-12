# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:26:38 2022

@author: Mehmet Yigit Turali
"""

from matplotlib import pyplot
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import sys
import glob
import os
import time
import datetime
import matplotlib
import matplotlib.pyplot as plt
import csv
  
x = []
y = []
  
with open("/auto/data2/yturali/Runs/ensemblelog.csv",'r') as csvfile:
    plots = csv.reader(csvfile, delimiter = '\t')
    for row in plots:
        x.append(row)
x_arr= np.array(x)
fig, ax =plt.subplots(1,1)
column_labels = x[0][:]
#column_labels=[s + ' (% value)' for s in column_labels]
data = np.array(x[1:][:])
data[:,1:] = np.round_((data[:,1:].astype(np.float)*100),decimals=2)
data = data[np.argsort(data[:, 1])]
data = data[::-1]
for i in range (12):
  for j in range (1,6):
    data[i,j] ='{:.2f}'.format(float(data[i,j]))+'%'
print(data)
print(data.shape)
dff = pd.DataFrame(data)
#column_labels = ['Mean AUC','STD']
ax.axis('tight')
ax.axis('off')

plt.title('Ensemble and Normal Model Analysis')
ccolors = plt.cm.BuPu(np.full(len(column_labels), 0.1))
fontsize = 10
tab1 = ax.table(cellText=data,colLabels=column_labels,loc="center",cellLoc='center',colColours=ccolors)
fontsize = 10
tab1.set_fontsize(fontsize)
tab1.scale(1.2, 1.2)
tab1.auto_set_column_width(col=list(range(len(dff.columns))))
# Save the figure and show
# Save the figure and show
plt.savefig('/auto/data2/yturali/Runs/Ensemble_Model_Analysis.png', bbox_inches='tight', dpi =160)
plt.show()
plt.close('all')