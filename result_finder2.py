# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import time
import datetime
import pandas as pd
name_list=['Densenet without Gender','Resnet without Gender','Densenet without Age','Resnet without Age','Densenet without age and gender','Resnet without age and gender']
time_ = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
auc_list=[]
epoch_list=[]
mean_auc_list=[]
std_list=[]
value_dict={}
count=0
start_run=int(input('Input First Run Number: '))
end_run=int(input('Input End Run Number: '))+1
sequence= int(input('Input Sequence Number: '))
analysis_name=str(input('Input Analysis Name: '))
value_dict={}

#for i in range (0,10):
#    for j in range (start_run+i,end_run+i,sequence):
#        #for name in glob.glob('/auto/data2/yturali/Runs/Run_'+str(j)+'/*/data/*_11_*.npy'):
#        for name in glob.glob('/auto/data2/yturali/Runs/Run_'+str(j)+'/*/data/*.npy'):
#            print(name)
#            array = name.split("_")
#            auc = ((array[8]).split("."))[1]
#            auc_list.append(float("0."+str(auc)))
#    auc_arr = np.array(auc_list)
#    print(auc_list)
#    print()
#    mean_auc = np.mean(auc_arr)
#    auc_std = np.std(auc_arr)
#    dict_list= ["Mean AUC:"+str(mean_auc),"Standard Deviation:"+str(auc_std)]
#    mean_auc_list.append(mean_auc)
#    std_list.append(auc_std)
#    auc_list=[]     


for i in range (0,50,10):
  for j in range (start_run+i,end_run+i,sequence):
    start_run_chk=start_run+i
    end_run_chk=end_run+i
    if (start_run_chk) ==294:
      print('********!!Warning!!*********')
      (start_run_chk) +=1
      (end_run_chk) +=1
      start_run = start_run_chk-i
      end_run = end_run_chk-i
      j+=1
      print(j)
    #for name in glob.glob('/auto/data2/yturali/Runs/Run_'+str(j)+'/*/data/*_11_*.npy'):
    for name in glob.glob('/auto/data2/yturali/Runs/Run_'+str(j)+'/*/data/*.npy'):
      print(name)
      array = name.split("_")
      auc = ((array[8]).split("."))[1]
      auc_list.append(float("0."+str(auc)))
    auc_arr = np.array(auc_list)
    print(auc_list)
    print()
    mean_auc = np.mean(auc_arr)
    auc_std = np.std(auc_arr)
    dict_list= ["Mean AUC:"+str(mean_auc),"Standard Deviation:"+str(auc_std)]
    mean_auc_list.append(mean_auc)
    std_list.append(auc_std)
    auc_list=[]
    #value_dict[name_list[i]]=dict_list
mean_auc_arr=np.array([mean_auc_list])
std_arr= np.array([std_list])
print(auc_std)
optimal_choice = np.argmax(mean_auc_arr)
#for i in range (0,sequence):
#  print(str(name_list[i]) + ":\n")
#  for value in value_dict[name_list[i]]:
#    print(value)
#  print("\n")
print()
print("Optimal Pretrained Model and its Mean AUC, STD are {} with values respectively:\n {}".format(name_list[optimal_choice],value_dict[name_list[optimal_choice]]))
fig, ax =plt.subplots(1,1)
decay_s=np.transpose(np.array([range(1, 11, 1)]))
print(decay_s.shape)
print(decay_s)
mean_auc_T= np.transpose(mean_auc_arr)
auc_std_T=np.transpose(std_arr)
#table_data=np.round_(np.concatenate((mean_auc_T,auc_std_T),axis=1),decimals=10)
raw_data=np.concatenate((mean_auc_T,auc_std_T),axis=1)
#raw_data[4],raw_data[10]=raw_data[10],raw_data[4]
table_data=np.round_(raw_data*100,decimals=2)
table_data_list=[]
print(table_data.shape)
print(table_data)
for i in range (10):
  table_data_list.append('{:.2f}'.format(table_data[i,0])+'% +-'+'{:.2f}'.format(table_data[i,1])+'%')
data = np.transpose(np.array([table_data_list]))
print(data)
data = np.concatenate((decay_s,data),axis=1)
print(data)
dff = pd.DataFrame(data)
column_labels = ['Mean AUC','STD']
#column_labels = ['Decay Size','Mean AUC with STD']
rows = name_list
ax.axis('tight')
ax.axis('off')
plt.title(analysis_name)
ccolors = plt.cm.BuPu(np.full(len(column_labels), 0.1))
fontsize = 10
tab1 = ax.table(cellText=data,colLabels=column_labels,loc="center",cellLoc='center',colColours=ccolors)
fontsize = 10
tab1.set_fontsize(fontsize)
tab1.scale(1.5, 1.5)
#tab1 = ax.table(cellText=data,colLabels=column_labels,loc="center")
tab1.auto_set_column_width(col=list(range(len(dff.columns))))
# Save the figure and show
plt.savefig('/auto/data2/yturali/Runs/Results/Date:{}_Run_{}_to_{}_Mean_AUC_with_STD_Table.png'.format(time_,start_run,end_run), bbox_inches='tight', dpi =160)
plt.show()



