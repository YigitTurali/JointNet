# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 22:08:22 2021

@author: Mehmet Yigit Turali
"""
import glob
import numpy as np

name_list=['densenet_with_f','densenet_without_f','resnet_with_f','resnet_without_f','alexnet_with_f',
'alexnet_without_f','vgg_with_f','vgg_without_f','inception_with_f','inception_without_f']
auc_list=[]
epoch_list=[]
value_dict={}
matrix=[]
matrix_arr= np.array(matrix)
count=0
conc=0
for i in range (0,10):
    for j in range (44+i,135+i,10):
        for name in glob.glob('/auto/data2/yturali/Runs/Run_'+str(j)+'/*/data/*.npy'):
            if count == 100:
                break
            array = name.split("_")
            auc = ((array[8]).split("."))[1]
            auc_list.append(float("0."+str(auc)))
            count += 1
        error=auc_list[5]
        auc_list.pop(5)
        auc_list.insert(28,error)
        auc_arr = np.array(auc_list)
        if (conc==0):
            matrix_arr= auc_arr.reshape(-1,1)
            print(matrix_arr)
            print(np.shape(matrix_arr))
            conc +=1
        else:
            matrix_arr= np.concatenate((matrix_arr,auc_arr.reshape(-1,1)), axis=1)
#        print(auc_list)
        max_auc_index = np.argmax(auc_arr)
#        print(max_auc_index)
        auc_list=[]
        count=0
        epoch_list.append(max_auc_index)
matrix_arr=np.matrix(np.transpose(matrix_arr)) 
avg_matrix=matrix_arr.mean(0)
optimal_index= np.argmax(avg_matrix)
print(matrix_arr)
print(np.shape(matrix_arr))
print(avg_matrix)
print(np.shape(avg_matrix))
print('optimal epoch: ',optimal_index+1)
np.savetxt('matrix.txt', matrix_arr)
#epoch_arr= np.array(epoch_list)
#print(epoch_arr)
#print(np.mean(epoch_arr))
        
        
