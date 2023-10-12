# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 02:45:04 2021

@author: PC
"""
import csv

headers = ('Name','AUC','Sensitivity','spesificity','PPV','NPV')
with open('/auto/data2/yturali/Runs/ensemblelog.csv','w', newline='',encoding='UTF8') as file:
    writer = csv.writer(file,delimiter='\t')
    writer.writerow(headers)