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
np.set_printoptions(threshold=sys.maxsize)
import scipy.stats as st

auc_list=[]

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return [m,m-h,m+h]
    
def calculate_auc(x,y):
    s = 0
    for i in range(400-1):
        
        s = s + (-x[i+1]+x[i])*y[i]
        auc_list.append(s)
    return s

def f1(TP,FP,FN):
    return 2*TP/(2*TP + FP + FN)
    
def precision(TP,FP,FN,TN):
    try:
        return TP/(TP+FP)
    except:
        return 1
    
def recall(TP,FP,FN,TN):
    try:
        return TP/(TP+FN)
    except:
        return 0

def NPV(TP,FP,FN,TN):
    try:
        return TN/(TN+FN)
    except:
        return 0


    
#predictions = np.array([0,2])
dict_model={'densenet':44,
            'resnet':46,
            'alexnet':48,
            'vgg':50,
            'inception':52}
model1=str(input('Input first model as modelname-fold-FeatureExtractionMode(T=0 F=1): '))
model2=str(input('Input second model as modelname-fold-FeatureExtractionMode: '))

model1=model1.split(sep='-')
model2=model2.split(sep='-')
fold= model1[1]

model1[1] = ((int(model1[1])-1)*10)+dict_model[model1[0]]+int(model1[2])
model2[1] = ((int(model2[1])-1)*10)+dict_model[model2[0]]+int(model2[2])

path1_choices= glob.glob('/auto/data2/yturali/Runs/Run_{}/*/data/{}_*_output_*.npy'.format(model1[1],model1[0]))
path2_choices= glob.glob('/auto/data2/yturali/Runs/Run_{}/*/data/{}_*_output_*.npy'.format(model2[1],model2[0]))
path1_choices.sort(key = lambda x: x.split(sep='_')[8])
#for paths in path1_choices:
#  print(paths) 
#choice1=int(input('Input epoch number choice: '))
#path2_choices.sort(key = lambda x: x.split(sep='_')[8])
#for paths in path2_choices:
#  print(paths)
#choice2=int(input('Input epoch number choice: '))

choice1=path1_choices[-1]
choice2=path2_choices[-1]

result_dir = '/auto/data2/yturali/Runs/ROC'
time_ = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
current_res_dir = '{}/{}_{}_{}_ensemble_fold_{}'.format(result_dir,time_,model1[0],model2[0],fold)
os.mkdir(current_res_dir)

path1= glob.glob('/auto/data2/yturali/Runs/Run_{}/*/data/{}_{}_output_*.npy'.format(model1[1],model1[0],choice1))[0]
print('First Model:{}'.format(path1))
path2= glob.glob('/auto/data2/yturali/Runs/Run_{}/*/data/{}_{}_output_*.npy'.format(model2[1],model2[0],choice2))[0]
print('Second Model:{}'.format(path2))
file2 = np.load(path1)
file3 = np.load(path2)
file2 = (file2+file3)/2
class_0 = np.sort(file2[0,:].flatten())-0.01
class_1 = np.sort(file2[2,:].flatten())+0.01


sensitivity = np.zeros(400)
specifity = np.zeros(400)
recalls = np.zeros(400)
precisions = np.zeros(400)
NPVs = np.zeros(400)
f1s = np.zeros(400)
for threshold in range(-200,200):
    value = threshold/200
    index = threshold + 200
    TP = sum(i > value for i in class_1)
    TN = sum(i < value for i in class_0)
    FP = sum(i > value for i in class_0)    
    FN = sum(i < value for i in class_1)
    recalls[index] = recall(TP,FP,FN,TN)
    precisions[index] = precision(TP,FP,FN,TN)
    sensitivity[index] = TP/(TP+FN)
    specifity[index] = TN/(TN+FP)
    f1s[index] = f1(TP,FP,FN)
    NPVs[index]= NPV(TP,FP,FN,TN)

precisions = np.nan_to_num(precisions,nan=1)
sensitivity = np.nan_to_num(sensitivity,nan=1)
specifity =  np.nan_to_num(specifity,nan=1)
NPVs      =  np.nan_to_num(NPVs,nan=1)

PPV_rate  =  mean_confidence_interval(precisions)
sensitiv  =  mean_confidence_interval(sensitivity)
spec      =  mean_confidence_interval(specifity)
NPV_rate  =  mean_confidence_interval(NPVs)
auc=calculate_auc(1-specifity,sensitivity)
auc_arr = np.array(auc_list)
#print(auc_arr.shape)
auc_arr = np.nan_to_num(auc_arr,nan=1)
#print(auc_arr)
auc_mean  =  mean_confidence_interval(auc_arr)
#print(auc_mean)
PPV_rate_arr= np.array([PPV_rate])
sensitivity_arr= np.array([sensitiv])
specificity_arr= np.array([spec])
NPV_rate_arr= np.array([NPV_rate])
AUC_mean_arr=np.array([auc_mean])

PPV_rate_lis_c=[str(np.round_(PPV_rate[0]*100,decimals=1))+'%',str(np.round_(PPV_rate[1]*100,decimals=1))+'% - '+str(np.round_(PPV_rate[2]*100,decimals=1))+'%']
PPV_rate_arr_c= np.array([PPV_rate_lis_c])
print(PPV_rate_arr_c)
sensitivity_lis_c=[str(np.round_(sensitiv[0]*100,decimals=1))+'%',str(np.round_(sensitiv[1]*100,decimals=1))+'% - '+str(np.round_(sensitiv[2]*100,decimals=1))+'%']
sensitivity_arr_c= np.array([sensitivity_lis_c])
print(sensitivity_arr_c)
specificity_lis_c=[str(np.round_(spec[0]*100,decimals=1))+'%',str(np.round_(spec[1]*100,decimals=1))+'% - '+str(np.round_(spec[2]*100,decimals=1))+'%']
specificity_arr_c= np.array([specificity_lis_c])
print(specificity_arr_c)
NPV_rate_lis_c=[str(np.round_(NPV_rate[0]*100,decimals=1))+'%',str(np.round_(NPV_rate[1]*100,decimals=1))+'% - '+str(np.round_(NPV_rate[2]*100,decimals=1))+'%']
NPV_rate_arr_c= np.array([NPV_rate_lis_c])
print(NPV_rate_arr_c)
AUC_mean_lis_c=[str(np.round_(auc_mean[0]*100,decimals=1))+'%',str(np.round_(auc_mean[1]*100,decimals=1))+'% - '+str(np.round_(auc_mean[2]*100,decimals=1))+'%']
AUC_mean_arr_c= np.array([AUC_mean_lis_c])
print(AUC_mean_arr_c)

print(PPV_rate_arr)
print(sensitivity_arr)
print(specificity_arr)
print(NPV_rate_arr)
print(AUC_mean_arr)


a = np.arange(400)/400
calculate_auc(a,a)

plt.plot(a,a, linestyle='-')
plt.plot(a,f1s, linestyle='solid',label='F1s')
plt.plot(a,recalls, linestyle='solid',label='Recall')
plt.plot(a,precisions, linestyle='solid',label='precision')
plt.plot(1-specifity,sensitivity, linestyle='--',label='ROC')
plt.plot([], [], ' ', label=str('AUC={:.6f}'.format(auc)))

plt.xlabel('False Positive Rate (1- Specifity   FP/(TN+FP))')
plt.ylabel('True Positive Rate (Sensitivity TP/(TP+FN))')
# show the legend
plt.legend()
# show the plot
plt.savefig('{}/{}_{}_fold_{}_ensemble_roc.png'.format(current_res_dir,model1[0],model2[0],fold), bbox_inches='tight', dpi =160)
plt.grid()
plt.show()



plt.plot(recalls[2:],precisions[2:], linestyle='solid')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('{}/{}_{}_fold_{}_ensemble_precision_recall.png'.format(current_res_dir,model1[0],model2[0],fold), bbox_inches='tight', dpi =160)
plt.grid()
plt.show()


TP = sum(i > 0 for i in class_1)
TN = sum(i < 0 for i in class_0)
FP = sum(i > 0 for i in class_0)    
FN = sum(i < 0 for i in class_1)

cf_matrix = np.array([[TN,FP],[FN,TP]])
sn.heatmap(cf_matrix, annot=True)
plt.title( "Confusion Matrix" )
plt.savefig('{}/{}_{}_fold_{}_ensemble_Confusion_Matrix.png'.format(current_res_dir,model1[0],model2[0],fold), bbox_inches='tight', dpi =160)
plt.show()

fig, ax =plt.subplots(1,1)
column_labels = [model1[0]+'-'+model2[0]+' ensemble',' % Value','95% Confidence Interval (CI)']
row_list= ['AUC','Sensitivity','Spesificity','Positive Predictive Value','Negative Predictive Value']
rows= np.array([row_list])
rows =np.transpose(rows)
table_data=np.concatenate((PPV_rate_arr_c,specificity_arr_c,sensitivity_arr_c,PPV_rate_arr_c,NPV_rate_arr_c))
finished_data=np.concatenate((rows,table_data),axis=1)
data = finished_data
print(data)
print(data.shape)
dff = pd.DataFrame(data)
#column_labels = ['Mean AUC','STD']
ax.axis('tight')
ax.axis('off')

plt.title('{}-{} at Fold {} Ensemble Model'.format(model1[0],model2[0],fold))
tab1 = ax.table(cellText=data,colLabels=column_labels,loc="center")
tab1.auto_set_column_width(col=list(range(len(dff.columns))))
# Save the figure and show
plt.savefig('{}/{}_{}_fold_{}_ensemble_table.png'.format(current_res_dir,model1[0],model2[0],fold), bbox_inches='tight', dpi =160)
plt.show()

# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# fpr[0], tpr[0], _ = roc_curve(file2[predictions+1, 0:40], file2[predictions, 0:40])
# roc_auc[0] = auc(fpr[0], tpr[0])

# fpr["micro"], tpr["micro"], _ = roc_curve(file2[predictions+1, 0:40].ravel(), file2[predictions, 0:40].ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# plt.figure()
# lw = 2
# plt.plot(fpr[2], tpr[2], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()


