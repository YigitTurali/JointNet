from matplotlib import pyplot
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
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
import itertools
import csv


all_data=[]
auc_list=[]
class_0_all=[]
class_1_all=[]
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
#dict_model={'densenet_wo_g':283,
#            'resnet_wo_g':294,
#            'densenet_wo_a':304,
#            'resnet_wo_a':314,
#            'densenet_wo_e':324,
#            'resnet_wo_e':334}
dict_model={'densenet':44,
            'resnet':46,
            'alexnet':48,
            'vgg':50,
            'inception':52}
model1_stable=str(input('Input first model as modelname-fold-FeatureExtractionMode(T=0 F=1): '))
model2_stable=str(input('Input second model as modelname-fold-FeatureExtractionMode: '))
p1=model1_stable.split(sep='-')
p2=model2_stable.split(sep='-')
result_dir = '/auto/data2/yturali/Runs/ROC'
time_ = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
current_res_dir = '{}/{}_{}_{}_Feature_Extraction_Mode_{}_ensemble'.format(result_dir,time_,p1[0],p2[0],p1[2])
os.mkdir(current_res_dir)

for i in range (1,11):
    model1=model1_stable.lower()
    model2=model2_stable.lower()
    model1=model1.split(sep='-')
    model2=model2.split(sep='-')
    if model1[2]=='1':
      fe='w/o FE'
    else:
      fe='w FE'
    #model1[1] = ((int(i)-1))+dict_model[model1[0]]+int(model1[2])
    model1[1] = ((int(i)-1)*10)+dict_model[model1[0]]+int(model1[2])
    print(model1[1])
    print((model1[0].split('_'))[0])
    #model2[1] = ((int(i)-1))+dict_model[model2[0]]+int(model2[2])
    model2[1] = ((int(i)-1)*10)+dict_model[model2[0]]+int(model2[2])
    print(model2[1])
    print((model2[0].split('_'))[0])
    path1_choices= glob.glob('/auto/data2/yturali/Runs/Run_{}/*/data/{}_*_output_*.npy'.format(model1[1],(model1[0].split('_'))[0]))
    path2_choices= glob.glob('/auto/data2/yturali/Runs/Run_{}/*/data/{}_*_output_*.npy'.format(model2[1],(model2[0].split('_'))[0]))
    path1_choices.sort(key = lambda x: x.split(sep='_')[8])
    print('here')
    for paths in path1_choices:
        print(paths) 
    print('here2')
#choice1=int(input('Input epoch number choice: '))
    path2_choices.sort(key = lambda x: x.split(sep='_')[8])
    for paths in path2_choices:
        print(paths)
#choice2=int(input('Input epoch number choice: '))
    path1_chosen=path1_choices[len(path1_choices)-1]
    path2_chosen=path2_choices[len(path2_choices)-1]
    choice1=(path1_chosen.split('_'))[6]
    choice2= (path2_chosen.split('_'))[6]

    path1= glob.glob('/auto/data2/yturali/Runs/Run_{}/*/data/{}_{}_output_*.npy'.format(model1[1],(model1[0].split('_'))[0],choice1))[0]
    print('First Model:{}'.format(path1))
    path2= glob.glob('/auto/data2/yturali/Runs/Run_{}/*/data/{}_{}_output_*.npy'.format(model2[1],(model2[0].split('_'))[0],choice2))[0]
    print('Second Model:{}'.format(path2))
    file2 = np.load(path1)
    file3 = np.load(path2)
    file2 = (file2+file3)/2
    class_0 = np.sort(file2[0,:].flatten())-0.01
    class_0_all.append(class_0)
    class_1 = np.sort(file2[2,:].flatten())+0.01
    class_1_all.append(class_1)
    
    y = np.concatenate((file2[1],file2[3])).flatten()
    pred = np.concatenate((file2[0]-0.01,file2[2]+0.01)).flatten()
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
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
    auc=metrics.auc(fpr, tpr)
    print(auc)
    TP = sum(i > 0 for i in class_1)
    TN = sum(i < 0 for i in class_0)
    FP = sum(i > 0 for i in class_0)    
    FN = sum(i < 0 for i in class_1)
    print(TP,FP,FN,TN)
    
    PPV_rate_mean =  precision(TP,FP,FN,TN)
    sensitiv_mean =  TP/(TP+FN)
    spec_mean     =  TN/(TN+FP)
    NPV_rate_mean =  NPV(TP,FP,FN,TN)
    
    all_data.append([auc,spec_mean,sensitiv_mean,PPV_rate_mean,NPV_rate_mean])
    
    if i == 1:
      file2all = file2
      file3all = file3
    else:
      file2all = np.concatenate((file2all,file2),axis=1)
      file3all = np.concatenate((file3all,file3),axis=1)
filee = (file2all+file3all)/2
print(filee)
class_0 = np.sort(filee[0,:].flatten())-0.01
class_0_all.append(class_0)
class_1 = np.sort(filee[2,:].flatten())+0.01
class_1_all.append(class_1)

y = np.concatenate((filee[1],filee[3])).flatten()
pred = np.concatenate((filee[0]-0.01,filee[2]+0.01)).flatten()
print(pred)
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
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
auc=metrics.auc(fpr, tpr)
print(auc)
TP = sum(i > 0 for i in class_1)
TN = sum(i < 0 for i in class_0)
FP = sum(i > 0 for i in class_0)    
FN = sum(i < 0 for i in class_1)
print(TP,FP,FN,TN)

PPV_rate_mean =  precision(TP,FP,FN,TN)
sensitiv_mean =  TP/(TP+FN)
spec_mean     =  TN/(TN+FP)
NPV_rate_mean =  NPV(TP,FP,FN,TN)
#print(i,'fold completed!')

a = np.arange(400)/400
calculate_auc(a,a)
fig1 = plt.figure()
plt.suptitle('{} {} Model Curves'.format(p1[0],fe))
#plt.suptitle('{}-{} {} Ensemble Model Curves'.format(p1[0],p2[0],fe))
fig1.add_subplot(2, 2, 1)
plt.title("F1 Score")
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.xlabel('Threshold')
plt.ylabel('F1')
plt.plot(a,a, linestyle='--')
plt.plot(a,f1s, linestyle='solid',label='F1s')
plt.legend(prop={"size":8})

# create a confidence band of +/- 5% error
y_lower_f1 = [i - 0.05 * i for i in f1s]
y_upper_f1 = [i + 0.05 * i for i in f1s]

# plot our confidence band
plt.fill_between(a, y_lower_f1, y_upper_f1, alpha=0.2, color='tab:orange')

fig1.add_subplot(2, 2, 2)
plt.title("Recall")
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.xlabel('Threshold')
plt.ylabel('Recall')
plt.plot(a,a, linestyle='--')
plt.plot(a,recalls, linestyle='solid',label='Recall')
plt.legend(prop={"size":8})

# create a confidence band of +/- 5% error
y_lower_recall = [i - 0.05 * i for i in recalls]
y_upper_recall = [i + 0.05 * i for i in recalls]

# plot our confidence band
plt.fill_between(a, y_lower_recall, y_upper_recall, alpha=0.2, color='tab:orange',label = "5% CI")

fig1.add_subplot(2, 2, 3)
#plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
#plt.title("Precision")
#plt.xlabel('Threshold')
#plt.ylabel('Precision')
#plt.plot(a,a, linestyle='--')
#plt.plot(a,precisions, linestyle='solid',label='Precision')
#plt.legend(prop={"size":8})
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.plot(a,a, linestyle='--')
plt.plot(recalls[:],precisions[:],linestyle='-',label='Precision vs. Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(prop={"size":8})
# create a confidence band of +/- 5% error
y_lower_precisions = [i - 0.05 * i for i in precisions[:]]
y_upper_precisions = [i + 0.05 * i for i in precisions[:]]

# plot our confidence band
plt.fill_between(recalls[:], y_lower_precisions, y_upper_precisions, alpha=0.2, color='tab:orange',label = "5% CI")

fig1.tight_layout()



fig1.add_subplot(2, 2, 4)
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.title("ROC Curve")
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(a,a, linestyle='--')
pyplot.plot(fpr, tpr, linestyle='-',label='ROC')
plt.plot([], [], ' ', label=str('AUC={:.6f}'.format(auc)))
plt.legend(prop={"size":8})
# create a confidence band of +/- 5% error
y_lower_tpr = [i - 0.05 * i for i in  tpr]
y_upper_tpr = [i + 0.05 * i for i in tpr]

# plot our confidence band
plt.fill_between(fpr, y_lower_tpr, y_upper_tpr, alpha=0.2, color='tab:orange',label = "5% CI")
fig1.tight_layout()
#plt.title('{}-{} {} Ensemble Precision-Recall Curve'.format(p1[0],p2[0],fe))
plt.savefig('{}/{}_{}_ensemble_precision_recall.png'.format(current_res_dir,model1[0],model2[0]), bbox_inches='tight', dpi =160)
# show the plot
fig1.savefig('{}/{}_{}_ensemble_roc.png'.format(current_res_dir,model1[0],model2[0]), bbox_inches='tight', dpi =160)



fig2 = plt.figure()
plt.plot(recalls[2:],precisions[2:], linestyle='solid')
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('{} {} Precision-Recall Curve'.format(p1[0],fe))
#plt.title('{}-{} {} Ensemble Precision-Recall Curve'.format(p1[0],p2[0],fe))
# create a confidence band of +/- 5% error
y_lower_p = [i - 0.05 * i for i in  precisions[2:]]
y_upper_p = [i + 0.05 * i for i in precisions[2:]]

# plot our confidence band
plt.fill_between(recalls[2:], y_lower_p, y_upper_p, alpha=0.2, color='tab:blue',label = "5% CI")
plt.savefig('{}/{}_{}_ensemble_precision_recall.png'.format(current_res_dir,model1[0],model2[0]), bbox_inches='tight', dpi =160)

fig3 = plt.figure()
plt.plot(np.linspace(0,1,num = 400),sensitivity, linestyle='solid',label="Sensitivity")
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
# create a confidence band of +/- 5% error
y_lower_sensitivity = [i - 0.05 * i for i in  sensitivity]
y_upper_sensitivity = [i + 0.05 * i for i in sensitivity]
# plot our confidence band
plt.fill_between(np.linspace(0,1,num =400), y_lower_sensitivity, y_upper_sensitivity, alpha=0.2, color='tab:green',label = "5% CI")
plt.plot(np.linspace(0,1,num =400),specifity, linestyle='solid',label="Specificity")
plt.grid(color = 'orange', linestyle = '-', linewidth = 0.5)
# create a confidence band of +/- 5% error
y_lower_specifity = [i - 0.05 * i for i in  specifity]
y_upper_specifity = [i + 0.05 * i for i in specifity]
# plot our confidence band
plt.fill_between(np.linspace(0,1,num = 400), y_lower_specifity, y_upper_specifity, alpha=0.2, color='tab:orange',label = "5% CI")
plt.legend(["Sensitivity","Specificity"])
plt.grid()
plt.xlabel('Criterion Value')
plt.ylabel('Sensitivity/Specificity')
plt.title('{} {} Sensitivity-Specificity Curve'.format(p1[0],fe))
#plt.title('{}-{} {} Ensemble Precision-Recall Curve'.format(p1[0],p2[0],fe))
plt.savefig('{}/{}_{}_ensemble_Sensitivity_Specifity.png'.format(current_res_dir,model1[0],model2[0]), bbox_inches='tight', dpi =160)

fig4 = plt.figure()
cf_matrix = np.array([[TP,FP],[FN,TN]])/10
sn.heatmap(cf_matrix, annot=True)
plt.title( "{} {} Mean Confusion Matrix".format(p1[0],fe))
#plt.title( "{}-{} {} Ensemble Mean Confusion Matrix".format(p1[0],p2[0],fe))
plt.savefig('{}/{}_{}_ensemble_Confusion_Matrix.png'.format(current_res_dir,model1[0],model2[0]), bbox_inches='tight', dpi =160)


fig6 = plt.figure()
colors = ['blue', 'orange']
bins = np.linspace(0.0, 1.0, 20)
plt.hist([(class_0+1)/2,(class_1+1)/2],bins,
         histtype ='bar',
         color = colors,
         label = ['Healthy', 'Active Inflammation'])
plt.axvline(x = 0.6, color = 'red',linestyle = '--',label = 'Threshold')
plt.xticks(np.linspace(0,1,21),rotation='vertical')
plt.legend(prop ={'size': 10})
plt.title( " {} {} Histogram of Outputs".format(p1[0],fe))
#plt.title( " {}-{} {} Ensemble Histogram of Outputs".format(p1[0],p2[0],fe))


#    plt.hist(class_0)
#    plt.hist(class_1)
plt.savefig('{}/{}_{}_Ensemble_Histogram.png'.format(current_res_dir,model1[0],model2[0]), bbox_inches='tight', dpi =160)
    
    
print(all_data)
all_data=np.array(all_data)
print(all_data.shape)
PPV_rate  =  mean_confidence_interval(list(all_data[:,3]))
sensitiv  =  mean_confidence_interval(list(all_data[:,2]))
spec      =  mean_confidence_interval(list(all_data[:,1]))
NPV_rate  =  mean_confidence_interval(list(all_data[:,4]))
auc_mean  =  mean_confidence_interval(list(all_data[:,0]))
#print(auc_arr.shape)
#print(auc_arr)
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
print("HERE")
print(sensitivity_arr)
print("HERE")
print(specificity_arr)
print(NPV_rate_arr)
print(AUC_mean_arr)


fig5 = plt.figure()
#prevalence = np.linspace(0.005,0.50,494)
prevalence = np.linspace(0.005,0.05,44)
def PPV_func(prevalence,sensitiv,spec):
  TP_n = 160*prevalence*sensitiv
  TN_n = 160*(1-prevalence)*spec
  FN_n = 160*prevalence*(1-sensitiv)
  FP_n = 160*(1-prevalence)*(1-spec)
  PPV_n = TP_n/(TP_n+FP_n)
  return PPV_n
  
def NPV_func(prevalence,sensitiv,spec):
  TP_n = 160*prevalence*sensitiv
  TN_n = 160*(1-prevalence)*spec
  FN_n = 160*prevalence*(1-sensitiv)
  FP_n = 160*(1-prevalence)*(1-spec)
  NPV_n = TN_n/(TN_n+FN_n)
  return NPV_n
  
plt.plot(prevalence,PPV_func(prevalence,sensitiv[0],spec[0]),linestyle='solid',label="PPV")
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
# plot our confidence band
plt.fill_between(prevalence,PPV_func(prevalence,sensitiv[1],spec[1]), PPV_func(prevalence,sensitiv[2],spec[2]), alpha=0.2, color='tab:green',label = "5% CI")

plt.plot(prevalence,NPV_func(prevalence,sensitiv[0],spec[0]), linestyle='solid',label="NPV")
plt.grid(color = 'orange', linestyle = '-', linewidth = 0.5)
# plot our confidence band
plt.fill_between(prevalence, NPV_func(prevalence,sensitiv[1],spec[1]), NPV_func(prevalence,sensitiv[2],spec[2]), alpha=0.2, color='tab:orange',label = "5% CI")
plt.legend(["PPV","NPV"])
plt.grid()
plt.xlabel('Prevalence')
plt.ylabel('PPV/NPV')
plt.title('{} {} PPV/NPV vs. Prevalence Curve'.format(p1[0],fe))
#plt.title('{}-{} {} Ensemble Precision-Recall Curve'.format(p1[0],p2[0],fe))
plt.savefig('{}/{}_{}_ensemble_PPV_NPV_prevalence.png'.format(current_res_dir,model1[0],model2[0]), bbox_inches='tight', dpi =160)


fig, ax =plt.subplots(1,1)
column_labels = ['Parameters' ,' % Value','95% Confidence Interval (CI)']
row_list= ['AUC','Sensitivity','Spesificity','Positive Predictive Value','Negative Predictive Value']
rows= np.array([row_list])
rows =np.transpose(rows)
table_data=np.concatenate((AUC_mean_arr_c,specificity_arr_c,sensitivity_arr_c,PPV_rate_arr_c,NPV_rate_arr_c))

row_list =[p1[0]+' '+ fe ,auc_mean[0],spec[0],sensitiv[0],PPV_rate[0],NPV_rate[0]]
#row_list =[p1[0]+'-'+p2[0]+' Ensemble '+ fe ,auc_mean[0],spec[0],sensitiv[0],PPV_rate[0],NPV_rate[0]]
with open('/auto/data2/yturali/Runs/ensemblelog.csv','a', newline='',encoding='UTF8') as file:
    writer = csv.writer(file,delimiter='\t')
    writer.writerow(row_list)
finished_data=np.concatenate((rows,table_data),axis=1)
data = finished_data
print(data)
print(data.shape)
print(precision)
dff = pd.DataFrame(data)
#column_labels = ['Mean AUC','STD']
ax.axis('tight')
ax.axis('off')

plt.title('{} {} Model'.format(p1[0],fe))
#plt.title('{}-{} Ensemble {} Model'.format(p1[0],p2[0],fe))
ccolors = plt.cm.BuPu(np.full(len(column_labels), 0.1))
fontsize = 10
tab1 = ax.table(cellText=data,colLabels=column_labels,loc="center",cellLoc='center',colColours=ccolors)
fontsize = 10
tab1.set_fontsize(fontsize)
tab1.scale(1.5, 1.5)
tab1.auto_set_column_width(col=list(range(len(dff.columns))))
# Save the figure and show
plt.savefig('{}/{}_{}_fold_{}_ensemble_table.png'.format(current_res_dir,model1[0],model2[0],'mean'), bbox_inches='tight', dpi =160)
plt.clf()
plt.close('all')
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


