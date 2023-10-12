# command for running
# cd Pytorch/
# source Pytorch/pytorch_torchvision/bin/activate  && cd Pytorch/dataset_y/ &&  ml PyTorch/1.2.0-fosscuda-2019a-Python-3.7.2
# python3 ensemble_gender_age_cv.py --gpu "3" --bs 20 --lr 0.0005 --decay 9 --fold 4
#27. normalizasyonsuz

#32 33 resnet 18
#35 resnet 34 patladi
#36 normalizing var resnet 18

import os
import argparse
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import datetime
import csv

run=int(input('Input Run Number:'))
epochs_num= int(input('Input Number of Epochs:'))
model_name=str(input('Input The Name of Pretrained Model(densenet,resnet,alexnet,vgg,squeezenet,inception):'))
pretraining_input=input("Input Pretraining Mode:")
feature_extract_input = input("Input feature extraction mode: \n(When False, we finetune the whole model,when True we only update the reshaped layer params): ")


def pretraining_mode(pretraining_input):
    if pretraining_input.lower() == 'true':
        return True
    else:
        return False
    
    
def feature_extract_mode(feature_extract_input):
    if feature_extract_input.lower() == 'true':
        return True
    else:
        return False
    
    
def input_size_interpreter(model_name):
    if model_name != "inception":
        input_size=224
    else:
        input_size=299
    return input_size

feature_extract = feature_extract_mode(feature_extract_input)   
input_size = input_size_interpreter(model_name)
pretraining =pretraining_mode(pretraining_input)
# if __name__ == '__main__' :
class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index), self.imgs[index]#return image path
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--gpu', default ='0', type=str, help='gpu')
parser.add_argument('--lr', default=0.0005, type=float, help='learning_rate')
parser.add_argument('--bs', default=20, type=int, help='batchsize')
parser.add_argument('--decay', default = 5, type=int, help='decay')
parser.add_argument('--fold', default=4, type=int, help='fold')
args = parser.parse_args()
gpu_ids = args.gpu
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_ids;  




fold_count = args.fold
batch_size = args.bs
learning_rate = args.lr
decay_step_size = args.decay

experiment = "+-"

plt.ioff()   # interactive mode

data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(input_size),
        transforms.RandomAffine(20, translate=None, scale=(0.8,1.2), shear=None, resample=False, fillcolor=0),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
      #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    'test': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(input_size),
#        transforms.RandomAffine(20, translate=None, scale=(0.8,1.2), shear=None, resample=False, fillcolor=0),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
}

test_accuracy = []
train_accuracy = []   
test_loss = []
train_loss = []

# data_dir = 'C:/Users/ahmet/Desktop/Belge/umram/summary/aktif/'
data_dir = '/auto/data2/yturali/new_data/training_directory3/fold_{}/'.format(fold_count)
result_dir = '/auto/data2/yturali/Runs/Run_New_{}/'.format(run)
time_ = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
#os.mkdir('{}/{}_{}_{}_{}_{}'.format(data_dir,experiment,time_,batch_size,learning_rate,decay_step_size))
os.mkdir('{}/{}_{}_{}_{}_{}'.format(result_dir,experiment,time_,batch_size,learning_rate,decay_step_size))
#current_data_dir = '{}/{}_{}_{}_{}_{}/data'.format(data_dir,experiment,time_,batch_size,learning_rate,decay_step_size)
current_result_dir = '{}/{}_{}_{}_{}_{}/data'.format(result_dir,experiment,time_,batch_size,learning_rate,decay_step_size)
os.mkdir(current_result_dir)
            
image_datasets = {x: MyImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'test']}

batch_sizes =	{
  "train": batch_size,
  "test": 160
}
shuf =	{
  "train": True,
  "test": False
}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x], shuffle= shuf[x], num_workers=0, pin_memory=False)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
print(dataset_sizes['train'],dataset_sizes['test'])
class_names = image_datasets['train'].classes

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def process_all_image_names(paths):
    #print('number of paths', len(paths))
    gender_age = np.zeros((len(paths),2))
    for i,path in enumerate(paths):
        image_name = paths[i].split('/')[-1]
        if '_F_' in image_name:
            gender_age[i,0] = 0
        elif '_M_' in image_name:
            gender_age[i,0] = 1
        else:
            gender_age[i,0] = 0.5
        if 'Age' in image_name:
            image_name_split = image_name.split('_')
            index_of_age = image_name_split.index('Age') + 1
            gender_age[i,1] = int(image_name_split[index_of_age])/100
        else:
            gender_age[i,1] = 0.42 #mean age /100
    #print(gender_age)
    return gender_age
        
        
            

def imshow(inp, title=None):
    """Imshow for Tensor."""
#    print(np.amax(inp))
#    print(np.amin(inp))
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(24,18))
    plt.imshow(inp)
    plt.savefig('input_image_normalized', bbox_inches='tight', dpi =160)

    if title is not None:
        plt.title(title)
    #plt.pause(0.001)  # pause a bit so that plots are updated

def print_AUC(class_0,class_1):
    sensitivity = np.zeros(400)
    specifity = np.zeros(400)
    recalls = np.zeros(400)
    precisions = np.zeros(400)
    f1s = np.zeros(400)

    for threshold in range(-200,200):
        value = threshold/200
        index = threshold + 200
#        print(class_1)
        TP = sum(i.cpu().detach().numpy() > value for i in class_1)
        TN = sum(i.cpu().detach().numpy() < value for i in class_0)
        FP = sum(i.cpu().detach().numpy() > value for i in class_0)    
        FN = sum(i.cpu().detach().numpy() < value for i in class_1)
 #       print(TP)
        recalls[index] = recall(TP,FP,FN,TN)
        precisions[index] = precision(TP,FP,FN,TN)
        sensitivity[index] = TP/(TP+FN)
        specifity[index] = TN/(TN+FP)
        f1s[index] = f1(TP,FP,FN)

    return calculate_auc(1-specifity,sensitivity)

def calculate_auc(x,y):
    s = 0
#    print(y[0],y[1],x[0])
    for i in range(400-1):
        s = s + (-x[i+1]+x[i])*y[i]
    print()
    print('AUC', s)
    print()
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

#inputs, classes = iter(dataloaders['train']).next()

#out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train','test']:
            confusion = np.zeros(shape=(2,2))
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            datasetsize = 0
            
            class_0 = []
            class_1 = []
            test_outputs = []
            for (inputs, labels), (gender_age,_) in dataloaders[phase]:
                #print('labels',labels)
                print('gender_age',gender_age)
                
                gender_and_age_processed = process_all_image_names(gender_age)
                #print(inputs.shape)
                #print(gender_and_age_processed.shape)
                
#                gender_and_age_processed = gender_and_age_processed.to(device)
                gender_and_age_processed = torch.tensor(gender_and_age_processed).float().to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels *= 2
                labels -= 1
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs,gender_and_age_processed)
                
                    preds = outputs
                    print(outputs)
                    print(labels)
                    #loss = criterion(torch.transpose(outputs,0,1)[0,:], labels.float())
                    loss = criterion(outputs.view(-1), labels.float())
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                        
                        
                running_loss += loss.item() * inputs.size(0)
                if phase == 'test':
                    test_outputs.append(outputs.cpu().detach().numpy().reshape(-1,1))
                    test_outputs.append(labels.cpu().detach().numpy().reshape(-1,1))

                    for i in range(len(preds)):
                        if labels[i].float() < 0:
                            class_0.append(preds.detach().cpu()[i][0])
                        else:
                            class_1.append(preds.detach().cpu()[i][0])
                with torch.no_grad():
                    for element in preds:
                        if element[0] > 0.6:
                            element[0] = 1
                        else:
                            element[0] = -1
                
                datasetsize += labels.shape[0]
                running_corrects += torch.sum(torch.abs(labels.float() - preds.T)<0.01)
                    
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double()/ datasetsize
            
            print()
            print(phase, ' Results')
            
            print('Loss: {:.4f} Acc: {:.4f} Corrects: {}, Dataset size:{}'.format(epoch_loss, epoch_acc,running_corrects.double(),datasetsize))
            print('Fold:',data_dir[-2:-1])
            print('Learning rate:',learning_rate,'decay_step_size',decay_step_size)
            print(confusion)
            print('LR',args.lr,'BS',args.bs,'Fold',args.fold,'Decay',args.decay)
            print()
            
            if phase == 'train':
                train_accuracy.append(epoch_acc)
                train_loss.append(epoch_loss)
            else:
                test_accuracy.append(epoch_acc)
                test_loss.append(epoch_loss)
            
            plt.figure(figsize=(10,6))
            plt.rcParams["axes.edgecolor"] = "black"
            plt.rcParams["axes.linewidth"] = 1
            plt.rcParams.update({'font.size': 13})
            plt.plot(test_accuracy, label = 'Test accuracy')
            plt.plot(train_accuracy, label = 'Train accuracy')
            plt.title('Accuracy')
            plt.xlabel('Epochs (n)')
            plt.ylabel('Accuracy')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1,markerscale=0.3)
            plt.grid(True)
            plt.savefig('{}/{}_accuracy_epoch_{}.png'.format(current_result_dir,model_name,epoch), bbox_inches='tight', dpi =160)
            plt.close()
            
            plt.figure(figsize=(10,6))
            plt.rcParams["axes.edgecolor"] = "black"
            plt.rcParams["axes.linewidth"] = 1
            plt.rcParams.update({'font.size': 13})
            plt.plot(test_loss, label = 'Test Loss')  
            plt.plot(train_loss , label = 'Train Loss')
            plt.xlabel('Epochs (n)')
            plt.title('Loss')
            plt.ylabel('Loss')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1,markerscale=0.3)
            plt.grid(True)
            plt.savefig('{}/{}_loss_epoch_{}.png'.format(current_result_dir,model_name,epoch), bbox_inches='tight', dpi =160)
            plt.close()
            
            # deep copy the model
            if phase == 'test':
                best_acc = epoch_acc
                print('best acc so far',max(test_accuracy))
                #best_model_wts = copy.deepcopy(model.state_dict())
                #torch.save(model_ft.state_dict(), '/auto/data2/ademir/Dataset Clohe/{}/data/classes_{}'.format(experiment,experiment))
                df_cm = pd.DataFrame(confusion, index = [i for i in experiment],
                              columns = [i for i in experiment])
                #plt.figure(figsize = (10,7))
                #plt.xlabel('Prediction')
                #plt.ylabel('Truth')
                #plt.title('Confusion Matrix for {} set'.format(phase))     
                #sn.heatmap(df_cm, annot=True)
                #plt.savefig('{}/{}{}/data/densenet_{}_confusion_{}_batchize_{}_learning_rate_{}_{}.png'.format(data_dir,experiment,time_,experiment,epoch,batch_size,learning_rate,phase), bbox_inches='tight', dpi =160)
                #plt.show()  
                
                plt.figure(figsize = (10,7))
                plt.title('Histogram of outputs')
                plt.xlabel('Output')     
                bins = np.linspace(-1.5, 1.5, 30)
                plt.hist([class_0, class_1], bins, label=['class_0', 'class_1'])
                plt.legend(loc='upper right')
                plt.savefig('{}/{}_{}_histogram_{}.png'.format(current_result_dir,epoch,model_name,phase), bbox_inches='tight', dpi =160)
                plt.close()
                s = print_AUC(class_0, class_1)
                print("resnet sgd")
                print(np.asarray(test_outputs).shape)
                np.save('{}/{}_{}_{}.npy'.format(current_result_dir,model_name,epoch,'train_loss'),np.asarray(train_loss))
                np.save('{}/{}_{}_{}.npy'.format(current_result_dir,model_name, epoch,'test_loss'), np.asarray(test_loss))
                np.save('{}/{}_{}_output_{}.npy'.format(current_result_dir,model_name,epoch,s),np.asarray(test_outputs))
                print('densenet')
            torch.save(model.state_dict(), 'age_and_gender_model_{}'.format(epoch))

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))  
    print('Best val Acc: {:4f}'.format(best_acc))
    row_list =[run,args.bs,args.lr,80,args.decay,args.fold,epochs_num,'AdamW','MSE','{:4f}'.format(best_acc),model_name,str(pretraining),str(feature_extract)]
    with open('/auto/data2/yturali/Runs/runlog.csv','a', newline='',encoding='UTF8') as file:
        writer = csv.writer(file,delimiter='\t')
        writer.writerow(row_list)
    #model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=14):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            print(outputs)
            print(labels)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])
                
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training) 


class MyModel(nn.Module):
    def __init__(self,model_name,pretraining,feature_extract):
        super(MyModel, self).__init__()
        num_classes = 1
        if model_name == "densenet":
            
            self.cnn = models.densenet121(pretrained=pretraining)
            set_parameter_requires_grad(self.cnn, feature_extract)
            num_ftrs = self.cnn.classifier.in_features
            self.cnn.classifier = nn.Sequential()
            self.fc1 = nn.Sequential(nn.Linear(num_ftrs+2, 1), nn.Tanh())
            
        elif model_name == "alexnet":
            
            self.cnn = models.alexnet(pretrained=pretraining)
            set_parameter_requires_grad(self.cnn, feature_extract)
            num_ftrs = self.cnn.classifier[6].in_features
            self.cnn.classifier[6] = nn.Sequential()
            self.fc1 = nn.Sequential(nn.Linear(num_ftrs+2, 1), nn.Tanh())
            
        elif model_name == "resnet":

            self.cnn = models.resnet152(pretrained=pretraining)
            set_parameter_requires_grad(self.cnn, feature_extract)
            num_ftrs = self.cnn.fc.in_features
            self.cnn.fc = nn.Sequential()
            self.fc1 = nn.Sequential(nn.Linear(num_ftrs+2, 1), nn.Tanh())
            
        elif model_name == "vgg":
            
            self.cnn = models.vgg11_bn(pretrained=pretraining)
            set_parameter_requires_grad(self.cnn, feature_extract)
            num_ftrs = self.cnn.classifier[6].in_features
            self.cnn.classifier[6] = nn.Sequential()
            self.fc1 = nn.Sequential(nn.Linear(num_ftrs+2, 1), nn.Tanh())
        
        elif model_name == "squeezenet":
            self.cnn = models.squeezenet1_0(pretrained=pretraining)
            set_parameter_requires_grad(self.cnn, feature_extract)
            self.cnn.classifier[1] = nn.Sequential()
            self.fc1 = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            self.cnn.num_classes = num_classes
        
        elif model_name == "inception":
            
            self.cnn = models.inception_v3(pretrained=pretraining)
            self.cnn.aux_logits=False
            set_parameter_requires_grad(self.cnn, feature_extract)
            # # Handle the auxilary net
            # num_ftrs = self.cnn.AuxLogits.fc.in_features
            # self.cnn.AuxLogits.fc = nn.Sequential()
            # self.fc1 = nn.Sequential(nn.Linear(num_ftrs+2, 1), nn.Tanh())
            # Handle the primary net
            num_ftrs = self.cnn.fc.in_features
            self.cnn.fc = nn.Sequential()
            self.fc1 = nn.Sequential(nn.Linear(num_ftrs+2, 1), nn.Tanh())
            
        else:
            print("Invalid model name, exiting...")
            exit()
        
            
        
        
    def forward(self, image, data):
        x1 = self.cnn(image)
        x2 = data
#        print('x1.shape, x2.shape', x1.shape, x2.shape)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        return x



#num_classes = 1
#model_ft = models.densenet121(pretrained=True)
#num_ftrs = model_ft.classifier.in_features
#model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Tanh())
model_ft = MyModel(model_name,pretraining,feature_extract)
device = torch.device("cuda:0")
print(torch.cuda.get_device_name(device))
model_ft = nn.DataParallel(model_ft)
model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)


criterion = nn.MSELoss()
#optimizer_ft = torch.optim.ASGD(model_ft.parameters(), lr=learning_rate, t0=1800)
optimizer_ft = torch.optim.AdamW(params_to_update, lr=learning_rate/2)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=decay_step_size, gamma=0.80)#try 3

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=epochs_num)

#visualize_model(model_ft)

plt.ioff()
