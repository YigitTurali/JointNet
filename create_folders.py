import os, random
from shutil import copyfile
from distutils.dir_util import copy_tree

def get_file_count(directory):
    print(directory)
    path, dirs, files = next(os.walk(directory))
    print(len(files))
    return len(files)

def create_test_train(data_dir,k):
    
    for folder in ['','/train','/test','/train/inaktif','/test/inaktif','/train/aktif','/test/aktif']:
        print(folder)
        if not os.path.exists('{}training_directory3/fold_{}{}'.format(data_dir,k,folder)):
            os.mkdir('{}training_directory3/fold_{}{}'.format(data_dir,k,folder))
    copy_tree('{}crop_and_clahe/aktif'.format(data_dir), "{}training_directory3/fold_{}/train/aktif".format(data_dir,k))    
    copy_tree('{}crop_and_clahe/inaktif'.format(data_dir), "{}training_directory3/fold_{}/train/inaktif".format(data_dir,k))    
    print('done')


    for class_ in ['aktif','inaktif']:
        for i in range(80):
            print('{}training_directory3/fold_{}/train/{}'.format(data_dir,k,class_))
            file_name = random.choice(os.listdir('{}training_directory3/fold_{}/train/{}'.format(data_dir,k,class_)))    
            os.rename('{}training_directory3/fold_{}/train/{}/{}'.format(data_dir,k,class_,file_name),'{}training_directory3/fold_{}/test/{}/{}'.format(data_dir,k,class_,file_name))
            print(file_name)
            
    ratio = 3#round(get_file_count('{}training_directory3/fold_{}/train/inaktif'.format(data_dir,k))/get_file_count('{}training_directory3/fold_{}/train/aktif'.format(data_dir,k)))
    
    for root, dirs, files in os.walk("{}training_directory3/fold_{}/train/aktif".format(data_dir,k)):
        if len(files) >= 1:
            for file_name in files:
                print('{}/{}'.format(root,file_name))
                for i in range(1,int(ratio)):
                    copyfile('{}/{}'.format(root,file_name), '{}/copy_{}{}'.format(root,i,file_name))
    
data_dir = '/auto/data2/yturali/new_data/'
for i in range(1,11):
    create_test_train(data_dir,i)
    print("Done",i)