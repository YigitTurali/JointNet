import os, random
from shutil import copyfile
from distutils.dir_util import copy_tree
data_dir = '/auto/data2/yturali/new_data/'
def get_file_count(directory):
    path, dirs, files = next(os.walk(directory))
    return len(files)
for i in range(1,11):
  ratio = round(get_file_count('{}training_directory3/fold_{}/train/inaktif'.format(data_dir,i))/get_file_count('{}training_directory3/fold_{}/train/aktif'.format(data_dir,i)))
  print(ratio)
  print(get_file_count('{}training_directory3/fold_{}/train/inaktif'.format(data_dir,i)))
  print(get_file_count('{}training_directory3/fold_{}/train/aktif'.format(data_dir,i)))