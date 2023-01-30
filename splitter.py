import os
import random
import shutil
"""It defines the path of the data folder as "yolo_data" It creates two lists, im_file_list and label_file_list, 
which contain the file names of all the images and labels respectively. The lists are sorted alphabetically and only 
includes files that have the '.jpg' and '.txt' extensions. It creates a dictionary named mapped_lists that maps the 
image files to the corresponding label files. It converts the dictionary to a list of tuples named items and shuffles 
the order of the items using the random.shuffle() function. It splits the items list into two parts, the first 80% is 
named train_80 and the last 20% is named val_20. It defines the paths of the folders where the images and labels will 
be copied to: im_train_folder, lb_train_folder, im_val_folder, lb_val_folder It iterates through the train_80 
dictionary and copies the image files and label files to the corresponding training folders. It iterates through the 
val_20 dictionary and copies the image files and label files to the corresponding validation folders. """

path = "yolo_data"
im_file_list = sorted([f for f in os.listdir(path) if f.endswith('.jpg')])
label_file_list = sorted([f for f in os.listdir(path) if f.endswith('.txt')])
mapped_lists = dict(zip(im_file_list, label_file_list))
items = list(mapped_lists.items())
random.shuffle(items)
shuffled_dict = dict(items)
# Splitting the items list
split = int(len(items) * 0.8)
train_80 = dict(items[:split])
val_20 = dict(items[split:])
im_train_folder = r"data\yolo_data\images\train"
lb_train_folder = r"data\yolo_data\labels\train"
im_val_folder = r"data\yolo_data\images\val"
lb_val_folder = r"data\yolo_data\labels\val"
for im, lb in train_80.items():
    shutil.copy(path + '\\' + im, im_train_folder + '\\' + im)
    shutil.copy(path + '\\' + lb, lb_train_folder + '\\' + lb)

for im, lb in val_20.items():
    shutil.copy(path + '\\' + im, im_val_folder + '\\' + im)
    shutil.copy(path + '\\' + lb, lb_val_folder + '\\' + lb)
