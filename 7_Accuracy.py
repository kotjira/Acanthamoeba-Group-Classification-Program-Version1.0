#-----------------------------------
# ACCURACY
#Created on Wed Jan 13 17:40:42 2021
#@author: Mas
#-----------------------------------

# Importing the libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 
import imutils
import glob
import h5py
import os
import h5py
import glob
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#--------------------
# tunable-parameters
#--------------------
num_trees = 100
test_size = 0.10
seed      = 9
train_path = "Dataset/Training"
test_path  = "Dataset/Test"
h5_data    = 'output/data.h5'
h5_labels  = 'output/labels.h5'
scoring    = "accuracy"

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()

if not os.path.exists(test_path):
    os.makedirs(test_path)

# import the feature vector and trained labels
h5f_data  = h5py.File("h5_data.hdf5", 'r')
h5f_label = h5py.File("h5_labels.hdf5", 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string   = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels   = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))


#-----------------------------------
# TESTING OUR MODEL
#-----------------------------------

clf = SVC(random_state=seed)
#clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed) 

clf.fit(trainDataGlobal, trainLabelsGlobal)

prediction = clf.predict(testDataGlobal)
print(prediction)
    
accuracy = accuracy_score(testLabelsGlobal, prediction)
print ("accuracy = ", accuracy)
cm = confusion_matrix(testLabelsGlobal, prediction)
print (cm)
plt.imshow(cm, cmap='binary')

print("\n")

c_m = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(c_m, annot=True, fmt='.2f')

print (c_m)
plt.imshow(c_m, cmap='binary')

plt.show()