#-----------------------------------
# GLOBAL FEATURE EXTRACTION
#-----------------------------------

# Importing the libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from skimage import feature
import numpy as np
import cv2
import os
import h5py


#--------------------
# tunable-parameters
#--------------------
images_per_class    = 250
fixed_size          = tuple((100, 100))
train_path          = "Dataset/Training"
h5_data             = 'output/data.h5'
h5_labels           = 'output/labels.h5'


# feature-descriptor-1: Histogram of oriented gradients

def hog_feature(image):
    #Grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Gaussain Blur Deleted noise
    gaussain = cv2.GaussianBlur(img,(5,5),0)
    # Canny Edge Detection
    canny_image = cv2.Canny(gaussain,100,10)
    hog_desc = feature.hog(canny_image, orientations=9, pixels_per_cell=(6, 6),
                cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    return hog_desc


# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels          = []

# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    # loop over the images in each sub-folder
    for x in range(1,images_per_class+1):
        # get the image file name
        file = dir + "/" + str(x) + ".jpg"

        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)
        
        ####################################
        # Global Feature extraction
        ####################################
        featurehog = hog_feature(image)
        

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([featurehog])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")

print("\n")


#2
# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le          = LabelEncoder()
target      = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

# scale features in the range (0-1)
scaler            = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")
print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File("h5_data.hdf5", 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File("h5_labels.hdf5", 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")
