#-----------------------------------
# TESTING CLASSIFIER
#-----------------------------------

# Importing the libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from skimage import feature
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
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt

#--------------------
# tunable-parameters
#--------------------
fixed_size = tuple((100, 100))
num_trees = 100
test_size = 0.10
seed      = 9
train_path = "Dataset/Training"
test_path  = "Dataset/Test"
h5_data    = 'output/data.h5'
h5_labels  = 'output/labels.h5'
scoring    = "accuracy"

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
# to visualize results

clf = SVC(random_state=seed)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)

# loop through the test images
for file in glob.glob(test_path + "/*.jpg"):
    # read the image
    image = cv2.imread(file)

    # resize the image
    image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
    featurehog = hog_feature(image)

    ###################################
    # Concatenate global features
    ###################################
    #global_feature = np.hstack([fv_hu_moments])
    global_feature = np.hstack([featurehog])

    # scale features in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_feature = scaler.fit_transform([global_feature])
    
    # predict label of test image
   # prediction = clf.predict(rescaled_feature.reshape(1,-1))[0]
    prediction = clf.predict(global_feature.reshape(1,-1))[0]
    print(prediction)
    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()