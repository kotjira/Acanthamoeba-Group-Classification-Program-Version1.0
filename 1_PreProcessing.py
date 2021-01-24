#-----------------------------------
# PRE-PROCESSING 
#-----------------------------------

# Importing the libraries
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('Dataset/Test/test1.jpg')
image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gaussianblur = cv2.GaussianBlur(image_gray,(5,5),0)
canny = cv2.Canny(gaussianblur,100,10)


plt.subplot(1,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(canny,cmap = 'gray')
plt.title('Outputcanny'), plt.xticks([]), plt.yticks([])

plt.show()