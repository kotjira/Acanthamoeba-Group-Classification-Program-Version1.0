import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Dataacanthamoeba/1.jpg')
grayscale = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
gaussianblur = cv2.GaussianBlur(grayscale,(5,5),0)
canny = cv2.Canny(gaussianblur,100,10)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(canny,cmap = 'gray')
plt.title('Outputcanny'), plt.xticks([]), plt.yticks([])

plt.show()