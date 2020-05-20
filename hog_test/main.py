import cv2
import numpy as np

image = cv2.imread('./dataset/test.jpg')
hog = cv2.HOGDescriptor()
hog_feats = hog.compute(image)
print(hog_feats)
print(hog_feats.shape)