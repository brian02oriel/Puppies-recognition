from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def setting_dataset(data_path, class_name):
    Data_Set, Labels = [], []
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    hog = cv2.HOGDescriptor()

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path)
        images = cv2.resize(images, (360, 480))
        #print('channels: ', len(images.shape))
        gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        #print(i)
        #print(onlyfiles[i])

        hog_feats = hog.compute(images)
        hog_feats = np.asarray(hog_feats)
        #print(hog_feats.shape)

        Data_Set.append(hog_feats[:, 0])
        Labels.append(class_name)
    print("Process completed")

    #print(len(features))
    return Data_Set, Labels

        
data_path_1 = './n02085620-Chihuahua/'

class1_features, class1_labels = setting_dataset(data_path_1, 1.0)
class1_features = np.array(class1_features)
class1_labels = np.array(class1_labels)

print(class1_features.shape)
print(class1_labels.shape)


data_path_2 = './n02085782-Japanese_spaniel/'

class2_features, class2_labels = setting_dataset(data_path_2, 2.0)
class2_features = np.array(class2_features)
class2_labels = np.array(class2_labels)

print(class2_features.shape)
print(class2_labels.shape)

X = class1_features + class2_features
y = class1_labels + class2_labels
X = np.array(X)
y = np.array(y)

print("len: {0} | shape: {1}".format(len(X), X.shape))
print("len: {0} | shape: {1}".format(len(y), y.shape))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 4)

print(X_train.shape)
print(y_train.shape)

clf = svm.SVC()
clf.fit(X_train, y_train)
svm_prediction = clf.predict(X_test)

output = pd.DataFrame({'Real value': y_test, 'Prediction': svm_prediction})
output.to_csv('my_submission.csv', index=False)
print(output)
print("finalizado")

