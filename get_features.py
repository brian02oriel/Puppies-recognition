from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def setting_dataset(df, data_path, class_name):
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
        new_row = {"File":onlyfiles[i], "Class": class_name, "Features": hog_feats[:, 0] }
        df = df.append(new_row, ignore_index=True)
    print("Process completed")
    return df
    #print(len(features))

df = pd.DataFrame(columns=["File", "Class", "Features"])

data_path_1 = './train_set/n02085620-Chihuahua/'

df = setting_dataset(df, data_path_1, 1.0)

data_path_2 = './train_set/n02085782-Japanese_spaniel/'

df = setting_dataset(df, data_path_2, 2.0)

df.to_csv('./data/features.csv', index=False)




