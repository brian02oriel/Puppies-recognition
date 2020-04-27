from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

Data_Set, Labels = [], []

def setting_dataset(data_path, class_name):
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path)
        gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)


        # h x w in pixels
        cell_size = (8, 8) 

        # h x w in cells
        block_size = (2, 2) 

        # number of orientation bins
        nbins = 9

        # Using OpenCV's HOG Descriptor
        # winSize is the size of the image cropped to a multiple of the cell size
        hog = cv2.HOGDescriptor(_winSize=(gray.shape[1] // cell_size[1] * cell_size[1],
                                    gray.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
        #print(hog)
        # Create numpy array shape which we use to create hog_feats
        #n_cells = (gray.shape[0] // cell_size[0], gray.shape[1] // cell_size[1])

        #print(n_cells)

        # We index blocks by rows first.
        # hog_feats now contains the gradient amplitudes for each direction,
        # for each cell of its group for each group. Indexing is by rows then columns.
        hog_feats = hog.compute(gray)
        #print(hog_feats)

        Data_Set.append(hog_feats)
        Labels.append(class_name) # 1 for japanese spaniel | 0 for non japanese spaniel

        
data_path_1 = './n02085782-Japanese_spaniel/'

setting_dataset(data_path_1, 1.0)

data_path_2 = './n02085620-Chihuahua/'

setting_dataset(data_path_2, 2.0)

print("len", len(Data_Set))
print("shape 1", Data_Set[0].shape)


print(type(Data_Set))
print(type(Labels))
X = np.array(Data_Set)
y = np.array(Labels)
y = y.reshape(len(y), 1)
X = X.reshape(len(X), 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 4)

print(type(X_train))
print(type(y_train))
print(X_train.shape)
print(y_train.shape)
#print(X_train)
#print(y_train)

clf = svm.SVC()
#clf.fit(X_train, y_train)
print("finalizado")

