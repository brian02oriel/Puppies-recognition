import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
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
