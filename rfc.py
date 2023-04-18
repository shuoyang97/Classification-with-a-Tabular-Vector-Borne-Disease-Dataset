from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("/Users/shuoyang/Desktop/playground-series-s3e13/train.csv")
train = np.array(train_df)

# test_df = pd.read_csv("/Users/shuoyang/Desktop/playground-series-s3e13/test.csv")
# test = np.array(test_df)

length = len(train)
depth = len(train[0])

X = train[:, 1:64]
y = train[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

clf = RandomForestClassifier(max_depth=10, random_state=1)
clf.fit(X_train, y_train)
pred = clf.predict(X_train)
para = clf.get_params()
acc = clf.score(X_test, y_test)

print(para)
print(pred)
print(acc)
