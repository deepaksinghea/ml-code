#Sample ML WorkFlow

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy : "+str(round(accuracy,2)*100)+"%")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :")
print(cm)

from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1',
                'class 2']
print(classification_report(y_test, y_pred,
                            target_names=target_names))

from joblib import dump 
dump(clf, 'classifer.joblib') 
