import os
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data 
y = iris.target


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42) 

from sklearn.tree import DecisionTreeClassifier
Model = DecisionTreeClassifier(random_state = 42) 

Model.fit(X_train,y_train)

y_pred = Model.predict(X_test) 

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
confusion_matrix_score = confusion_matrix(y_test,y_pred) 
print ("Confusion matrix:", confusion_matrix_score)
