import Data
import logistic_regression
import NaiveBayes
import SVM
from sklearn.metrics import accuracy_score
import numpy as np
x_train, x_test, y_train, y_test = Data.process()

log1,log2 = logistic_regression.train_outputs()
svm1,svm2 = SVM.train_outputs()
nb1,nb2 = NaiveBayes.train_outputs()

X = np.concatenate((log1.reshape(len(log1),1) , log2.reshape(len(log2),1), svm1.reshape(len(svm1),1), svm2.reshape(len(svm2),1),nb1.reshape(len(nb1),1),nb2.reshape(len(nb2),1)),axis = 1)

from sklearn import linear_model
model = linear_model.LogisticRegression()
model.fit(X,y_train)

log1,log2 = logistic_regression.test()
svm1,svm2 = SVM.test()
nb1,nb2 = NaiveBayes.test()

X = np.concatenate((log1.reshape(len(log1),1) , log2.reshape(len(log2),1), svm1.reshape(len(svm1),1), svm2.reshape(len(svm2),1),nb1.reshape(len(nb1),1),nb2.reshape(len(nb2),1)),axis = 1)

prediction = model.predict(X)

print accuracy_score(y_test,prediction)

def s(x):
    log1,log2 = logistic_regression.predict(x)
    svm1,svm2 = SVM.predict(x)
    nb1,nb2 = NaiveBayes.predict(x)
    X = np.concatenate((log1.reshape(len(log1),1) , log2.reshape(len(log2),1), svm1.reshape(len(svm1),1), svm2.reshape(len(svm2),1),nb1.reshape(len(nb1),1),nb2.reshape(len(nb2),1)),axis = 1)
    prediction = model.predict(X)
    return prediction
