from sklearn import linear_model
import Data
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = Data.load()

vec1 = Data.tfidf(x_train)
x_train1 = vec1.transform(x_train)


model1 = linear_model.LogisticRegression()
model1.fit(x_train1,y_train)
	
	
	
vec2 = Data.bag_of_words(x_train)
x_train2 = vec2.transform(x_train)


model2 = linear_model.LogisticRegression()
model2.fit(x_train2,y_train)

def train_outputs():
    pred1 = model1.predict(x_train1)
    pred2 = model2.predict(x_train2)
    return pred1,pred2
    
def test(x=x_test):
	x_test1 = vec1.transform(x_test)
	x_test2 = vec2.transform(x_test)
	pred1 = model1.predict(x_test1)
	pred2 = model2.predict(x_test2)
	return pred1,pred2
	
	
def accuracy(predictions,y=y_test):
	return accuracy_score(y_test,predictions)


def predict(x):
    x = vec1.transform(x)
    pred1 = model1.predict(x)
    pred2 = model2.predict(x)
    return pred1,pred2