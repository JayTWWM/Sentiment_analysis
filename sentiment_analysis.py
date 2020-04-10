import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV

train_frame = pd.read_csv('Train.csv') 
test_frame = pd.read_csv('Test.csv')

# print(train_frame.describe())

x_train = train_frame['text']
y_train = train_frame['label']

x_test = test_frame['text']
y_test = test_frame['label']

cv = CountVectorizer()  
features = cv.fit_transform(x_train)

tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}

model = GridSearchCV(svm.SVC(), tuned_parameters)

model.fit(features,y_train)

print('Accuracy = ',model.score(cv.transform(x_test),y_test))