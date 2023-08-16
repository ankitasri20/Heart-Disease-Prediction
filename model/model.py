from copyreg import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle


#loading the data
heart_data = pd.read_csv('Heart_Disease_Prediction.csv')

#separating the result from data to train our model
train_model = heart_data.drop(columns='Heart Disease', axis=1)
results = heart_data['Heart Disease']

#We have decided to train our model on 80% of the data and rest 20% for validating it.
train_model_train, train_model_test, result_train, result_test = train_test_split(train_model, results, test_size=0.2, stratify=results, random_state=2)




def fit_and_score(models, X_train, X_test, y_train, y_test):
    # Random seed for reproducible results
    np.random.seed(42)
    # Make a list to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


models = {
          "Logistic Regression": LogisticRegression(), 
          "Random Forest": RandomForestClassifier(),
          "KNN":KNeighborsClassifier(),
          "Gauss Naive Bayes":GaussianNB(),
          "Support Vector Machine": svm.SVC()
        }

model_scores = fit_and_score(models= models,
                         X_train=train_model_train,
                         X_test=train_model_test,
                         y_train=result_train,
                         y_test=result_test)
    

print(model_scores)
max_acc = -1
model = None
for i in model_scores:
    if max_acc < model_scores[i]:
        max_acc = model_scores[i]
        model = models[i]

# model = LogisticRegression()
print(model)
model.fit(train_model_train, result_train)


filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))