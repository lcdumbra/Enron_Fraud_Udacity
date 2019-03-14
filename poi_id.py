#!/usr/bin/python
from time import time
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,  GridSearchCV, StratifiedShuffleSplit

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees'
                 'exercised_stock_options', 'expenses', 'from_messages', 'from_poi_to_this_person',
				 'from_this_person_to_poi', 'loan_advances', 'long_term_incentive', 'other',
				 'restricted_stock', 'restricted_stock_deferred', 'shared_receipt_with_poi',
                 'to_messages', 'total_payments', 'total_stock_value'] # You will need to use more features

### Load the dictionary containing the dataset 
with open("final_project_dataset_2.pkl", "rb") as data_file: 
    data_dict = pickle.load(data_file)
	
# Converting data to a Pandas dataframe
enron_df = pd.DataFrame.from_records(list(data_dict.values()))
enron_df.head(5)

# Paste dataframe content to be a series of employees
employees = pd.Series(list(data_dict.keys()))
enron_df.set_index(employees, inplace=True)

#Exploring data
print('Number of features:', len(enron_df.columns))
print('Number of employees:', len(employees))

poi_count = 0
for person in data_dict:
	if data_dict[person]["poi"]==1:
		poi_count +=1 
print("Number of POI in dataset:", poi_count)

enron_df.head(5)

enron_df.dtypes

enron_df = enron_df.apply(lambda x : pd.to_numeric(x, errors = 'coerce'))
enron_df.head()

# Finding the quantity of NaN values
list_missing_values = []
for feature in enron_df.columns:
    list_missing_values.append(enron_df[feature].isnull().sum())
print(list_missing_values)

# Bar plot
plt.bar(enron_df.columns, list_missing_values)
plt.xlabel("Features")
plt.xticks(rotation='vertical')
plt.ylabel("Missing Values")
plt.title("Features X Missing Values")
plt.show()

#Removing 'email_address' and replace nan values for zero
df_copy = enron_df
df_copy = df_copy.drop('email_address', axis=1)
df_copy = df_copy.fillna(0)
print('Number of features:', len(df_copy.columns))

df_copy.head()

# Updating features_list
features_list = ['poi','salary', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees',
                 'exercised_stock_options', 'expenses', 'from_messages', 'from_poi_to_this_person',
				 'from_this_person_to_poi', 'loan_advances', 'long_term_incentive', 'other',
				 'restricted_stock', 'restricted_stock_deferred','shared_receipt_with_poi',
                 'to_messages', 'total_payments', 'total_stock_value']

### Task 2: Remove outliers

plt.scatter(df_copy['salary'][df_copy['poi'] == True],df_copy['bonus'][df_copy['poi'] == True],color = 'red', label = 'POI')
plt.scatter(df_copy['salary'][df_copy['poi'] == False],df_copy['bonus'][df_copy['poi'] == False],color = 'blue', label = 'NoPOI')

plt.xlabel("bonus")
plt.ylabel("salary")
plt.title("Salary X Bonus")
plt.legend(loc='best')
plt.show()

df_copy['salary'].idxmax()

#Removing the first outlier: TOTAL
df_copy = df_copy.drop('TOTAL', axis=0)

plt.scatter(df_copy['salary'][df_copy['poi'] == True],df_copy['bonus'][df_copy['poi'] == True],color = 'red', label = 'POI')
plt.scatter(df_copy['salary'][df_copy['poi'] == False],df_copy['bonus'][df_copy['poi'] == False],color = 'blue', label = 'NoPOI')

plt.xlabel("bonus")
plt.ylabel("salary")
plt.title("Salary X Bonus")
plt.legend(loc='best')
plt.show()

df_copy.loc['THE TRAVEL AGENCY IN THE PARK']

# Removing the second outlier: THE TRAVEL AGENCY IN THE PARK
df_copy = df_copy.drop('THE TRAVEL AGENCY IN THE PARK', axis = 0)

plt.scatter(df_copy['total_payments'][df_copy['poi'] == True], df_copy['deferral_payments'][df_copy['poi'] == True],
            color = 'r', label = 'POI')

plt.scatter(df_copy['total_payments'][df_copy['poi'] == False], df_copy['deferral_payments'][df_copy['poi'] == False],
            color = 'b', label = 'NoPOI')

    
plt.xlabel('total_payments')
plt.ylabel('deferral_payments')
plt.title('total_payments X deferral_payments')
plt.legend(loc='best')
plt.show()

df_copy['deferral_payments'].idxmax()

# Removing the third outlier: 'FREVERT MARK A'
df_copy = df_copy.drop('FREVERT MARK A', axis = 0)

# Findind the numerical index of POIs e NoPOIs
poi_index = []
non_poi_index = []

for i in range(len(df_copy['poi'])):
    if(df_copy['poi'][i] == True):
        poi_index.append(i+1)
    else:
        non_poi_index.append(i+1)
        
# Scatterplot
plt.scatter(poi_index, df_copy['long_term_incentive'][df_copy['poi'] == True],
            color = 'r', label = 'POI')
plt.scatter(non_poi_index, df_copy['long_term_incentive'][df_copy['poi'] == False],
            color = 'b', label = 'NoPOI')

    
plt.xlabel('Employees')
plt.ylabel('long_term_incentive')
plt.title("Employee number com long_term_incentive")
plt.legend(loc='best')
plt.show()

df_copy['long_term_incentive'].idxmax()

#Removing the fourth outlier: 'MARTIN AMANDA K'
df_copy = df_copy.drop('MARTIN AMANDA K', axis = 0)

plt.scatter(df_copy['restricted_stock'][df_copy['poi'] == True],df_copy['restricted_stock_deferred'][df_copy['poi'] == True],
            color = 'r', label = 'POI')

plt.scatter(df_copy['restricted_stock'][df_copy['poi'] == False], df_copy['restricted_stock_deferred'][df_copy['poi'] == False],
            color = 'b', label = 'Not-POI')

    
plt.xlabel('restricted_stock')
plt.ylabel('restricted_stock_deferred')
plt.title("Scatterplot of restricted_stock X restricted_stock_deferred")
plt.legend(loc='best')
plt.show()

df_copy['restricted_stock_deferred'].idxmax()

# Removendo the fifth outlier: 'BHATNAGAR SANJAY'
df_copy = df_copy.drop('BHATNAGAR SANJAY', axis = 0)

plt.scatter(df_copy['from_poi_to_this_person'][df_copy['poi'] == True], df_copy['from_this_person_to_poi'][df_copy['poi'] == True],
            color = 'r', label = 'POI')
plt.scatter(df_copy['from_poi_to_this_person'][df_copy['poi'] == False], df_copy['from_this_person_to_poi'][df_copy['poi'] == False],
            color = 'b', label = 'Not-POI')
    
plt.xlabel('from_poi_to_this_person')
plt.ylabel('from_this_person_to_poi')
plt.title("Contagem de e-mails enviados e recebidos por POI")
plt.legend(loc='best')
plt.show()

## Only to test without created data. ##

## Converting the modified dataframe in a dictionary

# data_dict = df_copy.to_dict('index')
# print("Total number of datapoints:",len(data_dict))
# print("Total number of features:",len(data_dict['ELLIOTT STEVEN']))

## Store to my_dataset_first for easy export below.

# my_dataset = data_dict

## Extract features and labels from dataset for local testing
# data = featureFormat(my_dataset, features_list, sort_keys = True)
# labels, features = targetFeatureSplit(data)

#########################################

df_copy['bonus_salary_ratio'] = df_copy['bonus']/df_copy['salary']

df_copy.head()

df_copy['proportion_mail_from_poi'] = df_copy['from_poi_to_this_person']/df_copy['from_messages'] 
df_copy['proportion_mail_to_poi'] = df_copy['from_this_person_to_poi']/df_copy['to_messages']

# In case of division by zero
df_copy = df_copy.replace('inf', 0)
df_copy = df_copy.fillna(0)

df_copy.head()

# Converting the modified dataframe to a dictionary
data_dict = df_copy.to_dict('index')
print("Total number of datapoints:",len(data_dict))
print("Total number of features:",len(data_dict['ELLIOTT STEVEN']))

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# features_list updated
features_list = ['poi','salary', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees',
                 'exercised_stock_options', 'expenses', 'from_messages', 'from_poi_to_this_person',
				 'from_this_person_to_poi', 'loan_advances', 'long_term_incentive', 'other',
				 'restricted_stock', 'restricted_stock_deferred','shared_receipt_with_poi',
                 'to_messages', 'total_payments', 'total_stock_value', 'bonus_salary_ratio',
                 'proportion_mail_from_poi', 'proportion_mail_to_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Split data into training and testing datasets
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
	
# Decision Tree
tree_clf = DecisionTreeClassifier(random_state=42)
start = time()
tree_clf.fit(features_train, labels_train)
end = time()
tree_fit_time = end - start
print("Tempo de treinamento:", tree_fit_time, "segundos")
start = time()
pred = tree_clf.predict(features_test)
end = time()
tree_pred_time = end - start
print("Tempo de previsão:", tree_pred_time, "segundos")
print()
report = classification_report(labels_test, pred)
print(report)

#SVM
svm_clf = SVC(gamma='auto', random_state=42)
start = time()
svm_clf.fit(features_train, labels_train)
end = time()
svm_fit_time = end - start
print("Tempo de treinamento:", svm_fit_time, "segundos")
start = time()
pred = svm_clf.predict(features_test)
end = time()
svm_pred_time = end - start
print("Tempo de previsão:", svm_pred_time, "segundos")
print()
report = classification_report(labels_test, pred)
print(report)

#GaussianNB
gaussian_clf = GaussianNB()
start = time()
gaussian_clf.fit(features_train, labels_train)
end = time()
gaussian_fit_time = end - start
print("Tempo de treinamento:", gaussian_fit_time, "segundos")
start = time()
pred = gaussian_clf.predict(features_test)
end = time()
gaussian_pred_time = end - start
print("Tempo de previsão:", gaussian_pred_time, "segundos")
print()
report = classification_report(labels_test, pred)
print(report)

#Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
start = time()
rf_clf.fit(features_train, labels_train)
end = time()
rf_fit_time = end - start
print("Tempo de treinamento:", rf_fit_time, "segundos")
start = time()
pred = rf_clf.predict(features_test)
end = time()
rf_pred_time = end - start
print("Tempo de previsão:", rf_pred_time, "segundos")
print()
report = classification_report(labels_test, pred)
print(report)

#AdaBoost
ada_clf = AdaBoostClassifier(random_state=42)
start = time()
ada_clf.fit(features_train, labels_train)
end = time()
ada_fit_time = end - start
print("Tempo de treinamento:", ada_fit_time, "segundos")
start = time()
pred = ada_clf.predict(features_test)
end = time()
ada_pred_time = end - start
print("Tempo de previsão:", ada_pred_time, "segundos")
print()
report = classification_report(labels_test, pred)
print(report)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#Decision Tree
parameters = dict(min_samples_split = range(2,50), max_depth = range(2,20))
tree_gs = GridSearchCV(tree_clf, parameters, cv=5)
start = time()
tree_gs.fit(features_train, labels_train)
end = time()
tree_gs_fit_time = end - start
print("Tempo de treinamento: ", tree_gs_fit_time, "segundos")
start = time()
pred = tree_gs.predict(features_test)
end = time()
tree_gs_pred_time = end - start
print("Tempo de previsão: ", tree_gs_pred_time, "segundos")
print()
tree_gs_clf = tree_gs.best_estimator_
report = classification_report(labels_test, pred)
print(report)
print()
print("Descrição da melhor Árvore de Decisão:")
print()
print(tree_gs_clf)

#SVM
estimators = [('reduce_dim', PCA()), ('clf', svm_clf)]
pipe = Pipeline(estimators)
parameters = {'clf__C':[0.001,0.1,10,100,1000], 'clf__gamma':[1.0,0.1,0.01]}
svm_gs = GridSearchCV(pipe, parameters, cv=5)
start = time()
svm_gs.fit(features_train, labels_train)
end = time()
svm_gs_fit_time = end - start
print("Tempo de treinamento: ", svm_gs_fit_time, "segundos")
start = time()
pred = svm_gs.predict(features_test)
end = time()
svm_gs_pred_time = end - start
print("Tempo de previsão: ", svm_gs_pred_time, "segundos")
print()
svm_gs_clf = svm_gs.best_estimator_
report = classification_report(labels_test, pred)
print(report)
print()
print("Descrição da melhor Máquina de Vetores de Suporte:")
print()
print(svm_gs_clf)

# GaussianNB
pipeline = Pipeline(steps = [("SKB", SelectKBest()), ("clf",gaussian_clf)])
param_grid = {"SKB__k":[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]}

grid = GridSearchCV(pipeline, param_grid, verbose = 0, cv = 5, scoring = 'f1')

start = time()
grid.fit(features, labels)
end = time()
grid_fit_time = end - start
print("Tempo de Treinamento Grid:", grid_fit_time, "segundos")

# best algorithm
gaussian_gs_clf = grid.best_estimator_

# refit the best algorithm:
gaussian_gs_clf.fit(features_train, labels_train)

start = time()
pred = gaussian_gs_clf.predict(features_test)
end = time()
gaussian_gs_pred_time = end - start
print("Tempo de previsão do Melhor GaussianNB:", gaussian_gs_pred_time, "segundos")

report = classification_report(labels_test, pred)
print(report)

#Random Forest
parameters = dict(n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200])
rf_gs = GridSearchCV(rf_clf, parameters, cv=5)
start = time()
rf_gs.fit(features_train, labels_train)
end = time()
rf_gs_fit_time = end - start
print("Tempo de treinamento: ", rf_gs_fit_time, "segundos")
start = time()
pred = rf_gs.predict(features_test)
end = time()
rf_gs_pred_time = end - start
print("Tempo de previsão: ", rf_gs_pred_time, "segundos")
print()
rf_gs_clf = rf_gs.best_estimator_
report = classification_report(labels_test, pred)
print(report)
print()
print("Descrição da melhor Floresta Aleatória:")
print()
print(rf_gs_clf)

# AdaBoost
parameters = dict(n_estimators = [50, 100, 200, 400],
                  learning_rate = [0.01, 0.1, 1])
ada_gs = GridSearchCV(ada_clf, parameters, cv=5)
start = time()
ada_gs.fit(features_train, labels_train)
end = time()
ada_gs_fit_time = end - start
print("Tempo de treinamento: ", ada_gs_fit_time, "segundos")
start = time()
pred = ada_gs.predict(features_test)
end = time()
ada_gs_pred_time = end - start
print("Tempo de previsão: ", ada_gs_pred_time, "segundos")
print()
ada_gs_clf = ada_gs.best_estimator_
report = classification_report(labels_test, pred)
print(report)
print()
print("Descrição da melhor Floresta Aleatória:")
print()
print(ada_gs_clf)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
clf = gaussian_gs_clf
dump_classifier_and_data(clf, my_dataset, features_list)