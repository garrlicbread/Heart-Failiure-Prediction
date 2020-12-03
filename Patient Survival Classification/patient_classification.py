# A classification python project predicts the likelihood of a patient passing away using different models.
# Features include age, smoking, gender, diabetes, anemia, plateletes, serum creatinine or sodium etc

# dataset: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

# Initializing starting time
import time
start = time.time()

# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Importing data and assigning variables
dataset = pd.read_csv('C:/Users/Sukant Sidnhwani/Desktop/Python/Projects/Patient Survival Classification/heart_failure_clinical_records_dataset.csv')
del dataset['time']

# Regression Sensitivity Analysis brought to light the following learnings:
# Features that improve performance = Age, Ejection_fraction, serum creatinine, smoking
# Features that impair performance = serum Sodium, diabetes, sex, anaemia, creatinine_phosphokinase, high_blood_pressure, platelets, sex
# Features that are worthless = Time

X = dataset[['age', 'ejection_fraction', 'serum_creatinine', 'smoking']]  
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)

# Visualizing the correlations
corr = dataset.corr()
# corr[abs(corr['DEATH_EVENT']) > 0.1]['DEATH_EVENT']
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns)

print()
print("We will decide the variables to be included based on their correlations with the Dependant Variable i.e. Death_Event and consider those features that have correlations greater than +/- 0.2")
print()
print(corr[abs(corr['DEATH_EVENT']) > 0.2]['DEATH_EVENT'])
print()
print("Based on these results, our independant variables will be: \n\n 1) Age \n 2) Ejection Fraction \n 3) Serum Creatinine\n")
print("However, following facts and a fairly reasonable hunch, let's include Smoking as a feature too.")

# Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)
y_train = y_train.reshape(-1, 1).reshape(-1)

# Scaling the features.
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)

# Fitting a logistic regression model 
log_cl = LogisticRegression(max_iter = 10000).fit(X_train, y_train)
y_pred1 = log_cl.predict(X_test)

ac1 = accuracy_score(y_test, y_pred1)

# Fitting a K nearest neighbours model 

# Default method i.e. manually choose the number of neighbours 
# knn_cl = KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2).fit(X_train, y_train)
# y_pred2 = knn_cl.predict(X_test)

# ac2 = accuracy_score(y_test, y_pred2)

# Iterative method i.e. run a loop of different neighbours and choose the best one 
scorelist = []
kindex = []

for i, neighbours in enumerate(range(1, 25)):
    knn_cl = KNeighborsClassifier(n_neighbors = neighbours)
    knn_cl.fit(X_train, y_train)
    y_pred2 = knn_cl.predict(X_test)
    ac2 = accuracy_score(y_test, y_pred2)
    scorelist.append(ac2)
    kindex.append(i)
       
ac2 = sorted(scorelist, reverse = True)[:1]
ac2 = float(np.array(ac2))
kindex = sorted(kindex, reverse = True)[:1]
kindex = int(np.array(kindex))

knn_cl = KNeighborsClassifier(n_neighbors = kindex).fit(X_train, y_train)

# Fitting Support Vector Machine model [Linear]
svm_cl = SVC(kernel = 'linear').fit(X_train, y_train)
y_pred3 = svm_cl.predict(X_test)

ac3 = accuracy_score(y_test, y_pred3)

# Fitting Support Vector Machine model [Gaussian]
svmg_cl = SVC(kernel = 'rbf').fit(X_train, y_train)
y_pred4 = svmg_cl.predict(X_test)

ac4 = accuracy_score(y_test, y_pred4)

# Fitting Naive Bayes Model
nb_cl = GaussianNB().fit(X_train, y_train)
y_pred5 = nb_cl.predict(X_test)

ac5 = accuracy_score(y_test, y_pred5)

# Fitting a Decision tree model 
dt_cl = DecisionTreeClassifier(criterion = 'entropy').fit(X_train, y_train)
y_pred6 = dt_cl.predict(X_test)

ac6 = accuracy_score(y_test, y_pred6)

# Fitting a Random Forrest model with an iterative loop 
rflist = []
rfindex = []

for i, trees in enumerate(range(10, 200)):
    rf_cl = RandomForestClassifier(n_estimators = trees, criterion = 'entropy')
    rf_cl.fit(X_train, y_train)
    y_pred7 = rf_cl.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred7)
    rflist.append(rf_accuracy)
    rfindex.append(i)
       
ac7 = sorted(rflist, reverse = True)[:1]
ac7 = float(np.array(ac7))
rindex = sorted(rfindex, reverse = True)[:1]
rindex = int(np.array(rindex))

rf_cl = RandomForestClassifier(n_estimators = rindex, criterion = 'entropy').fit(X_train, y_train)
    
# Fitting a Gradient Boost with an iterative loop 
gblist = []
gbindex = []

for i, estimators in enumerate(range(5, 200)):
    gb_cl = GradientBoostingClassifier(learning_rate = 0.01, n_estimators = estimators)
    gb_cl.fit(X_train, y_train)
    y_pred8 = gb_cl.predict(X_test)
    gb_accuracy = accuracy_score(y_test, y_pred8)
    gblist.append(gb_accuracy)
    gbindex.append(i)
    
ac8 = sorted(gblist, reverse = True)[:1]
ac8 = float(np.float64(ac8))
gbindex = sorted(gbindex, reverse = True)[:1]
gbindex = int(np.array(gbindex))

gb_cl = GradientBoostingClassifier(learning_rate = 0.01, n_estimators = gbindex).fit(X_train, y_train)

# Evaluating all accuracies into a list
list1 = [ac1, ac2, ac3, ac4, ac5, ac6, ac7, ac8]
list1 = [i * 100 for i in list1]
list1 = [round(num, 2) for num in list1]

# Predicting a single scenario with randomized independant variables

# Creating demo values 
demo = [66,     # Age
        37,     # Ejection Fraction
        0.87,   # Serum Creatinine
        1]      # Smoking 
      
demo = pd.DataFrame(demo)
demo = demo.transpose()
demo = demo.to_numpy()

# Applying different regressors 
a = log_cl.predict(demo)
b = knn_cl.predict(demo)
c = svm_cl.predict(demo)
d = svmg_cl.predict(demo)
e = nb_cl.predict(demo)
f = dt_cl.predict(demo)
g = rf_cl.predict(demo)
h = gb_cl.predict(demo)

# Making a list of demo predictions
list2 = [a, b, c, d, e, f, g, h]
list2 = np.array(list2, dtype = 'object')

classification = pd.DataFrame()
classification['Classification Models'] = ['Logistic Regression Classification', 
                                           'K Nearest Neighbors Classification', 
                                           'Support Vector Machine [Linear] Classification', 
                                           'Support Vector Machine [Gaussian] Classification',
                                           'Naive Bayes Classification',
                                           'Decision Tree Classification', 
                                           'Random Forrest Classification',
                                           'Gradient Boosting Classification']

classification['Accuracies'] = list1
classification['Single Predictions'] = list2

# Initiating ending time
end = time.time()
print()
print(f"This program executes in {round((end - start), 2)} seconds.")
# print()

# print(classification)

## Saving the classification report to a .csv file
# classification.to_csv('Models.csv')