# Heart-Failiure-Prediction

This is a simple classification project that predicts if a patient suffering from heart failure is going to survive or not. It builds a classifier trained on eight different machine learning models and predicts the outcome of an artificial patient. 

The dataset, taken from kaggle, contains 12 variables besides the one we want to predict i.e. The outcome of the patient. The stanza below explains why and how only 4 of the 12 features were considered.

The independant features to be included were decided by their correlations with the dependant variable i.e. DEATH_EVENT and only those were selected that had correlations greater than +/- 0.2. The features satisfying this are:

|Features              |Correlations|
|:--------------------:|:----------:|
|age                   |  0.253729  |
|ejection_fraction     | -0.268603  |
|serum_creatinine      |  0.294278  |
|DEATH_EVENT           |     1      |

Based on this table, our independant variables are obviously limited but following facts and a fairly reasonable hunch, smoking was also included as a feature.

 1) Age - Age of the patient
 2) Ejection Fraction - Percentage of blood being pushed out of the left ventricle of a heart at each contraction
 3) Serum Creatinine - Level of creatinine in the blood (Product of creatine phosphate from muscle and protein metabolism)
 4) Smoking - If the patient smokes
 
The classifier is trained in ~40 seconds and after training we initialize an aritifical patient to test the predictions.
Lets assume that our aritifical patient is an old chain smoker with 0.87 mg/dL of serum creatinine and 37% ejection fraction.
We don't know for sure if the patient will survive or not but the numbers above don't seem impressive.
Now we call all our eight models to predict the outcome of this patient.

The table below summarizes the accuracies of the models employed as well as their predictions for our aritifical patient. The results are listed below:

|No.|Classification Models                            |Accuracies|      Single Predictions     |
|--:|------------------------------------------------:|---------:|----------------------------:|
| 1 |Logistic Regression Classification               |   80.00  |The patient will not survive.|
| 2 |K-Nearest Neighbors Classification               |   90.00  |The patient will not survive.|
| 3 |Support Vector Machine [Linear] Classification   |   83.33  |The patient will not survive.|
| 4 |Support Vector Machine [Gaussian] Classification |   80.00  |The patient will not survive.|
| 5 |Naive Bayes Classification                       |   80.00  |The patient will not survive.|
| 6 |Decision Tree Classification                     |   76.67  |The patient will survive.    |
| 7 |Random Forrest Classification                    |   83.33  |The patient will not survive.|
| 8 |Gradient Boosting Classification                 |   85.00  |The patient will not survive.|

Results:

1) It is clear that K-nearest neighbors classifier has the highest accuracy at 90%.
2) All other models barring Decision Tree classifier have decent accuracies ranging between 80-85%.
3) The Decision Tree classifier is the only one that predicts an outcome favoring the patient. All other models predict that the patient will unfortunately, not survive the failiure.

Thanks: 

This project is based on a research published by BMC Medical Informatics and Decision Making where they predict the survival of a patient suffering from heart failure from serum creatinine and ejection fraction alone. 

References:

1) BioMedCentral's open access thesis: https://rdcu.be/cbD65
2) Dataset: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data
