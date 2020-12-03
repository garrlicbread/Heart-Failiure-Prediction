# Heart-Failiure-Prediction

This is a simple classification project that predicts if a patient suffering from a heart failiure is going to survive or not.
The dataset, taken from kaggle, contains 12 variables besides the one we want to predict i.e. The outcome of the patient. The stanza below explains why and how only 4 of the 12 features were considered.

The independant features to be included were decided by their correlations with the dependant variable i.e. DEATH_EVENT and only those were selected that had correlations greater than +/- 0.2. The features satisfying this are:

|Features              |Correlations|
|:--------------------:|:----------:|
|        age           |  0.253729  |
|ejection_fraction     | -0.268603  |
|serum_creatinine      |  0.294278  |
|    DEATH_EVENT       |     1      |

Based on this table, our independant variables are obviously limited but following facts and a fairly reasonable hunch, smoking was also included as a feature.

 1) Age - Age of the patient
 2) Ejection Fraction - Percentage of blood leaving the heart at each contraction
 3) Serum Creatinine - Level of creatinine in the blood (Product of creatine phosphate from muscle and protein metabolism)
 4) Smoking - If the patient smokes
 
The script executes in 41.23 seconds and summarizes the accuracies of eight machine learning models employed as well as their predictions for a single patient with randomized but realistic features. The results are listed below:

|No.|Classification Models                            |Accuracies|Single Predictions|
|--:|------------------------------------------------:|---------:|-----------------:|
| 1 |Logistic Regression Classification               |   80     |                 1|
| 2 |K Nearest Neighbors Classification               |   90     |                 1|
| 3 |Support Vector Machine [Linear] Classification   |   83.33  |                 1|
| 4 |Support Vector Machine [Gaussian] Classification |   80     |                 1|
| 5 |Naive Bayes Classification                       |   80     |                 1|
| 6 |Decision Tree Classification                     |   76.67  |                 0|
| 7 |Random Forrest Classification                    |   83.33  |                 1|
| 8 |Gradient Boosting Classification                 |   85     |                 1|

Thanks: 

This project is based on a reaserch published by BMC Medical Informatics and Decision Making where they predict the survival of a patient suffering from heart failure from serum creatinine and ejection fraction alone. 

References:

1) BioMedCentral's open access thesis: https://rdcu.be/cbD65
2) Dataset: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data
