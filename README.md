This Python project analyses a COVID-19 dataset and builds a logistic regression predictor using scikit-learn.
It applies feature scaling, hyperparameter tuning (GridSearchCV), and recursive feature elimination (RFECV) to evaluate and optimize model performance.

**COVID-19 Dataset:**
Nexoid (2024), a software company located in London, has a large global medical dataset focusing on COVID-19 infection.
Formed from collecting people’s responses to an online calculator called “Nexoid Covid survival calculator”. 
https://www.covid19survivalcalculator.com/
Original dataset contains about one million records
Calculator generates two risk values based on the person’s response to the questions asked by the calculator. 
One value is about infection risk and the other is about mortality risk.
Dataset consists of demographical, geographical, behaviour, segmentation, health condition, and risk values. 
Data used is a Subset consisting of 5789 records 

**Project Features**

Data pre-processing:
* Age range mapping to midpoints
* One-hot encoding of categorical variables
* Removal of non-predictive fields (e.g., income)
* Logistic Regression baseline model
* Hyperparameter tuning with GridSearchCV
* Feature selection with RFECV

Model comparison using:
* Accuracy
* Classification Report
* ROC AUC Score
* ROC Curves

Visualizations:
* Coefficient plots
* Hyperparameter tuning curves
* ROC curves

Python Libraries
pip install numpy pandas matplotlib scikit-learn

Output
Console logs with:
* Training and test accuracy
* Classification reports
* Optimal hyperparameters
* Selected features via RFE

Plots for:
* Model coefficients
* Hyperparameter tuning
* ROC curves

**Dataset variables in Logistic Model are:**

No.	Column name	     Description

1	Gender		        Male, Female or other

2	Age		            Age quantile

3	Height		        Height of the person in cm

4	Weight		        Weight of the person in kg

5	Blood type        Type of the person's blood

6	Insurance	        If the person has insurance or not?

7	Income		        Type of income (e.g., low, medium, high, or gov)

8	Smoking		        Information on how often the person smokes

9	Alcohol		        Level of alcohol consumption

10	Contacts count  Number of people the person has contacted

11	Working		      Status of the person's work

12	Worried		      On a scale of 1 to 5, indicating how worried the person is

13	Covid19 positive 1 or 0, indicating positive or negative status
