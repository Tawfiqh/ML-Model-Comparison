
# Model evaluation mini-project  
Aim: Develop a program to evaluate the performance of several supervised models on regression datasets  

In this repo we look at applying a variety of SupervisedLearning models to predict a variety of regression data-sets.  

## Setup
I used this dataset from Kaggle for car prices:
https://www.kaggle.com/CooperUnion/cardataset
(save it in the root directory as "car_data.csv")

I used this dataset from Kaggle for test_scprse:
https://www.kaggle.com/kwadwoofosu/predict-test-scores-of-students
(save it in the root directory as "test_scores.csv")



https://www.kaggle.com/cherngs/heart-disease-cleveland-uci
(save it in the root directory as "heart_cancer_data.csv")


## How to run the project
Run the project by running:
```
$ python3 main.py
```

Install dependencies with:
```
$ pip3 install -r requirements.txt
```


## Description

The following models were run on the dataset:
- K-Nearest-Neighbours
- Linear Regression
- Decision Tree
- Random Forest
- SVR

Evaluates the performance of 5 key models:
- it should evaluate the performance on the validation set ✅
- it should return a train, val and test loss value and R-squared score for that hyperparameterisation ✅
- it should return the hyperparameters which resulted in that score ✅
- the time taken to fit the model ✅
- evaluate them on all of sklearn's toy regression datasets available in sklearn.datasets ✅

- create a main.py file which loops through each dataset and each model, printing the results ✅

- graphical visualisations of
- time to fit each of the best models ✅
- final train, validation and test set loss/mse scores ✅
- final train, validation and test set R-squared scores (= model.score) ✅

