Which models to try (from sklearn):
- linear regression (Ben)
- KNN (Tawfiq)
- Decision Tree
- Random forest (Eddy)
- SVM (Kautar)

# Model evaluation mini-project
## Aim: Develop a program to evaluate the performance of several supervised models on sklearn's toy datasets
​
## Project requirements
- evaluate the performance of 4 key models:
    - Linear regression ✅
    - Decision tree 
    - KNN  ✅
    - SVM ✅

- implement a base class which all model classes inherit from ✅


- for each model tested, develop a class which implements a function to find the best hyperparameterisation on a given dataset ✅
    - it should evaluate the performance on the validation set ✅
    - it should return a train, val and test loss value and R-squared score for that hyperparameterisation ✅
    - it should return the hyperparameters which resulted in that score ✅
    - the time taken to fit the model ✅


- evaluate them on all of sklearn's toy regression datasets available in sklearn.datasets  ✅
- create a main.py file which loops through each dataset and each model, printing the results ✅



- graphical visualisations of
    - time to fit each of the best models
    - final train, validation and test set loss scores
    - final train, validation and test set R-squared scores

- small paragraph written for each model explaining hypotheses for your visualised results

- Grid-search-CV = we want to give our test/train split
	- KNN - we want to give it grid-search-CV
	- We need to update all our function-signatures - to fit on a train/test

        - Decision tree 
