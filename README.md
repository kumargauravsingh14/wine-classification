# wine-classification

**Our goal is to discover what’s the quality of a wine considering only its chemical properties.**

Our dataset is downloaded from [UCI open repository](http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv) and is composed by the following features:
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol
- quality

Implemented Random Forest classifier(with n_estimators = 100) to classify wine as bad, average or good here referred as 1, 2, and 3 respectively. We have used Grid Search cross-validation for tuning hyperparameters. We have also used PCA to get rid of unnecessary features. Metrics used are - precision, recall, f1 score, and accuracy. At last, we dump the model to store it for future use.
