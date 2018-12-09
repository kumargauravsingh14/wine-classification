
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns

# Visualisation
import matplotlib.pyplot as plt

# Ease data preprocessing
from sklearn import preprocessing

# Import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# PCA
from sklearn.decomposition import PCA

# Import cross-validation tools
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Import metrics to evaluate model performance
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Use model in future
from sklearn.externals import joblib


# In[28]:


# downloading dataset into pandas dataframe
url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(url, sep=';')


# In[29]:


# checking data - all integers
data.head()


# In[30]:


# data needs standardising
data.describe()


# In[31]:


# managable size
data.shape


# In[32]:


# different quality count
data.groupby('quality').size()


# In[33]:


# no NAs
data.isna().values.any()


# In[34]:


# three classes - bad, average and good
targets = []

for q in data['quality']:
    if q <= 4:
        targets.append(1)
    elif q >= 5 and q <= 6:
        targets.append(2)
    elif q >= 7:
        targets.append(3)
        
data['target'] = targets


# In[35]:


# skewed data
data.groupby('target').size()


# In[36]:


# no need of quality now
data = data.drop('quality', axis=1)


# In[37]:


# no strong correlation
data.corr()*100


# In[38]:


sns.pairplot(data)


# In[39]:


# test and train split
y = data.target
X = data.drop(['target'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                   test_size=0.2,
                                                   random_state=123)


# In[40]:


# pipelining
pipeline = make_pipeline(preprocessing.StandardScaler(),
                        RandomForestClassifier(n_estimators=100))


# In[41]:


# Removing useless features
pca = PCA()  
X_train = pca.fit_transform(X_train)  
X_test = pca.transform(X_test)


# In[16]:


# the variance caused by each feature on the dataset
explained_variance = pca.explained_variance_ratio_ 
for i in explained_variance:
    print(format(i*100, 'f'))


# In[42]:


# the four first features of our data capture almost 99.5% of the variance
pca = PCA(n_components=4)  
X_train = pca.fit_transform(X_train)  
X_test = pca.transform(X_test)


# In[18]:


# hyper parameters for our model
print(pipeline.get_params())


# In[43]:


# the hyper parameters we want to tune through cross-validation
hyperparameters = { 'randomforestclassifier__max_features' : ['auto', 'sqrt', 'log2'],
                   'randomforestclassifier__max_depth' : [None, 10, 7, 5, 3, 1]}


# In[44]:


# performs cross-validation across all possible permutations of hyper parameters
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)


# In[45]:


# best parameters
print(clf.best_params_)


# In[46]:


# predicting over test set
y_pred = clf.predict(X_test)


# In[47]:


# confusion matrix to check how the model classified the different wines on the dataset
print('Accuracy score:', accuracy_score(y_test, y_pred))
print("-"*80)
print('Confusion matrix\n')
conmat = np.array(confusion_matrix(y_test, y_pred, labels=[1,2,3]))
confusion = pd.DataFrame(conmat, index=['Actual 1', 'Actual 2', 'Actual 3'],
                         columns=['predicted 1','predicted 2', 'predicted 3'])
print(confusion)
print("-"*80)
print('Classification report')
print(classification_report(y_test, y_pred, target_names=['1','2', '3']))


# In[48]:


# save our previous model to apply it to future data
# joblib.dump(clf, 'wine-classifier.pkl')


# In[49]:


# load the model
# clf2 = joblib.load('wine-classifier.pkl')

# Predict new data
# clf2.predict(X_test)

