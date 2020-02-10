import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import scipy.stats as stats
from catboost import CatBoostRegressor
import seaborn as sns
sns.set_palette("husl")

#To read excel file
diamond = pd.read_excel (r'project.xlsm',index_col=0)

#To get overview of the dataset
print ("Dataset Length: ", len(diamond)) 
print ("Dataset Shape: ", diamond)  
print ("Dataset: ",diamond.head()) 

#For visualization of carat
fig, axarr = plt.subplots(1,2, figsize=(15, 5))
diamond['carat'].plot.hist(ax=axarr[0])
diamond['carat'].plot.box(ax=axarr[1])

#For visualization of depth
fig, axarr = plt.subplots(1,2, figsize=(15, 5))
diamond['depth'].plot.hist(ax=axarr[0])
diamond['depth'].plot.box(ax=axarr[1])

#For visualization of table
fig, axarr = plt.subplots(1,2, figsize=(15, 5))
diamond['table'].plot.hist(ax=axarr[0])
diamond['table'].plot.box(ax=axarr[1])

#For visualization of price
fig, axarr = plt.subplots(1,2, figsize=(15, 5))
diamond['price'].plot.hist(ax=axarr[0])
diamond['price'].plot.box(ax=axarr[1])

#For visualization of x
fig, axarr = plt.subplots(1,2, figsize=(15, 5))
diamond['x'].plot.hist(ax=axarr[0])
diamond['x'].plot.box(ax=axarr[1])

#For visualization of y
fig, axarr = plt.subplots(1,2, figsize=(15, 5))
diamond['y'].plot.hist(ax=axarr[0])
diamond['y'].plot.box(ax=axarr[1])

#For visualization of z
fig, axarr = plt.subplots(1,2, figsize=(15, 5))
diamond['z'].plot.hist(ax=axarr[0])
diamond['z'].plot.box(ax=axarr[1])

# For cut analysis 
diamond['cut'].value_counts()
# In percentage
(diamond['cut'].value_counts()*100)/len(diamond['cut'])
#For plotting
diamond['cut'].value_counts().plot.bar()

#Test of Hypothesis
#Null Hypothesis: All cut categories are similar in nature
#Alternate Hyporthests: ALl cut categories are not similar in nature
df_anova = diamond[['price','cut']]
grps = pd.unique(df_anova.cut.values)
diamond_data = {grp:df_anova['price'][df_anova.cut == grp] for grp in grps}
 
F, p = stats.f_oneway(diamond_data['Ideal'], diamond_data['Premium'], diamond_data['Very Good'],diamond_data['Good'],diamond_data['Fair'])
print("p-value for significance is: ", p)

if p<0.05:
    print("Reject Null hypothesis")
else:
    print("Accept Null hypothesis")
    

#For clarity analysis
diamond['clarity'].value_counts()
#In percentage
(diamond['clarity'].value_counts()*100)/len(diamond['clarity'])
#For plotting
diamond['clarity'].value_counts().plot.bar()

#Test of Hypothesis
#Null Hypothesis: All clarity categories are similar in nature
#Alternate Hyporthests: ALl clarity categories are not similar in nature
df_anova = diamond[['price','clarity']]
grps = pd.unique(df_anova.clarity.values)
diamond_data = {grp:df_anova['price'][df_anova.clarity == grp] for grp in grps}
 
F, p = stats.f_oneway(diamond_data['I1'],diamond_data['SI2'],diamond_data['SI1'],diamond_data['VS2'],diamond_data['VS1'],diamond_data['VVS2'],diamond_data['VVS1'],diamond_data['IF'])
print("p-value for significance is: ", p)

if p<0.05:
    print("Reject Null hypothesis")
else:
    print("Accept Null hypothesis")

#For correlation matrix
diamond.corr()

#plotting correlationship matrix
cor=diamond.corr()
plt.figure(figsize=(16,5))
sns.heatmap(cor)

diamond.plot.scatter('price','carat',figsize=(15, 5))
diamond.plot.scatter('price','depth',figsize=(15, 5))
diamond.plot.scatter('price','table',figsize=(15, 5))
diamond.plot.scatter('price','x',figsize=(15, 5))
diamond.plot.scatter('price','y',figsize=(15, 5))
diamond.plot.scatter('price','z',figsize=(15, 5))

#Creating the model
X = diamond[['carat','cut','color','clarity','depth','table','x','y','z']]
from sklearn import preprocessing
le_cut=preprocessing.LabelEncoder()
le_cut.fit(['Fair','Good','Very Good','Premium','Ideal'])
X.iloc[:,1]=le_cut.transform(X.iloc[:,1])

le_clarity=preprocessing.LabelEncoder()
le_clarity.fit(['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])
X.iloc[:,3]=le_clarity.transform(X.iloc[:,3])

le_color=preprocessing.LabelEncoder()
le_color.fit(['J','I','H','G','F','E','D'])
X.iloc[:,2]=le_color.transform(X.iloc[:,2])


Y = diamond["price"] 
Y[0:5]

#Dividing the orginal dataset into train and test, to check the accuracy of different Machine Learning Algorithms and selecting the best out of them
X_train, X_test, y_train, y_test = train_test_split(  
X, Y, test_size = 0.3, random_state = 100)  

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 
    

#Applying decision tree algorithm
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
clf_entropy.fit(X_train, y_train) 
   

#prediction
y_pred = clf_entropy.predict(X_test) 
print("Predicted values:") 
print(y_pred) 

#Accuracy
print("Decision Tree Accuracy:",accuracy_score(y_test,y_pred))

#Applying random forest algorithm
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)

predictions = rf.predict(X_test)
errors = abs(predictions - y_test)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
#Accuracy: 93.51 %.

#Applying XGBoost algorithm
	
model =xgb.XGBRegressor()
model.fit(X_train, y_train)
print(model)

	
#Predictions for test data
y_predict = model.predict(X_test)
prediction = [round(value) for value in y_predict]


#Evaluating predictions
accuracy = accuracy_score(y_test, prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#Applying LGBM algorithm
import lightgbm as lgb
d_train = lgb.Dataset(X_train, label=y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
clf = lgb.train(params, d_train, 100)

#Prediction
y_pred=clf.predict(X_test)
#converting into binary values
for i in range(0,99):
    if y_pred[i]>=.5:       # setting threshold to 0.5
       y_pred[i]=1
    else:  
       y_pred[i]=0
       
Accuracy=accuracy_score(y_pred,y_test)
#Accuracy=98%

#Applying CatBoost algorithm
from catboost import CatBoostRegressor
model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')


model.fit(X_train, y_train,plot=True)

y_prediction = model.predict(X_test)
predict = [round(value) for value in y_prediction]

Accuracy = accuracy_score(y_test, predict)
print("Accuracy: %.2f%%" % (Accuracy * 100.0))

#So it can be concluded that LightGBM(LGBM) gives the most accurate outcome for the dataset.

























