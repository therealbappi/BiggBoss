#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix,roc_curve,auc
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


# In[2]:


bigg_boss = pd.read_csv('F:\\B-TECH\\datasets\\archive\\Bigg_Boss_India.csv', encoding = "ISO-8859-1")
nRow, nCol = bigg_boss.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[3]:


bigg_boss.tail()


# In[4]:


bigg_boss.tail(10).T


# In[5]:


bigg_boss.tail(10)


# In[6]:


bigg_boss.info()


# In[7]:


bigg_boss.describe()


# In[8]:


for col in bigg_boss.columns:
    print("Number of unique values in", col,"-", bigg_boss[col].nunique())


# In[9]:


# Number of seasons in all Indian languages
print(bigg_boss.groupby('Language')['Season Number'].nunique().sum())


# In[10]:


# Number of seasons in each Indian language
print(bigg_boss.groupby('Language')['Season Number'].nunique().nlargest(10))


# In[11]:


fig = plt.figure(figsize=(10,4))
ax = sns.countplot(x='Language', data=bigg_boss)
ax.set_title('Bigg Boss Series - Indian Language')
for t in ax.patches:
    if (np.isnan(float(t.get_height()))):
        ax.annotate(0, (t.get_x(), 0))
    else:
        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))


# In[12]:


print(bigg_boss['Wild Card'].value_counts(), "\n")
print(round(bigg_boss['Wild Card'].value_counts(normalize=True)*100))
sns.countplot(x='Wild Card', data=bigg_boss)


# In[13]:


# Participant's Profession
print(bigg_boss['Profession'].value_counts())
fig = plt.figure(figsize=(20,5))
sns.countplot(x='Profession', data=bigg_boss)
plt.xticks(rotation=90)


# In[14]:


fig = plt.figure(figsize=(20,5))
ax = sns.countplot(x='Broadcasted By', data=bigg_boss, palette='RdBu')
ax.set_title('Bigg Boss Series - Indian Broadcastor & Total Number of Housemates')
for t in ax.patches:
    if (np.isnan(float(t.get_height()))):
        ax.annotate(0, (t.get_x(), 0))
    else:
        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))


# In[15]:


bigg_boss.groupby('Host Name')['Season Number'].nunique().nlargest(25)


# In[16]:


# Maximum TRP of Bigg Boss Hindi/India seasons
print("Maximum TRP",bigg_boss['Average TRP'].max(), "\n")
print(bigg_boss.loc[bigg_boss['Average TRP']==bigg_boss['Average TRP'].max()][["Language","Season Number"]].head(1).to_string(index=False))


# In[17]:


# All BB Winners
bigg_boss.loc[bigg_boss.Winner==1]


# In[18]:


# Profession of  Winners
bigg_boss.loc[bigg_boss.Winner==1,'Profession'].value_counts()


# In[19]:


# Gender of Season title Winners
bigg_boss.loc[bigg_boss.Winner==1,'Gender'].value_counts()


# In[20]:


# Number of eliminations or evictions faced by the Bigg Boss competition winners
bigg_boss.loc[bigg_boss.Winner==1,'Number of Evictions Faced'].value_counts().sort_index()


# In[21]:


# Entry type of the Season Winners
#No Wild card has won
bigg_boss.loc[bigg_boss.Winner==1,'Wild Card'].value_counts()


# In[22]:


# No re-entered contestant won Bigg Boss title
bigg_boss.loc[bigg_boss.Winner==1,'Number of re-entries'].value_counts()


# In[23]:


bigg_boss.loc[bigg_boss.Winner==1,'Number of times elected as Captain'].value_counts().sort_index()
# data is not up-to-date


# In[24]:


bigg_boss.loc[(bigg_boss['Language']=='Telugu')]


# In[25]:


# Bigg Boss Telugu Winners
bigg_boss.loc[(bigg_boss['Language']=='Telugu') & (bigg_boss['Winner']==1), :]


# In[26]:


# Bigg Boss Telugu current season participants
# Use this for weekly eliminations 
bigg_boss.loc[(bigg_boss['Language']=='Telugu') & (bigg_boss['Winner'].isnull()), :]


# In[27]:


# Handling NULL values
bigg_boss.isnull().sum()


# In[28]:


# Removing records where Name field is empty
bigg_boss = bigg_boss.loc[bigg_boss.Name.notnull()]
bigg_boss.reset_index(drop=True,inplace=True)


# In[29]:


# Contestant might have faced at least one eviction, so filling NaN with 'Number of Evictions Faced' with 1
bigg_boss['Number of Evictions Faced'] = bigg_boss['Number of Evictions Faced'].fillna(1)

# Number of re-entries are very less, so filling NULLs in 'Number of re-entries' with 0
bigg_boss['Number of re-entries'] = bigg_boss['Number of re-entries'].fillna(0)

# Filling blank values in 'Average TRP' column with average
bigg_boss['Average TRP'] = bigg_boss['Average TRP'].fillna(bigg_boss['Average TRP'].mean())


# In[30]:


bigg_boss['Season Start Date'] = pd.to_datetime(bigg_boss['Season Start Date'])
bigg_boss['Season End Date'] = pd.to_datetime(bigg_boss['Season End Date'])
bigg_boss['Entry Date'] = pd.to_datetime(bigg_boss['Entry Date'])
bigg_boss['Elimination Date'] = pd.to_datetime(bigg_boss['Elimination Date'])


# In[31]:


bigg_boss.head()


# In[32]:


bigg_boss.tail()


# In[33]:


train = bigg_boss.loc[(bigg_boss['Winner'].notnull()), :]
train.sample(10)


# In[34]:


test = bigg_boss.loc[(bigg_boss['Language']=='Telugu') & (bigg_boss['Winner'].isnull()), :]
test


# In[35]:


bigg_boss.isnull().sum()


# In[36]:


bigg_boss = bigg_boss.loc[bigg_boss.Name.notnull()]
bigg_boss.reset_index(drop=True,inplace=True)


# In[37]:


bigg_boss['Number of Evictions Faced'] = bigg_boss['Number of Evictions Faced'].fillna(1)
bigg_boss['Number of re-entries'] = bigg_boss['Number of re-entries'].fillna(0)
bigg_boss['Average TRP'] = bigg_boss['Average TRP'].fillna(bigg_boss['Average TRP'].mean())


# In[38]:


bigg_boss['Season Start Date'] = pd.to_datetime(bigg_boss['Season Start Date'])
bigg_boss['Season End Date'] = pd.to_datetime(bigg_boss['Season End Date'])
bigg_boss['Entry Date'] = pd.to_datetime(bigg_boss['Entry Date'])
bigg_boss['Elimination Date'] = pd.to_datetime(bigg_boss['Elimination Date'])


# In[39]:


bigg_boss.head()


# In[40]:


bigg_boss.tail()


# In[41]:


train = bigg_boss.loc[(bigg_boss['Winner'].notnull()), :]
train.sample(10)


# In[42]:


test = bigg_boss.loc[(bigg_boss['Language']=='Telugu') & (bigg_boss['Winner'].isnull()), :]
test


# In[43]:


BB_telugu_participant = test[['Name']]
BB_telugu_participant.reset_index(drop=True, inplace=True)
BB_telugu_participant


# In[44]:


train.drop(["Name","Entry Date","Elimination Date","Season Start Date","Season End Date","Elimination Week Number"], axis=1, inplace=True)
test.drop(["Name","Entry Date","Elimination Date","Season Start Date","Season End Date","Elimination Week Number","Winner"], axis=1, inplace=True)


# In[45]:


train.head()


# In[46]:


# One Hot Encoding

target = train.pop('Winner')
data = pd.concat([train, test])
dummies = pd.get_dummies(data, columns=data.columns, drop_first=True, sparse=True)
train2 = dummies.iloc[:train.shape[0], :]
test = dummies.iloc[train.shape[0]:, :]


# In[47]:


x_train, x_val, y_train, y_val = train_test_split(train2, target, test_size=0.2, random_state=2019)
print(x_train.shape, x_val.shape)


# In[48]:


def plot_confusion_matrix():
    cm = confusion_matrix(y_val, y_predicted_val).T
    cm = cm.astype('float')/cm.sum(axis=0)
    ax = sns.heatmap(cm, annot=True, cmap='Blues');
    ax.set_xlabel('True Label',size=12)
    ax.set_ylabel('Predicted Label',size=12)


# # Logistic Regression

# In[49]:


for c in [0.01, 1, 10, 100, 1000]:
    lr = LogisticRegression(random_state=2019, C=c).fit(x_train, y_train)
    print ("F1 score for C=%s: %s" % (c, f1_score(y_val, lr.predict(x_val), average='weighted')*100))


# In[50]:


logi = LogisticRegression(random_state=2019,C=100).fit(x_train, y_train)
logi


# In[51]:


predicted_val_logi = logi.predict_proba(x_val)[:, 1]
y_predicted_val = (predicted_val_logi > 0.3).astype("int").ravel()
print('F1 Score -',f1_score(y_val, y_predicted_val, average='weighted')*100)
print('Accuracy Score -',accuracy_score(y_val, y_predicted_val)*100)


# In[52]:



predicted_val_logi = logi.predict_proba(test)[:, 1]
winner = pd.concat([BB_telugu_participant, pd.DataFrame(predicted_val_logi, columns=['Predicted_Winner'])],axis=1)
print(winner[['Name','Predicted_Winner']])


# In[53]:


print(min(winner['Predicted_Winner'])*10000)


# # RandomForest

# In[54]:



rf = RandomForestClassifier(n_estimators=200, random_state=2019).fit(x_train, y_train)
rf


# In[55]:


predicted_val_rf = rf.predict_proba(x_val)[:, 1]
y_predicted_val = (predicted_val_rf > 0.3).astype("int").ravel()
print('F1 Score -',f1_score(y_val, y_predicted_val, average='weighted')*100)
print('Accuracy Score -',accuracy_score(y_val, y_predicted_val)*100)

# n_estimators=100 accuracy 99.4
# n_estimators=200 accuracy 100


# In[56]:


predicted_val_rf = rf.predict_proba(test)[:,1]
winner = pd.concat([BB_telugu_participant, pd.DataFrame(predicted_val_rf, columns=['Predicted_Winner'])],axis=1)
winner[['Name','Predicted_Winner']]


# # MLP Classifier

# In[57]:


NN = MLPClassifier(random_state=2019)
#NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 20), random_state=2019)
NN.fit(x_train, y_train)


# In[58]:


predicted_val_nn = NN.predict(x_val)
# predicted_val_nn = NN.predict_proba(x_val)[:,1]
# y_predicted_val = (predicted_val_nn > 0.03).astype("int").ravel()
print('F1 Score -',f1_score(y_val, y_predicted_val, average='weighted')*100)
print('Accuracy Score -',accuracy_score(y_val, y_predicted_val)*100)


# In[59]:


predicted_val_nn = NN.predict(test)
winner = pd.concat([BB_telugu_participant, pd.DataFrame(predicted_val_nn, columns=['Predicted_Winner'])],axis=1)
winner[['Name','Predicted_Winner']]


# # KNN Regressor

# In[60]:


from pprint import pprint
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
regressor=neighbors.KNeighborsRegressor(n_neighbors=2)


# In[61]:


regressor.fit(x_train,y_train)


# In[62]:


y_pred1=regressor.predict(x_val)


# In[63]:


from sklearn.metrics import mean_squared_error
import numpy as np
mean_squared_error(y_val,y_pred1)


# In[64]:


from sklearn.metrics import r2_score
r2=r2_score(y_val,y_pred1)
print(r2)


# In[65]:


predicted_val_nn = regressor.predict(test)
winner = pd.concat([BB_telugu_participant, pd.DataFrame(predicted_val_nn, columns=['Predicted_Winner'])],axis=1)
winner[['Name','Predicted_Winner']]


# # Ridge Regression

# In[66]:


from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
rr = Ridge(alpha=0.01)
rr.fit(x_train, y_train) 
pred_train_rr= rr.predict(x_train)



pred_test_rr= rr.predict(x_val)
print(np.sqrt(mean_squared_error(y_val,pred_test_rr))) 
print(r2_score(y_val, pred_test_rr))


# In[67]:


predicted_val_nn = rr.predict(test)
winner = pd.concat([BB_telugu_participant, pd.DataFrame(predicted_val_nn, columns=['Predicted_Winner'])],axis=1)
winner[['Name','Predicted_Winner']]


# # Lasso Regression

# In[68]:


from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=0.01)
model_lasso.fit(x_train, y_train) 
pred_train_lasso= model_lasso.predict(x_train)


pred_test_lasso= model_lasso.predict(x_val)
print(np.sqrt(mean_squared_error(y_val,pred_test_lasso))) 
print(r2_score(y_val, pred_test_lasso))


# In[69]:


predicted_val_nn = model_lasso.predict(test)
winner = pd.concat([BB_telugu_participant, pd.DataFrame(predicted_val_nn, columns=['Predicted_Winner'])],axis=1)
winner[['Name','Predicted_Winner']]


# # LSTM 

# In[ ]:


# Open For Contribution


# In[ ]:




