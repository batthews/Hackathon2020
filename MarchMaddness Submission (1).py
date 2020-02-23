#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import sklearn


# In[262]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# In[300]:


seasonData = pd.read_csv('MDataFiles_Stage1/'+'MRegularSeasonDetailedResults.csv')
seasonList = seasonData['Season'].unique()
teamList = seasonData['LTeamID'].unique()


# In[284]:


kenPomData = pd.read_csv('Hackathon2020/NCAA2020_Kenpom.csv')
print(kenPomData.columns)
newDataSet = kenPomData.drop(['FirstD1Season','LastD1Season','adj_o_rank', 'adj_d_rank','adj_tempo_rank','luck_rank',
                              'sos_adj_em_rank','sos_adj_d_rank','nc_sos_adj_em_rank','team','sos_adj_o_rank'], axis = 1)


# In[285]:


newDataSet = newDataSet[newDataSet.Season > 2002]
newDataSet


# In[302]:


tourneyData = pd.read_csv('MDataFiles_Stage1/'+'MNCAATourneyDetailedResults.csv')
seasons = tourneyData['Season'].unique()
columnsWeCareAbout = ['adj_em','adj_o','luck','adj_tempo','sos_adj_em','sos_adj_o','sos_adj_d','nc_sos_adj_em','ncaa_seed']
trainingData = []
for season in seasons:
    seasonalTourneyData = tourneyData[tourneyData.Season == season]
    for row in seasonalTourneyData.iterrows():
        wTeamID = row[1]['WTeamID']
        lTeamID = row[1]['LTeamID']
        #include grab all features from the seasonAvg dataframe based on season and teamID
        wTeam = (newDataSet.loc[((newDataSet.TeamID == wTeamID) & (newDataSet.Season == season)), columnsWeCareAbout ].values[0])
        lTeam = (newDataSet.loc[((newDataSet.TeamID == lTeamID) & (newDataSet.Season == season)), columnsWeCareAbout ].values[0])
        featureList = np.concatenate([wTeam,lTeam])
        featureList = np.append(featureList, 1)
        feature2 = np.concatenate([lTeam,wTeam])
        feature2 = np.append(feature2, 0)
        trainingData.append(featureList)
        trainingData.append(feature2)
        
temp = pd.DataFrame(trainingData, columns = ['adj_em','adj_o','luck','adj_tempo','sos_adj_em',
                                            'sos_adj_o','sos_adj_d','nc_sos_adj_em','ncaa_seed',
                                            'T2_adj_em','T2_adj_o','T2_luck','T2_adj_tempo','T2_sos_adj_em','T2_sos_adj_o',
                                             'T2_sos_adj_d','T2_nc_sos_adj_em','T2_ncaa_seed','Team1Win'])



temp.to_csv("modelData_v1.csv", index = False)


# In[ ]:





# In[279]:


mydata = pd.read_csv('modelData_v1.csv')

X= mydata.iloc[:,:-1] 
y = mydata.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .25)


# In[280]:


model1 = LogisticRegression()
model1.fit(X_train, y_train)


# In[281]:


y_pred = model1.predict(X_test)
y_test1 = y_test.tolist()


# In[282]:


totalright = 0
for i in range(0, len(y_pred)):
    zero1 = 0
    if y_pred[i] > .5:
        zero1 = 1
    
    if y_test1[i] == zero1:
        totalright+=1

percentRight = totalright/len(y_pred)
print(percentRight)


# In[283]:


y_pred


# In[278]:


y_test


# In[ ]:





# In[ ]:




