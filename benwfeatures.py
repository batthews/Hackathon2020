#!/usr/bin/env python
# coding: utf-8

# In[502]:


get_ipython().system('pip install bracketeer')


# In[2]:


import numpy as np
import pandas as pd
import sklearn


# In[262]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# In[325]:


seasonData = pd.read_csv('MDataFiles_Stage1/'+'MRegularSeasonDetailedResults.csv')
seasonList = seasonData['Season'].unique()
teamList = seasonData['LTeamID'].unique()
seasonData


# In[465]:


kenPomData = pd.read_csv('Hackathon2020/NCAA2020_Kenpom.csv')
print(kenPomData.columns)
newDataSet = kenPomData.drop(['FirstD1Season','LastD1Season','adj_o_rank', 'adj_d_rank','adj_tempo_rank','luck_rank',
                              'sos_adj_em_rank','sos_adj_d_rank','nc_sos_adj_em_rank','team','sos_adj_o_rank'], axis = 1)
newDataSet = newDataSet[newDataSet.Season > 2002]
newDataSet


# In[ ]:





# In[534]:


fgpct = []
twopct = []
ftpct = []
win14daypct = []
oR = []
dR = []
tO = []
for i, row in newDataSet.iterrows():
    w1 = seasonData.loc[(seasonData.Season == newDataSet.at[i,'Season']) & (seasonData.WTeamID == newDataSet.at[i,'TeamID'])]
    l1 = seasonData.loc[(seasonData.Season == newDataSet.at[i,'Season']) & (seasonData.LTeamID == newDataSet.at[i,'TeamID'])]
    fgpct.append((w1['WFGM3'].sum() + l1['LFGM3'].sum()) / (w1['WFGA3'].sum() + l1['LFGA3'].sum()))
    twopct.append((w1['WFGM'].sum() + l1['LFGM'].sum()) / (w1['WFGA'].sum() + l1['LFGA'].sum()))
    ftpct.append((w1['WFTM'].sum() + l1['LFTM'].sum()) / (w1['WFTA'].sum() + l1['LFTA'].sum()))
    win14daypct.append((w1[w1.DayNum > 118].shape[0]) / (w1[w1.DayNum>118].shape[0] + l1[l1.DayNum > 118].shape[0]))
    oR.append((w1['WOR'].sum() + l1['LOR'].sum()) / (w1['WOR'].shape[0] + l1['LOR'].shape[0]))
    dR.append((w1['WDR'].sum() + l1['LDR'].sum()) / (w1['WDR'].shape[0] + l1['LDR'].shape[0]))
    tO.append((w1['WTO'].sum() + l1['LTO'].sum()) / (w1['WTO'].shape[0] + l1['LTO'].shape[0]))


# In[535]:


newDataSet = newDataSet.assign(threeptavg = fgpct)
newDataSet = newDataSet.assign(twoptavg = twopct)
newDataSet = newDataSet.assign(winpct = win14daypct)
newDataSet = newDataSet.assign(ftpct = ftpct)
newDataSet = newDataSet.assign(OffRebGame = oR)
newDataSet = newDataSet.assign(DefRebGame = dR)
newDataSet = newDataSet.assign(TOperGame = tO)


# In[536]:


newDataSet


# In[538]:


tourneyData = pd.read_csv('MDataFiles_Stage1/'+'MNCAATourneyDetailedResults.csv')
seasons = tourneyData['Season'].unique()
columnsWeCareAbout = ['adj_em','adj_o','luck','adj_tempo','sos_adj_em','sos_adj_o','sos_adj_d','nc_sos_adj_em',
                      'ncaa_seed','threeptavg','twoptavg','winpct','ftpct','OffRebGame','DefRebGame','TOperGame']
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
                                            'sos_adj_o','sos_adj_d','nc_sos_adj_em','ncaa_seed','threeptavg','twoptavg',
                                             'win14pct','ftpct','OffRebGame','DefRebGame','TO_game',
                                            'T2_adj_em','T2_adj_o','T2_luck','T2_adj_tempo','T2_sos_adj_em','T2_sos_adj_o',
                                             'T2_sos_adj_d','T2_nc_sos_adj_em','T2_ncaa_seed','T2_threeptavg',
                                             'T2_twoptavg','T2_win14pct','T2_ftpct','T2_OffRebGame','T2_DefRebGame','T2_TO_game',
                                             'Team1Win'])


temp.to_csv("modelData_v1.csv", index = False)


# In[ ]:





# In[539]:


mydata = pd.read_csv('modelData_v1.csv')


X= mydata.iloc[:,:-1] 
y = mydata.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .25)


# In[540]:


model1 = LogisticRegression()
model1.fit(X_train, y_train)


# In[541]:


print(model1.score(X_test,y_test))


# In[542]:


y_pred = model1.predict(X_test)
y_test1 = y_test.tolist()


# In[543]:


totalright = 0
for i in range(0, len(y_pred)):
    zero1 = 0
    if y_pred[i] > .5:
        zero1 = 1
    
    if y_test1[i] == zero1:
        totalright+=1

percentRight = totalright/len(y_pred)
print(percentRight)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




