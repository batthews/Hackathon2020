# Importing libraries we will be using
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# build data we want for initial model, save to csv file
def buildNewCSV():
    # Grab season and team id columns
    seasonData = pd.read_csv('MRegularSeasonDetailedResults.csv')
    seasonList = seasonData['Season'].unique()
    teamList = seasonData['LTeamID'].unique()

    # Read Ken Pom data
    kenPomData = pd.read_csv('NCAA2020_Kenpom.csv')
    print(kenPomData.columns)

    # Create new data set with columns we want
    newDataSet = kenPomData.drop(['FirstD1Season','LastD1Season','adj_o_rank', 'adj_d_rank','adj_tempo_rank','luck_rank',
                                  'sos_adj_em_rank','sos_adj_d_rank','nc_sos_adj_em_rank','team','sos_adj_o_rank'], axis = 1)

    # Only include season after 2002, we don't have data on those beforehand in other file
    newDataSet = newDataSet[newDataSet.Season > 2002]
    print(newDataSet)

    tourneyData = pd.read_csv('MNCAATourneyDetailedResults.csv')
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

def runLogisticRegression():
    mydata = pd.read_csv('modelData_v1.csv')

    # X is all data minus last column, which is win or loss
    X= mydata.iloc[:,:-1]
    # Y is only last column, which is if 1st team won or loss, 1 if won, 0 if loss
    y = mydata.iloc[:,-1]
    # Split data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .25)

    # Run Logistic Regression
    model1 = LogisticRegression()
    model1.fit(X_train, y_train)

    y_pred = model1.predict(X_test)
    y_test1 = y_test.tolist()

    #print(y_test1)
    #print(y_pred)
    #print(y_test)

def percentRight():
    totalright = 0
    for i in range(0, len(y_pred)):
        zero1 = 0
        if y_pred[i] > .5:
            zero1 = 1

        if y_test1[i] == zero1:
            totalright+=1

    percentRight = totalright/len(y_pred)
    print(percentRight)

runLogisticRegression()