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
    #print(teamList)
    # Read Ken Pom data
    kenPomData = pd.read_csv('NCAA2020_Kenpom.csv')
    print(kenPomData.columns)

    # Create new data set with columns we want
    newDataSet = kenPomData.drop(['FirstD1Season','LastD1Season','adj_o_rank', 'adj_d_rank','adj_tempo_rank','luck_rank',
                                  'sos_adj_em_rank','sos_adj_d_rank','nc_sos_adj_em_rank','team','sos_adj_o_rank'], axis = 1)

    # Only include season after 2002, we don't have data on those beforehand in other file
    newDataSet = newDataSet[newDataSet.Season > 2002]
    #print(newDataSet)

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
    #print(trainingData)
    temp = pd.DataFrame(trainingData, columns = ['adj_em','adj_o','luck','adj_tempo','sos_adj_em',
                                                'sos_adj_o','sos_adj_d','nc_sos_adj_em','ncaa_seed',
                                                'T2_adj_em','T2_adj_o','T2_luck','T2_adj_tempo','T2_sos_adj_em','T2_sos_adj_o',
                                                 'T2_sos_adj_d','T2_nc_sos_adj_em','T2_ncaa_seed','Team1Win'])

    temp.to_csv("modelData_v1.csv", index = False)

def runLogisticRegression():
    mydata = pd.read_csv('modelData_v1.csv')
    global y_pred
    global y_test1
    # X is all data minus last column, which is win or loss
    X= mydata.iloc[:,:-1]
    # Y is only last column, which is if 1st team won or loss, 1 if won, 0 if loss
    y = mydata.iloc[:,-1]
    # Split data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .25)

    # Run Logistic Regression
    model1 = LogisticRegression()
    model1.fit(X_train, y_train)
    print(model1.score(X_test,y_test))
    y_pred = model1.predict(X_test)
    y_test1 = y_test.tolist()


    print(y_pred)
    print(y_test)

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

def buildSubmissionFile(year):
    # Grab season and team id columns
    seasonData = pd.read_csv('MRegularSeasonDetailedResults.csv')
    seasonList = seasonData['Season'].unique()
    teamList = seasonData['LTeamID'].unique()
    #print(teamList)
    # Read Ken Pom data
    kenPomData = pd.read_csv('NCAA2020_Kenpom.csv')
    #print(kenPomData.columns)

    # Create new data set with columns we want
    newDataSet = kenPomData.drop(['FirstD1Season','LastD1Season','adj_o_rank', 'adj_d_rank','adj_tempo_rank','luck_rank',
                                  'sos_adj_em_rank','sos_adj_d_rank','nc_sos_adj_em_rank','team','sos_adj_o_rank'], axis = 1)

    # Only include season after 2002, we don't have data on those beforehand in other file
    newDataSet = newDataSet[newDataSet.Season > 2002]
    newDataSet = newDataSet[newDataSet.Season < year]
    #print(newDataSet)

    tourneyData = pd.read_csv('MNCAATourneyDetailedResults.csv')
    tourneyData = tourneyData[tourneyData.Season < year]
    #print(tourneyData)
    seasons = tourneyData['Season'].unique()
    #print(seasons)
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
    #print(trainingData)
    temp = pd.DataFrame(trainingData, columns = ['adj_em','adj_o','luck','adj_tempo','sos_adj_em',
                                                'sos_adj_o','sos_adj_d','nc_sos_adj_em','ncaa_seed',
                                                'T2_adj_em','T2_adj_o','T2_luck','T2_adj_tempo','T2_sos_adj_em','T2_sos_adj_o',
                                                 'T2_sos_adj_d','T2_nc_sos_adj_em','T2_ncaa_seed','Team1Win'])

    temp.to_csv("modelData_v1."+str(year)+".csv", index = False)

def buildPredictionFile(matchups, year):
    # Grab season and team id columns
    seasonData = pd.read_csv('MRegularSeasonDetailedResults.csv')
    seasonList = seasonData['Season'].unique()
    teamList = seasonData['LTeamID'].unique()
    #print(teamList)
    # Read Ken Pom data
    kenPomData = pd.read_csv('NCAA2020_Kenpom.csv')
    #print(kenPomData.columns)

    # Create new data set with columns we want
    newDataSet = kenPomData.drop(['FirstD1Season','LastD1Season','adj_o_rank', 'adj_d_rank','adj_tempo_rank','luck_rank',
                                  'sos_adj_em_rank','sos_adj_d_rank','nc_sos_adj_em_rank','team','sos_adj_o_rank'], axis = 1)
    #print(newDataSet)
    # Only include season from year in our dataset
    newDataSet = newDataSet[newDataSet.Season < (year + 1)]
    newDataSet = newDataSet[newDataSet.Season > (year - 1)]
    #print(newDataSet)
    #print(newDataSet)
    tourneyData = pd.read_csv('MNCAATourneyDetailedResults.csv')
    tourneyData = tourneyData[tourneyData.Season < (year + 1)]
    tourneyData = tourneyData[tourneyData.Season > (year - 1)]
    seasons = tourneyData['Season'].unique()
    columnsWeCareAbout = ['adj_em','adj_o','luck','adj_tempo','sos_adj_em','sos_adj_o','sos_adj_d','nc_sos_adj_em','ncaa_seed']
    trainingData = []
    for i in range(0,len(matchups)):
        team1 = (newDataSet.loc[((newDataSet.TeamID == matchups[i][0])), columnsWeCareAbout ].values[0])
        team2 = (newDataSet.loc[((newDataSet.TeamID == matchups[i][1])), columnsWeCareAbout ].values[0])
        # Submit
        #print(team1)
        featureList = np.concatenate([team1,team2])
        trainingData.append(featureList)
    #print(trainingData)
    temp = pd.DataFrame(trainingData, columns = ['adj_em','adj_o','luck','adj_tempo','sos_adj_em',
                                                'sos_adj_o','sos_adj_d','nc_sos_adj_em','ncaa_seed',
                                                'T2_adj_em','T2_adj_o','T2_luck','T2_adj_tempo','T2_sos_adj_em','T2_sos_adj_o',
                                                 'T2_sos_adj_d','T2_nc_sos_adj_em','T2_ncaa_seed'])
    #print(temp)
    temp.to_csv("modelData_v1."+str(year)+"pred.csv", index = False)

def prediction(matchups, year):
    # Past tournament data to train our model
    mydata = pd.read_csv('modelData_v1.'+str(year)+'.csv')

    # X is all data minus last column, which is win or loss
    X= mydata.iloc[:,:-1]
    # Y is only last column, which is if 1st team won or loss, 1 if won, 0 if loss
    y = mydata.iloc[:,-1]

    predictionData = pd.read_csv('modelData_v1.'+str(year)+'pred.csv')
    #print(predictionData)
    # Run Logistic Regression
    model1 = LogisticRegression()
    model1.fit(X, y)
    #print(model1.score(X_test,y_test))
    predictions = model1.predict_proba(predictionData)
    #y_test1 = y_test.tolist()
    #model1.score(predictionData,predictions)
    #print(predictions)
    submissionList = []
    for i in range(0,len(matchups)):
        submissionList.append([str(year)+'_'+str(matchups[i][0])+'_'+str(matchups[i][1]),predictions[i][1]])
    #print(submissionList)
    # Turn 2d array into datafram to make into csv
    submissionDF = pd.DataFrame(submissionList, columns=['ID', 'Pred'])
    submissionDF.to_csv("SubmissionStage1_"+str(year)+".csv", index = False)

    #print(y_pred)
    #print(y_test)

def matchUps(year):
    tourneyData = pd.read_csv('MNCAATourneySeeds.csv')
    tourneyData = tourneyData[tourneyData.Season > year - 1]
    tourneyData = tourneyData[tourneyData.Season < year + 1]

    #tourneyData.sort_values(by=['TeamID'])
    teamIds = tourneyData['TeamID']
    teamIds = teamIds.sort_values()
    matchups = []
    for i in range (0, len(teamIds)-1):
        for j in range(i, len(teamIds)-1):
            matchups.append([teamIds.values[i],teamIds.values[j+1]])
    #print(matchups)
    #print(len(matchups))
    return matchups

def combineToMakeSubmissionFile():
    sub2015 = pd.read_csv('SubmissionStage1_2015.csv')
    sub2016 = pd.read_csv('SubmissionStage1_2016.csv')
    sub2017 = pd.read_csv('SubmissionStage1_2017.csv')
    sub2018 = pd.read_csv('SubmissionStage1_2018.csv')
    sub2019 = pd.read_csv('SubmissionStage1_2019.csv')
    submissionFinal = pd.concat([sub2015,sub2016,sub2017,sub2018,sub2019], ignore_index=True)
    del submissionFinal['Unnamed: 0']
    print(submissionFinal)
    submissionFinal.to_csv("SubmissionFinal.csv", index = False)
#buildNewCSV()
#runLogisticRegression()
#buildSubmissionFile()
#percentRight()
#prediction2015()
#buildPredictionFile()
'''
for i in range(2015,2019):
    buildSubmissionFile(i)
    matchups = matchUps(i)
    buildPredictionFile(matchups, i)
    prediction(matchups, i)'''
combineToMakeSubmissionFile()