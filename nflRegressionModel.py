from discord.ext import commands
import datetime
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import nfl_data_py as nfl
import os
import urllib.request
import matplotlib.pyplot as pit
from matplotlib.offsetbox import AnnotationBbox
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
    
def getPlayerStats(playerName, weekNum, stat, year=datetime.datetime.now().year):
    # weekNum must be in form of num
    # year is optional, default is current year
    cleanWeek = int(weekNum)
    cleanName = playerName.title()
    weekly = nfl.import_weekly_data([int(year)])
    df = pd.DataFrame(weekly)
    playerData = df[df['player_display_name'] == cleanName]
    playerWeekData = playerData[playerData['week'] == cleanWeek]
    playerWeekStatData = playerWeekData[stat].to_string(index=False)

    return playerWeekStatData


def main():

    #pbp = nfl.import_pbp_data([2022])
    weekly = nfl.import_weekly_data([2022])
    yearly = nfl.import_seasonal_data([2022])
    weeklyCol = weekly.columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(weekly)
    dfSeason = pd.DataFrame(yearly)
    playerSeasonData = df[df['player_display_name'] == 'Mike Evans']

    playerData = df[df['player_display_name'] == 'Mike Evans']
    playerWeekData = playerData[playerData['week'] == 3]
    playerOpponent = playerWeekData['opponent_team'].to_string(index=False)
    #print(playerOpponent)
    #print(dfSeason.columns)
    #print(playerWeekData['position'].to_string(index=False))
    # playerName = "Jahmyr Gibbs"
    # cleanName = playerName.title()
    # playerData = df[df['player_display_name'] == cleanName]
    # print(playerData['headshot_url'])

    #playerImage = playerWeekData['headshot_url']
    #print(playerImage)
    # cleanTargetShare = playerWeekData['target_share'] * 100
    # cleanShare = cleanTargetShare.to_string(index=False)
 

    X = df[['attempts', 'interceptions', 'carries', 'targets', 'rushing_tds', 'receiving_tds' ]]
    #X = df['receiving_yards', 'fantasy_points_ppr']
    Y = df['fantasy_points_ppr']

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(Xtrain, Ytrain)
    Ypred = model.predict(Xtest)

    mse = mean_squared_error(Ytest, Ypred)
    r_squared = r2_score(Ytest, Ypred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r_squared}')
    print("Coefficients: ", model.coef_)
    print("Intercept: ", model.intercept_)
    print("Y predict: ", Ypred)
    

    # printInstructions()
    # print(weeklyCol)
    # mahomesPass = getPlayerStats('patrick mahomes', 2, 'passing_yards')
    # print(mahomesPass)
    #printStatChoices()


    # cleanWeek = int(weekNum)
    # cleanYear = int(year)
    # cleanName = playerName.title()
    # weekly = nfl.import_weekly_data([int(cleanYear)])
    # df = pd.DataFrame(weekly)
    # playerData = df[df['player_display_name'] == cleanName]
    # playerWeekData = playerData[playerData['week'] == cleanWeek]
    # playerWeekStatData = playerWeekData[stat].to_string(index=False)
    # playerImage = playerData['headshot_url']

    #print(playerWeekData['position'].to_string(index=False))
    #print(playerData)
    #print(cleanShare)

   
    
    #print(weekElevenDf)
    # filteredDf = df[df['player_display_name'] == 'Patrick Mahomes']
    # passing_yards = filteredDf['passing_yards']
    # print(passing_yards)
    # print(df[df['player_display_name'] == 'Patrick Mahomes'])
    #print(df)
    # pfile = "output.txt"
    # with open(pfile, "w+") as printer:
    #     print(weeklyCol, file=printer)
    


if __name__ == "__main__":
    main()