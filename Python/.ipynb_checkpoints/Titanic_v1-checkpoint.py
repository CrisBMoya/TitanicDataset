%config Completer.use_jedi = False

#Import Modules
import pandas as pd
import zipfile as zip
import numpy as np
import seaborn as sns
import pandas_profiling
import os
import re

#Set Working Directory
os.chdir(re.sub(pattern='Python', repl='', string=os.getcwd()))

#Load Training data
TrainDF=pd.read_csv(zip.ZipFile('Data/titanic.zip').open("train.csv"))
TrainDF.info()

#Generate a report
#TrainDF.profile_report().to_file(output_file="ProfileReport.html")

#See NA
sns.heatmap(TrainDF.isnull(), yticklabels=False, cbar=False, cmap='viridis')

#EDA -- Exploratory Data Analysis

#Get cabin letter only and then add class of passenger
ClassCabin=TrainDF[['Cabin','Pclass']].dropna()
ClassCabin=pd.DataFrame(ClassCabin['Cabin'].str.split(' ').tolist(), index=ClassCabin['Pclass']).stack()
ClassCabin=ClassCabin.reset_index([0,'Pclass'])
ClassCabin.columns=['Pclass','Cabin']
ClassCabin['Deck']=ClassCabin['Cabin'].str.replace('\d+', '')
ClassCabin.drop(columns='Cabin', inplace=True)

#Get count of cabin letter by class
SummaryCabin=ClassCabin.groupby(by=['Pclass','Deck']).size()
SummaryCabin=pd.DataFrame(SummaryCabin).reset_index()
SummaryCabin.columns=['Pclass','Deck','n']

#Compute Probabilities
SummaryCabin['Prob']=np.concatenate([SummaryCabin[SummaryCabin['Pclass']==X+1]['n']*100/int(SummaryCabin.groupby(by=['Pclass']).sum().iloc[X]) for X in range(0,3)])

