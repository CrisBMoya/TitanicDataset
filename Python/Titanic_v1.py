%config Completer.use_jedi = False

#Import Modules
import pandas as pd
import zipfile as zip
import numpy as np
import seaborn as sns
import pandas_profiling
import os
import re
from sklearn.impute import SimpleImputer

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

#Plot
sns.barplot(x=SummaryCabin['Deck'], y=SummaryCabin['n'], hue=SummaryCabin['Pclass'])

#Compute Probabilities
SummaryCabin['Prob']=np.concatenate([SummaryCabin[SummaryCabin['Pclass']==X+1]['n']*100/int(SummaryCabin.groupby(by=['Pclass']).sum().iloc[X]) for X in range(0,3)])

#Add deck to data based on distribution
TrainDF['Deck']='NA'
for i in range(0, TrainDF.shape[0]):
    if TrainDF['Cabin'].isna()[i]:
        DeckChoice=np.random.choice(a=SummaryCabin[SummaryCabin['Pclass']==TrainDF['Pclass'][i]]['Deck'], size=1, p=SummaryCabin[SummaryCabin['Pclass']==TrainDF['Pclass'][i]]['Prob']/100)
        TrainDF.loc[i, 'Deck']=DeckChoice
    else:
        TrainDF.loc[i, 'Deck']=TrainDF.loc[i, 'Cabin']        
pass

#Add Deck
TrainDF['Deck']=TrainDF['Deck'].str.replace('\d+', '')
TrainDF['Deck']=TrainDF['Deck'].astype(str).str[0]

#Imput Ages
Imp_v1=SimpleImputer(missing_values=np.nan, strategy='mean')
Imp_v1.fit(X=TrainDF[['Age']])
Imput_Age=Imp_v1.transform(X=TrainDF[['Age']])
TrainDF['Age']=Imput_Age

#Delete Cabin column
TrainDF.drop(columns='Cabin', inplace=True)

#Delete that one entry in embarked
TrainDF.dropna(inplace=True)

#Delete the one 'T' deck row
TrainDF.drop(labels=TrainDF[TrainDF['Deck']=='T'].index, inplace=True)

#Missmap
sns.heatmap(TrainDF.isnull(), yticklabels=False, cbar=False, cmap='viridis')

#Delete non usefull columns
TrainDF.drop(columns=['PassengerId','Name','Ticket'], inplace=True)

#Create dummy variables
TrainDF=pd.concat([TrainDF, pd.get_dummies(TrainDF[['Sex','Embarked','Deck']])], axis=1)

#Delete old columns
TrainDF.drop(columns=['Sex','Embarked','Deck'], inplace=True)

#Split Data
