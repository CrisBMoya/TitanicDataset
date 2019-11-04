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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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
X_train, X_test, y_train, y_test = train_test_split(TrainDF.drop(columns='Survived'), TrainDF['Survived'], test_size=0.2)

#GridSearch
Parameter_Grid={'max_depth':[1,2,4,6], 'n_estimators':[100,1000,10000]}
Grid_Search=GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=Parameter_Grid, verbose=3, refit=True)
Grid_Search.fit(X_train, y_train)
Grid_Search.best_params_
Grid_Search.best_estimator_

#Improve SVM model results
Grid_Predictions=Grid_Search.predict(X_test)
Grid_Predictions=Grid_Predictions >= 0.5
Grid_Predictions=Grid_Predictions.astype('int')

#Metrics
print(confusion_matrix(y_test, Grid_Predictions))
print(classification_report(y_test, Grid_Predictions))

########


#Predict and Submit
TestDF=pd.read_csv(zip.ZipFile('Data/titanic.zip').open("test.csv"))
PassID=TestDF['PassengerId']

#Delete non usefull columns
TestDF.drop(columns=['PassengerId','Name','Ticket'], inplace=True)

#Add Deck Column
#Add deck to data based on distribution
TestDF['Deck']='NA'
for i in range(0, TestDF.shape[0]):
    if TestDF['Cabin'].isna()[i]:
        DeckChoice=np.random.choice(a=SummaryCabin[SummaryCabin['Pclass']==TestDF['Pclass'][i]]['Deck'], size=1, p=SummaryCabin[SummaryCabin['Pclass']==TestDF['Pclass'][i]]['Prob']/100)
        TestDF.loc[i, 'Deck']=DeckChoice
    else:
        TestDF.loc[i, 'Deck']=TestDF.loc[i, 'Cabin']        
pass

#Add Deck
TestDF['Deck']=TestDF['Deck'].str.replace('\d+', '')
TestDF['Deck']=TestDF['Deck'].astype(str).str[0]

#Create dummy variables
TestDF=pd.concat([TestDF, pd.get_dummies(TestDF[['Sex','Embarked','Deck']])], axis=1)

#Delete old columns
TestDF.drop(columns=['Sex','Embarked','Deck'], inplace=True)

#Delete Cabin column
TestDF.drop(columns='Cabin', inplace=True)

#Imput Ages
Imp_v1_Test=SimpleImputer(missing_values=np.nan, strategy='mean')
Imp_v1_Test.fit(X=TestDF[['Age']])
Imput_Age_Test=Imp_v1_Test.transform(X=TestDF[['Age']])
TestDF['Age']=np.round(a=Imput_Age_Test, decimals=2)

#Manually handle stupid one entry in Fare with NA
TestDF.loc[TestDF['Fare'].isnull(),'Fare']=TestDF[TestDF['Embarked_S']==1]['Fare'].mean()

#Predict
Survivor_Predict=Grid_Search.predict(TestDF)
Survivor_Predict=Survivor_Predict >= 0.5
Survivor_Predict=Survivor_Predict.astype('int')

#Generate file
SubmitResults=pd.DataFrame(data={'PassengerId':PassID, 'Survived':Survivor_Predict})
SubmitResults.head()
SubmitResults.to_csv(path_or_buf='Python/Sub_V1.csv',index=False)

#Submit through API
import os
RE=True
if RE==True:
    os.system('kaggle competitions submit -c titanic -f Python/Sub_V1.csv -m "V1 Python Submission from API"')
pass