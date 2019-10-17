%config Completer.use_jedi = False

#Import Modules
import pandas as pd
import zipfile as zip
import numpy as np
import seaborn as sns
import pandas_profiling

#Load Training data
TrainDF=pd.read_csv(zip.ZipFile('Data/titanic.zip').open("train.csv"))
TrainDF.info()

#Generate a report
#TrainDF.profile_report().to_file(output_file="ProfileReport.html")

#See NA
sns.heatmap(TrainDF.isnull(), yticklabels=False, cbar=False, cmap='viridis')

#EDA -- Exploratory Data Analysis

#'fare' and 'cabin' must have a relationsip. You cannot possibly charge the same for different cabin categories (like first class)
#So 'pclass' probably has something to do with it too.
TrainDF[['Fare','Cabin','Pclass','Survived']]

sns.countplot(x="Pclass", hue="Survived", data=TrainDF)

#Subset only cabin with NA
OnlyNA=TrainDF[['Fare','Cabin','Pclass','Survived']]
OnlyNA=OnlyNA[pd.isna(OnlyNA['Cabin'])]

#Plot -- Most of the class 1 paid higher values, while class 2 and 3 paid less
#Probably the cabins follow the same patterns
sns.countplot(x='Pclass', hue='Survived', data=OnlyNA)
sns.barplot(x='Pclass', y='Fare', data=OnlyNA)
sns.barplot(x='Pclass', y='Fare', data=TrainDF)
sns.barplot(x='Cabin', y='Fare', data=TrainDF)

TrainDF[TrainDF['Pclass']==1]['Cabin'].to_csv('temp')

