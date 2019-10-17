rm(list=ls())

library(tidyverse)
library(ggplot2)
library(Amelia)
library(corrplot)
library(mice)
library(gbm)
library(dummies)
library(caret)
library(e1071)
#

setwd(gsub(pattern='Documents', replacement='Google Drive/Github/TitanicDataset', x=getwd()))

#Set seed
set.seed(101)

#Read data
TrainDF=read_delim(file=unz(description='Data/titanic.zip', filename='train.csv'), delim=',')

#Plot NA
missmap(TrainDF)

#Get cabin letter only and then add class of passenger
CabinTemp=strsplit(x=TrainDF$Cabin, split=' ')
ClassCabin=data.frame('Pclass' = rep(x=TrainDF$Pclass, sapply(X=CabinTemp, FUN=length)), 'Cabin' = unlist(CabinTemp))
ClassCabin$Deck=gsub(pattern='[[:digit:]]', replacement='', x=ClassCabin$Cabin)

#Get count of cabin letter by class
SummaryCabin=table(ClassCabin[,c('Pclass','Deck')])
SummaryCabin=as_tibble(SummaryCabin)
SummaryCabin=SummaryCabin[order(SummaryCabin$Pclass),]
ggplot(data=SummaryCabin, aes(x=Deck, y=n, fill=Pclass)) + geom_col(position='dodge2')

#Probability
SummaryCabin$Prob=0.0
SummaryCabin[SummaryCabin$Pclass==1,]$Prob=SummaryCabin[SummaryCabin$Pclass==1,]$n*100/sum(SummaryCabin[SummaryCabin$Pclass==1,]$n)
SummaryCabin[SummaryCabin$Pclass==2,]$Prob=SummaryCabin[SummaryCabin$Pclass==2,]$n*100/sum(SummaryCabin[SummaryCabin$Pclass==2,]$n)
SummaryCabin[SummaryCabin$Pclass==3,]$Prob=SummaryCabin[SummaryCabin$Pclass==3,]$n*100/sum(SummaryCabin[SummaryCabin$Pclass==3,]$n)

DeckFill=list()
for(i in 1:length(TrainDF$Pclass)){
  if(is.na(TrainDF[i,]$Cabin)){
    DeckFill[i]=sample(x=c(SummaryCabin[SummaryCabin$Pclass==TrainDF$Pclass[i],]$Deck), size=1, 
      prob=c(SummaryCabin[SummaryCabin$Pclass==TrainDF$Pclass[i],]$Prob))
  }else{
    DeckFill[i]=TrainDF[i,]$Cabin
  }
}
TrainDF$Deck=unlist(DeckFill)
TrainDF$Deck=substr(x=TrainDF$Deck, start=1, stop=1)

CheckDist=as_tibble(table(TrainDF[,c('Pclass','Deck')]))

#Check distribution
CabinDist=SummaryCabin
CabinDist$Pclass=paste0('Class ', SummaryCabin$Pclass,' Old')
CheckDist$Pclass=paste0('Class ', CheckDist$Pclass,' new')
CabinDist$Source='Old'
CheckDist$Source='New'
CheckDist=bind_rows(CabinDist, CheckDist)

#Distributions of imputed data are the same, just on a greater scale
ggplot(data=CheckDist, aes(x=Deck, y=n, group=Pclass, color=Source)) + geom_line() +
  facet_wrap(facets=~Pclass, scales='free_y', nrow=3, ncol=2)

#Check NA again -- remove cabin as we left only the deck letter
TrainDF$Cabin=NULL
missmap(TrainDF)

#Input age using mice
AgeInput=mice(data=TrainDF[,which(sapply(TrainDF, class)=='numeric')], method='rf')

#Check distribution
DistPlotAge=bind_rows(data.frame('S'='Original', table(TrainDF$Age), stringsAsFactors=FALSE), 
  data.frame('S'='Imput',table(AgeInput$imp$Age$`1`), stringsAsFactors=FALSE))
ggplot(data=DistPlotAge, aes(x=Var1, y=Freq, fill=S)) + geom_col(color='black', alpha=.5)

#Add Age NA
which(is.na(TrainDF$Age))==as.numeric(rownames(AgeInput$imp$Age))
TrainDF[is.na(TrainDF$Age),]$Age=AgeInput$imp$Age$`1`

#Check NA
missmap(TrainDF)

#Theres 1 NA left, just delete
TrainDF=na.omit(TrainDF)

#Delete the one 'T' example
TrainDF=TrainDF[TrainDF$Deck!='T',]

#Delete ID
TrainDF$PassengerId=NULL

#Delete Names and tickets as they may not have meaning -- they do, but is hard to get info from them
TrainDF$Name=NULL
TrainDF$Ticket=NULL

#Create Dummy for Sex
TrainDF=bind_cols(TrainDF, as.data.frame(dummy(x=TrainDF$Sex)))
TrainDF$Sex=NULL

#Create dummy for Embarked
TrainDF=bind_cols(TrainDF, as.data.frame(dummy(x=TrainDF$Embarked)))
TrainDF$Embarked=NULL

#Create dummy for Deck
TrainDF=bind_cols(TrainDF, as.data.frame(dummy(x=TrainDF$Deck)))
TrainDF$Deck=NULL

#Split data
RowPart=createDataPartition(y=TrainDF$Survived, times=1, p=0.2, list=FALSE)
TestSet=TrainDF[RowPart,]
Trainset=TrainDF[-RowPart,]

#GridSearch
GridSearch=expand.grid(distribution=c('gaussian','bernoulli','adaboost'), 
  n.trees=c(100,1000,10000),
  interaction.depth=c(1,2,4,6), stringsAsFactors=FALSE)

ResultsDist=list()
for(i in 1:nrow(GridSearch)){
  print(paste0('Dist ', i,' of ', nrow(GridSearch)))
  
  #Train model
  GBMModel=gbm(formula=Survived ~ ., data=Trainset, 
    n.trees=GridSearch[i,]$n.trees, 
    distribution=GridSearch[i,]$distribution, 
    interaction.depth=GridSearch[i,]$interaction.depth)
  
  #Predict values
  TreesPred=seq(from=100, to=100000, by=100)
  Predicted=predict(object=GBMModel, newdata=TestSet, n.trees=TreesPred)
  Predicted=as.data.frame(Predicted)
  
  TreeResult=unlist(lapply(Predicted, function(x){
    Res=confusionMatrix(factor(ifelse(test=x>=0.5, yes=1, no=0)),
      factor(TestSet$Survived))
    Res$overall['Accuracy']
  }))
  Chosen=gsub(pattern='\\..*', replacement='', x=names(which(TreeResult==max(TreeResult))[1]))
  
  #Other metrics
  ResultsDist[[paste0('Dist = ',GridSearch[i,]$distribution,'; nTree = ',GridSearch[i,]$n.trees,
    '; IntDepth = ',GridSearch[i,]$interaction.depth)]]=c(confusionMatrix(
      factor(ifelse(test=Predicted[Chosen]>=0.5, yes=1, no=0)),
      factor(TestSet$Survived), mode='prec_recall'),
      postResample(pred=ifelse(test=Predicted[Chosen]>=0.5, yes=1, no=0), obs=TestSet$Survived))
}

TEMP=lapply(ResultsDist, function(x){
  c('Accuracy'=x[['overall']]['Accuracy'],
    'Recall'=x[['byClass']]['Recall'],
    'RMSE'=x[['RMSE']]
  )
})
TEMP=ldply(TEMP)
which(TEMP$RMSE==min(TEMP$RMSE))
which(TEMP$Accuracy.Accuracy==max(TEMP$Accuracy.Accuracy))
which(TEMP$Recall.Recall==max(TEMP$Recall.Recall))
DF=data.frame('RMSE'=cut(x=TEMP$RMSE, breaks=4, labels=1:4),
  'Accuracy'=cut(x=TEMP$Accuracy.Accuracy, breaks=4, labels=1:4),
  'Recall'=cut(x=TEMP$Recall.Recall, breaks=4, labels=1:4), stringsAsFactors=FALSE)
DF=as.data.frame(sapply(X=DF, FUN=function(x){as.numeric(x)}))
DF$ID=1:nrow(DF)
DF[order(-DF$RMSE, DF$Accuracy, DF$Recall, decreasing=TRUE),]
TEMP[13,]

#Train model
GBMModel=gbm(formula=Survived ~ ., data=Trainset, n.trees=100000, distribution='bernoulli', interaction.depth=6)
summary(GBMModel)

#Predict and submit
#Read Test
TestDF=read_delim(file=unz(description='Data/titanic.zip', filename='test.csv'), delim=',')

#Delete Names and tickets as they may not have meaning -- they do, but is hard to get info from them
TestDF$Name=NULL
TestDF$Ticket=NULL

#Create Dummy for Sex
TestDF=bind_cols(TestDF, as.data.frame(dummy(x=TestDF$Sex)))
TestDF$Sex=NULL

#Create dummy for Embarked
TestDF=bind_cols(TestDF, as.data.frame(dummy(x=TestDF$Embarked)))
TestDF$Embarked=NULL

DeckFill=list()
for(i in 1:length(TestDF$Pclass)){
  if(is.na(TestDF[i,]$Cabin)){
    DeckFill[i]=sample(x=c(SummaryCabin[SummaryCabin$Pclass==TestDF$Pclass[i],]$Deck), size=1, 
      prob=c(SummaryCabin[SummaryCabin$Pclass==TestDF$Pclass[i],]$Prob))
  }else{
    DeckFill[i]=TestDF[i,]$Cabin
  }
}
TestDF$Deck=unlist(DeckFill)
TestDF$Deck=substr(x=TestDF$Deck, start=1, stop=1)

#Create dummy for Deck
TestDF=bind_cols(TestDF, as.data.frame(dummy(x=TestDF$Deck)))
TestDF$Cabin=NULL

#Change colnames so they fit to the ones in the model
colnames(TestDF)=gsub(pattern='Test', replacement='Train', x=colnames(TestDF))
colnames(TestDF)
colnames(TrainDF)

#Predict on test dataset
SubPred=as.data.frame(predict(object=GBMModel, newdata=TestDF, n.trees=TreesPred))
SubPred=ifelse(test=SubPred[Chosen]>=0.5, yes=1, no=0)
nrow(TestDF)
table(SubPred)
length(SubPred)

#Save
SubV=data.frame('PassengerId'=TestDF$PassengerId, 'Survived'=SubPred[,1])
write_delim(x=SubV, path='R/Sub_V3.csv', delim=',')

#Submit
RE=TRUE
if(RE){
  print('WARNING: A file will be uploaded!')
  list.files(path='R/')
  Sys.sleep(5)
  
  system('kaggle competitions submit -c titanic -f R/Sub_V3.csv -m "V3 Submission from API"')
}
