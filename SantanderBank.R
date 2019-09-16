rm(list=ls())

###########################Setting up the working directory and impoting the required dataset###########################
setwd('D:/Study/R programs')

train = read.csv('train.csv', sep = ',', header =  TRUE, na.strings = c("", " ", 'NA'))

###########################Loading all the required libraries###########################
libraries = c("plyr","dplyr","ggplot2", "rpart","DMwR", "randomForest", "usdm", "corrgram", "DataCombine", "sp", "raster", "usdm", "ggplot2")
lapply(X = libraries, require, character.only = TRUE)

###########################Exploratory data analysis###########################
summary(train)
head(train)
dim(train)
str(train)
names(train)

###########################Check for missing values###########################
sum(is.na(train))

###########################Feature Engineering###########################
cnames <- colnames(train[3:ncol(train)])
for(i in cnames){
  train$i <- as.factor(train$i)
  
}

###########################Check to see the distribution of the variables using graphs###########################
par(mfrow = c(4,50))

for(i in cnames){
  #print(i)
  #ggplot(data = train, aes(x = i))+geom_histogram(fill='#E69F00', bins = 25)+geom_tile("Disttribution plot of ")
  ggplot(data = train, aes(x = i))+geom_histogram(fill = '#999999', bins = 25) + ggtitle("Variable Distribution")
  
}

#ggplot(data = train, aes(x = var_0))+geom_histogram(fill = '#999999', bins = 25) + ggtitle("Temperature Distribution")

###########################Check for outliers using boxplot###########################
for (i in 3:length(cnames))
   {
     assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "responded"), data = subset(train))+ 
              stat_boxplot(geom = "errorbar", width = 0.5) +
              geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                           outlier.size=1, notch=FALSE) +
              theme(legend.position="bottom")+
              labs(y=cnames[i],x="responded")+
              ggtitle(paste("Box plot of responded for",cnames[i])))
   }

###########################Check for collinear variables using correlation graph###########################
corrgram(train, order = F, upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

train <- subset(train, select = -c(ID_code))
rmExcept(keepers = "train")

###########################Logistic Regression###########################
#Step 1: Divide the data into train and test sets
set.seed(123)

train_index =  sample(1:nrow(train), 0.8*nrow(train))
train_set = train[train_index,]
test_set = train[-train_index,]

library(ISLR)

logit_model  = glm(target ~., data =train_set, family = "binomial")
logit_pred = predict(logit_model, newdata = test_set, type = 'response')

logit_pred = ifelse(logit_pred > 0.5, 1, 0)

###########################Error Metrics###########################
ConfMatrix_logit = table(test_set$target, logit_pred)

Accuracy = sum(diag(ConfMatrix_logit))/nrow(test_set) #91.4
Precision = sum(diag(ConfMatrix_logit))/(482+1046) #23.94
Recall = sum(diag(ConfMatrix_logit))/(482+2925) #10.74

F1 = 2*((Precision*Recall)/(Precision+Recall)) #14.82
FNR = 2925/(2925+1046) #31.5

###########################Decision Tree Model###########################
#Sampling the data using stratified sampling since the data is very big
table(train$target)
library(survival)
library(optmatch)
library(raster)
library(sp)
library(nlme)

x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

install.packages(x)
lapply(x, require, character.only = TRUE)
#0 = 179902 1= 20098
train = train[order('target'),]
stratas = sampling:::strata(train, stratanames = 'target', size = c(17990, 2010), method = 'srswor')
table(stratas)
stratified_data = getdata(train,stratas)

stratified_data = subset(stratified_data, select = -c(ID_code,ID_unit, Prob, Stratum))

train_index =  sample(1:nrow(stratified_data), 0.8*nrow(stratified_data))
train_set = train[train_index,]
test_set = train[-train_index,]

train_set$target = factor(x = train_set$target, levels = c(0,1), labels = c('No', 'Yes'))
test_set$target = factor(x = test_set$target, levels = c(0,1), labels = c('No', 'Yes'))

DT_model = C5.0(target ~., train_set, trials = 100, rules  = TRUE)

DT_predictions = predict(DT_model, test_set[2:201])

ConfMatrix_dt = table(test_set$target, DT_predictions)

###########################Naive Bayes Model###########################
NB_model = naiveBayes(target ~., data = train_set)

NB_predictions = predict(NB_model, test_set[,2:201])

Conf_matrix_NB = table(observed = test_set$target, predicted = NB_predictions)

Accuracy = sum(diag(Conf_matrix_NB))/nrow(test_set) #91.9
Precision = sum(diag(Conf_matrix_NB))/(6380+2669) #18.69
Recall = sum(diag(Conf_matrix_NB))/(6360+12176) #9.12
