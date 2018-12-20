#THe final project is about a banking campign to determine whether a customer will subsribe a term deposit.
#The whole project will be consists of four parts: EDA, decision tree model, factor analysis and SVM
##############################################
#############################################
library(ggplot2)
library(dplyr)
library(tidyr)
library(rpart) #classification and regression trees
library(partykit) #treeplots
library(randomForest) #random forests
library(gbm) #gradient boosting
library(caret) #tune hyper-parameter
library(rattle)
library(rpart.plot)
library(RColorBrewer)

#Loading data set
setwd("C:\\Users\\luoso\\Desktop")
banking <- read.csv('banking.csv')
dim(banking)
summary(banking)
banking$y = as.factor(banking$y)

age_loan <- banking %>%
  filter(loan == 'yes') %>%
  select(age)
summary(age_loan)
data.class(age_loan)

age_noloan <- banking %>%
  filter(loan == 'no') %>%
  select(age)
summary(age_noloan)

#############################
ggplot(banking,aes(x=age,y=count(age),fill = loan)) +
         geom_boxplot() + 
         facet_grid(.~ loan) +
         labs(title = 'age distribution for loan and no loan',x = 'age',y='count')
################################


age_primary <- banking %>% filter(education == 'primary') %>% select(age)
summary(age_primary)

age_secondary <- banking %>% filter(education == 'secondary') %>% select(age)
summary(age_secondary)

age_tertiary <- banking %>% filter(education == 'tertiary') %>% select(age)
summary(age_tertiary)

#EDA part
#1.plot age distribution with normal distribution curve.

ggplot(data = banking, aes(x=age)) + 
  geom_density(fill = "blue",alpha = 0.5) + 
  labs(title="Age Distribution",x="age",y="count") + 
  stat_function(fun = dnorm,args = list(mean = mean(banking$age, na.rm = TRUE), 
                                        sd = sd(banking$age, na.rm = TRUE)), color ='red',size = 1)
ggplot(banking,aes(job))+
  geom_bar() +
  labs(title="Job bar plot",x="Job type",y="amount")
#2.density plot on education by age
ggplot(data = banking, aes(x=age,fill = education)) +
  geom_density(alpha = 0.3) +
  facet_grid(.~ education)+
  labs(title='Education by age',x = 'age')

#3.Plot histogram on education level
ggplot(data = banking, aes(x=education)) +
  scale_color_manual(values=c("orange","pink","navy","olivedrab")) +
  geom_histogram(fill = c("orange","pink","navy","olivedrab"),alpha = 1,stat = "count")+
  labs(title = 'Education Level')

#4.age and balance with yes or no classification
ggplot(data = banking, aes(x=age,y=balance,color=y , shape = y)) +
  geom_point() + facet_grid(.~ y) + 
  labs(title = 'Age and Balance effect on deposit subscription')

#5.plot bar graph on relation between housing and success
ggplot(data = banking,aes(x=housing, fill = housing)) +
  geom_bar() + facet_grid(.~y) +
  labs(title = 'Relation between housing loan and subscription success')
#As we can see from the summary, most of variables are binary or categorical data inculding target varible.
#Next, I will form a decision tree model for prediction.

banking$y = ifelse(banking$y == 'yes',1,0)
head(banking$y)
#split into 70% training and 30% testing
setseed(12345)
intrain <- createDataPartition(y = banking$y,p=0.7,list = FALSE)
training <- banking[intrain,]
testing <- banking[-intrain,]

training$y = factor(training$y)

treemodel <- rpart(y~., data = training)
plotcp(treemodel)
print(treemodel$cptable)
fancyRpartPlot(treemodel)
printcp(treemodel)

cp_treemodel = min(treemodel$cptable[4,])

prune_treemodel <- prune(treemodel,cp = cp_treemodel)
fancyRpartPlot(prune_treemodel)
fancyRpartPlot(treemodel)

predict_treemodel <- predict(treemodel,newdata = testing)
rpart.resid = predict_treemodel - testing$y #calculate residuals
mean(rpart.resid^2) #caluclate MSE 0.4
#MSE is really quite large. I will try to use factor analysis and then supporting vector machine to get a higher accuracy


######################################
#Factor analysis
banking$age = as.numeric(as.factor(banking$age))
banking$job = as.numeric(as.factor(banking$job))
banking$marital = as.numeric(as.factor(banking$marital))
banking$education = as.numeric(as.factor(banking$education))
banking$default = as.numeric(as.factor(banking$default))
banking$balance = as.numeric(as.factor(banking$balance))
banking$housing = as.numeric(as.factor(banking$housing))
banking$loan = as.numeric(as.factor(banking$loan))
banking$contact = as.numeric(as.factor(banking$contact))
banking$day = as.numeric(as.factor(banking$day))
banking$month = as.numeric(as.factor(banking$month))
banking$duration = as.numeric(as.factor(banking$duration))
banking$campaign = as.numeric(as.factor(banking$campaign))
banking$pdays = as.numeric(as.factor(banking$pdays))
banking$previous = as.numeric(as.factor(banking$previous))
banking$poutcome = as.numeric(as.factor(banking$poutcome))
banking$y = as.numeric(as.factor((banking$y)))
banking_stand = as.data.frame(scale(banking))
### Factor analysis with no rotation
banking_factor = factanal(banking_stand, factors = 8, rotation = "none", na.action = na.omit)
banking_factor$loadings
#factors from 1 - 4 are rather important

bankloadings_fac1 = banking_factor$loadings[,1]
eigenv_fact1 = sum(bankloadings_fac1^2); eigenv_fact1
# Compute proportion variance
eigenv_fact1/17

banking_factor$uniquenesses

bankloadings_distant = banking_factor$loadings[1,]
bankcommunality_distant = sum(bankloadings_distant^2); bankcommunality_distant
unique_distant = 1-bankcommunality_distant; unique_distant

### Plot loadings against one another
load = banking_factor$loadings[,1:2]
plot(load, type="n") # set up plot 
text(load,labels=names(banking),cex=.7) # add variable names


banking_rotate = factanal(banking_stand, factors = 8, rotation = "varimax", na.action = na.omit)
banking_rotate$loadings
### Plot loadings against one another
load = banking_rotate$loadings[,1:2]
plot(load, type="n") # set up plot 
text(load,labels=names(banking),cex=0.7) 

###Make a subset
need_data <- c('age','housing','month','campaign','pdays','poutcome','y')
bank_subset <- banking[need_data]
bank_subset$y = as.factor(bank_subset$y)
######Supporting vector machine
set.seed(12345)
intrain <- createDataPartition(y = bank_subset$y, p= 0.7, list = FALSE)
training <- bank_subset[intrain,]
testing <- bank_subset[-intrain,]

dim(training); dim(testing);
                         
training$y

training$y = as.factor(training$y)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3233)
svm_Linear <- train(y ~., data = training, method = "svmLinear",
                    trControl=trctrl,
                    tuneLength = 2)

svm_Linear

test_pred <- predict(svm_Linear, newdata = testing)
test_pred

testing$y = factor(testing$y)
confusionMatrix(test_pred, testing$y)
