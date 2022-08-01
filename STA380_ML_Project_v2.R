#STA380 Project 
#Group Members - Apurva Audi, James Anderson, Sameer Ahmed, Amanda Nguyen, Shubhada Kapre
# Modelling

# 1) Single Variable Logistic Regression
#Creation of Training and Test Sets

df_inv = read.csv(file = '/Users/SameerAhmed/Documents/GitHub/vcsuccess/StartupSuccessOverSampled.csv')

data_set_size = nrow(df_inv)
print(data_set_size)
test_set_size = round(0.30*data_set_size)
print(test_set_size)
RNGkind(sample.kind = "Rounding")
set.seed(8239)
tickets = sample(data_set_size,test_set_size)
df_Test = df_inv[tickets,]
df_Train = df_inv[-tickets,]
df_Train <- na.omit(df_Train)
df_Test <- na.omit(df_Test)

logistic_fit= glm(status_label ~ funding_rounds, data=df_Train,family='binomial')
summary(logistic_fit)

logistic_fit= glm(status_label ~ total_investment, data=df_Train,family='binomial')
summary(logistic_fit)

logistic_fit= glm(status_label ~ venture, data=df_Train,family='binomial')
summary(logistic_fit)

logistic_fit= glm(status_label ~ seed, data=df_Train,family='binomial')
summary(logistic_fit)

logistic_fit= glm(status_label ~ diff_funding_year, data=df_Train,family='binomial')
summary(logistic_fit)

#Attempt to fit probability vs eta. Our inability to do so made us scrap this model
#from our final project. It is clear here that there isn't a properly defined eta,
#and the graph is interpreting the code as something to draw a line graph of. 
z = seq(from=-5,to=5,length.out=1000)
Fz = exp(z)/(1+exp(z))
plot(z,Fz,type='l',col='blue',lwd=2.5,xlab=expression(eta),ylab='F',cex.lab=1.3)
eta = predict(logistic_fit)
pyx = predict(logistic_fit,type='response')
plot(eta,pyx,type='l',col='blue',lwd=2.5,xlab=expression(eta),ylab='F',cex.lab=1.3)

# 2) Lasso Regression, Ridge Regression, PCR and Subset Selection

# lasso regression 
library(readr)

data_set_size = nrow(df_inv)
test_set_size = round(0.30*data_set_size)

df_inv$status = as.factor(df_inv$status)

df_inv$region = as.factor(df_inv$region)
df_inv$status = as.factor(df_inv$status)

lm.fit = lm(status_label~  total_investment_cde + diff_funding_year_cde + funding_rounds_cde + seed_cde + venture_cde, data = df_inv)
summary(lm.fit)

library(glmnet)
set.seed(1)
library(dplyr)
train = sample(1:nrow(df_inv),nrow(df_inv)/2)
test=(-train)
y.test = y[test]

x = model.matrix(status_label ~ + market_type + total_investment_cde + diff_funding_year_cde + funding_rounds_cde + seed_cde + venture_cde, data = df_inv)[, -1]
y = df_inv$status_label + df_inv$total_investment_cde + df_inv$diff_funding_year_cde + df_inv$funding_rounds_cde + df_inv$seed_cde + df_inv$venture_cde + df_inv$market_type

cv.out = cv.glmnet(x[train, ], y[train], alpha = 1)
plot(cv.out)

cv.out$lambda.min

lasso.startup = glmnet(x[train,], y[train], alpha = 1, thresh = 1e-12, lambda = grid)
grid = 10^seq(10, -2, length = 100)
plot(lasso.startup)
summary(lasso.startup)
lasso.pred = predict(lasso.startup, s=cv.out$lambda.min, newx = x[test, ])
sqrt(mean((lasso.pred - y.test)^2))


out = glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef = predict(out, type = 'coefficients', s=cv.out$lambda.min)[1:7,]
lasso.coef

# ridge regression 
ridge.mod = glmnet(x[train,], y[train], alpha = 0, lambda = grid, thresh = 1e-12)
plot(ridge.mod)
dim(coef(ridge.mod))
sqrt(sum(coef(ridge.mod)[-1, 50]^2))

ridge.mod$lambda[60]
coef(ridge.mod)[, 60]
sqrt(sum(coef(ridge.mod)[-1, 60]^2))

cv.out = cv.glmnet(x[train,], y[train], alpha = 0)
plot(cv.out)
cv.out$lambda.1se
cv.out$lambda.min
ridge.pred = predict(cv.out, s = cv.out$lambda.min, newx = x[test, ], type = 'coefficients')[1:7,]
sqrt(mean((ridge.pred - y.test)^2))
ridge.pred
# 3) Logistic Regression

#Reading in the Data
##needs to be re-read 
df_inv = read.csv(file = '/Users/SameerAhmed/Documents/GitHub/vcsuccess/StartupSuccessOverSampled.csv')

#changing it to two factor (re run this again)
df_inv$status[df_inv$status == 'acquired'] <- 'success' 
df_inv$status[df_inv$status == 'operating'] <- 'success'

df_inv$status = as.factor(df_inv$status)
df_inv$region = as.factor(df_inv$region)
df_inv$status = as.factor(df_inv$status)

max(df_inv$total_investment)
which(df_inv$total_investment == 30079503000)

#removing Verizon
df_inv = df_inv[-c(11243), ]


##getting test data 
data_set_size = nrow(df_inv) 
test_set_size = round(0.30*data_set_size) #I want a 20% validation set.
RNGkind(sample.kind = "Rounding")
set.seed(8239) 
tickets = sample(data_set_size,test_set_size)
df_inv_Test = df_inv[tickets,]
df_inv_Train = df_inv[-tickets,]


#regsubset
library(leaps)
build = regsubsets(status ~ diff_funding_year + market_type + funding_rounds + total_investment_cde + seed_label + venture_label + funding_rounds_label, data = df_inv_Train, method = "backward", nvmax=6)
plot(build, scale="r2")

#different variable choice
logistic_fit= glm(status ~ funding_rounds + total_investment_cde,data=df_inv_Train,family='binomial')
summary(logistic_fit)

logistic_fit= glm(status ~ region + funding_rounds + total_investment, data=df_inv_Train,family='binomial')
summary(logistic_fit)

logistic_fit= glm(status ~  funding_rounds  + seed_label + venture_label + funding_rounds_label,data=df_inv_Train,family='binomial')
summary(logistic_fit)

logistic_fit= glm(status ~ region + funding_rounds + total_investment + seed_label + venture_label + funding_rounds_label,data=df_inv_Train,family='binomial')
summary(logistic_fit)

logistic_fit= glm(status ~ region + funding_rounds + total_investment + seed_label + venture_label + funding_rounds_label,data=df_inv_Train,family='binomial')
summary(logistic_fit)

logistic_fit= glm(status ~ diff_funding_year + market_type + funding_rounds + seed_label + total_investment_cde,data=df_inv_Train,family='binomial')
summary(logistic_fit)



prob_success_pred=predict(logistic_fit,df_inv_Test,type="response")

pred_success = ifelse(prob_success_pred>.55,"success","closed")
table(df_inv_Test$status, pred_success)

install.packages('caret')
library(caret)

confusionMatrix(as.factor(df_inv_Test$status), as.factor(pred_success), mode = "everything")

mean(pred_success==df_inv_Test$status)
mean(pred_success!=df_inv_Test$status)


##ploting lift

install.packages("pROC")
install.packages('ROCR')

library(ROCR)
library(pROC)


ROCR_pred_test <- prediction(prob_success_pred,df_inv_Test$status)

ROCR_perf_test <- performance(ROCR_pred_test,'tpr','fpr')

plot(ROCR_perf_test,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))
abline(0,1,lty=2)


auc_value <- roc(df_inv_Test$status, prob_success_pred)
auc(auc_value)

#### Attempt at changing beta for inbalanced dataset - failed attempt
#Prob_closed_in_data = mean(df_inv_Train == 'closed')
#Prob_closed_in_data



#true_prob_closed = 0.15
#new_beta0 = logistic_fit$coefficients[1] + log(true_prob_closed/(1-true_prob_closed)) - log(Prob_closed_in_data/(1-Prob_closed_in_data))

#logistic_fit$coefficients[1]
#new_beta0
#prob_yes[1]

#logistic_fit$coefficients[1] = new_beta0
#pred_yes_new_beta0 = predict(logistic_fit,type='response')
#pred_yes_new_beta0[1]

#pred_yes_new_beta0 = predict(glm1,heart,type='response')
#pred_yes_new_beta0[1]



# 4) Decision Trees - Single Tree, Bagging, Random Forests, Boosting

#install.packages('tree')
#install.packages('caret')
#install.packages('MLmetrics')
library(MLmetrics)
library(caret)
library(tree)

#Factoring the data
df_startup = subset(df_inv, select = c('status_label','market_type',
                                       'total_investment','diff_funding_year','seed','venture','funding_rounds') )
str(df_startup)
df_startup$market_type <- as.factor(df_startup$market_type)
df_startup$funding_rounds <- as.numeric(df_startup$funding_rounds)
df_startup$seed <- as.numeric(df_startup$seed)
df_startup$status_label <- as.factor(df_startup$status_label)
attach(df_startup)

#Splitting the data set into training and testing data
set.seed (2)
rows=dim(df_startup)[1]
tr <- sample(1:rows, (0.7*rows))
train=data.frame(df_startup)
test_data=data.frame(df_startup)
train=train[tr,]
test_data=test_data[-tr,]

#Predicting using single tree
tree.status <- tree(status_label ~ .-status_label,data=train,mindev=.001)
summary(tree.status)
plot(tree.status)
text(tree.status , pretty = 0)

tree.pred <- predict(tree.status ,test_data ,
                     type = "class")
table(tree.pred , test_data$status_label)
confusionMatrix(tree.pred, test_data$status_label,
                mode = "everything")

#Bagging
#install.packages('randomForest')
library(randomForest)
set.seed (10)
bag.status <- randomForest(status_label ~ ., data = train ,
                           mtry = 6, importance = TRUE)

#Predicting using Bagging
yhat.status <- predict(bag.status , test_data, type = 'class')
confusionMatrix(yhat.status, test_data$status_label,
                mode = "everything")

tree.number = c(200,400,500,1000,2000,3000,5000,6000,7000)
length.tree.number = length(tree.number)
train.errors.tree = rep(NA, length.tree.number)
test.errors.tree = rep(NA, length.tree.number)

for (i in 1:length.tree.number) {
  random.status = randomForest(status_label ~ ., data = train, mtry = 6, ntree = tree.number[i])
  train.pred = predict(random.status, train, n.trees = tree.number[i])
  test.pred = predict(random.status, test_data, n.trees = tree.number[i])
  table.status = table(test.pred , test_data$status_label)
  test.errors.tree[i]=(table.status[1]+table.status[4])/(table.status[1]+table.status[4]+table.status[2]+table.status[3])
}

plot(tree.number, test.errors.tree, type = "b", xlab = "tree number", ylab = "Accuracy", col = "red", pch = 20)
lines(tree.number,test.errors.tree,type = "b",col="blue", pch = 20)
max(test.errors.tree)
tree.number[which.max(test.errors.tree)]


#Random Forest
set.seed (12)
rf.status <- randomForest(status_label ~ ., data = train ,
                          mtry = 4, importance = TRUE)

yhat.rf <- predict(rf.status, test_data, type = 'class')
confusionMatrix(yhat.rf, test_data$status_label,
                mode = "everything")
importance(rf.status)
varImpPlot(rf.status)

#Boosting
library(gbm)
set.seed (19)
train$status_label <- as.character(train$status_label)
boost.startup <- gbm(train$status_label ~ ., data = train,
                     distribution = "bernoulli", n.trees = 4000, shrinkage = .2,
                     interaction.depth = 3)
summary(boost.startup)

pred.boost <- predict.gbm(boost.startup ,
                          test_data, n.trees = 4000,type='response',verbose=FALSE)

pred.boost
pred.boost = factor(ifelse(pred.boost<'0.5',"0","1"))
confusionMatrix(pred.boost, as.factor(test_data$status_label),
                mode = "everything")