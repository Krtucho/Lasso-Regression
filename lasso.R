#Load Libraries
library(stats)
library(caret)
library(glmnet)
library(pacman)
pacman::p_load(pacman,dplyr, ggplot2, rio, gridExtra, scales, ggcorrplot, e1071)

#Paquete GLMNET Regression Lasso

#We the next sintaxis for Lasso 
#glmnet(x,y...,alpha=1,nlambda=50..)
#alpha=1 runs Lasso's Regression

#Loading datasets for training and for testing
attach(train)
attach(test)

# Uncomment this lines to get more info about the data
#names(train)
#names(test)
#summary(train)
#summary(test)

# Show dimensions
dim(train)
dim(test)

# Getting independent variables.
x=model.matrix(price_range~.,train)

# Removing the intercept
x=model.matrix(price_range~.,train)[,-1]

#Getting dependent variable..
y=train$price_range

# Getting the Lasso Model
lasso.model=glmnet(x,y,alpha =1)
dim(coef(lasso.model))

Coeflasso=coef(lasso.model)
print(Coeflasso)
lasso.model$lambda
plot(lasso.model,"lambda",label=TRUE)

# Searching for the best lambda value with cross-validation.
cvob.cv=cv.glmnet(x,y,alpha=1)
plot(cvob.cv)
best.lambda =cvob.cv$lambda.min
best.lambda
log(best.lambda)

# Performing validation with the best lambda value.
x.test=model.matrix(price_range~.,test)[,-1]
coef(lasso.model)[,which(lasso.model$lambda==best.lambda)]
pred=predict(lasso.model, s=best.lambda,newx = x.test)
pred

# Root Mean Square Error -- R^2
data.frame(RMSE=RMSE(pred,test$price_range),Rsquare = R2(pred,test$price_range))