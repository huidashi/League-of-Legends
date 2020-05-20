rm(list=ls())
cat("\014")
library(glmnet)
library(dplyr)
library(randomForest)
library(reshape)
library(ggplot2)
library(gridExtra)

#standardize function
standardize=function(x){
  return (x/sqrt(mean((x-mean(x))^2)))
}

#clean results and remove values larger than | 1 | for boxplot representation
clean=function(x){
  return (x[abs(x)<1])
}


set.seed(1)
data1=read.csv("~/Downloads/league-of-legends-ranked-matches (1)/stats1.csv")
data2=read.csv("~/Downloads/league-of-legends-ranked-matches (1)/stats2.csv")
data=rbind(data1,data2)
glimpse(data)


#randomly select 5k rows for faster computation to work with for now

df=sample_n(data,8000)

sum(is.na(df))


#for ( x in 1:ncol(df)){
#  for (j in 1:nrow(df)){
#    data[x,j] = ifelse(is.na(df[x,j]),mean(df.matrix(df[,x]),na.rm=TRUE),df[j,x])
#  }
#}

df=df[,-c(3:9)]
df=select(df,-id)
df=select(df,-wardsbought)
df=select(df,-legendarykills)
df=select(df,-timecc)
#df=select(df,-win)





#standardizing 45 variables
df=df %>% mutate_all(.funs= standardize)

glimpse(df)

n = dim(df)[1]
p = dim(df)[2]-1
y = data.matrix(df['deaths'])
X = data.matrix(df[,-3])
mu = as.vector(apply(X, 2, 'mean'))
sd = as.vector(apply(X, 2, 'sd'))



n.train        =     floor(0.8*n)
n.test         =     n-n.train


M              =     30
Rsq.test.rf    =     rep(0,M)  # rf= randomForest
Rsq.train.rf   =     rep(0,M)
Rsq.test.en    =     rep(0,M)  #en = elastic net
Rsq.train.en   =     rep(0,M)
Rsq.test.las   =     rep(0,M)
Rsq.train.las  =     rep(0,M)
Rsq.test.rid   =     rep(0,M)
Rsq.train.rid  =     rep(0,M)

Res.test.rf    =     rep(0,M)  # rf= randomForest
Res.train.rf   =     rep(0,M)
Res.test.en    =     rep(0,M)  #en = elastic net
Res.train.en   =     rep(0,M)
Res.test.las   =     rep(0,M)
Res.train.las  =     rep(0,M)
Res.test.rid   =     rep(0,M)
Res.train.rid  =     rep(0,M)

for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  # fit elastic-net and calculate and record the train and test R squares 
  a=0.5 # elastic-net
  cv.fit           =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train, alpha = a, lambda = cv.fit$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.en[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Res.train.en[m]  =     (y.train - y.train.hat)^2
  Res.test.en[m]   =     (y.test - y.test.hat)^2
  
  
  # fit RF and calculate and record the train and test R squares 
  rf               =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.test.hat       =     predict(rf, X.test)
  y.train.hat      =     predict(rf, X.train)
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Res.train.rf[m]  =     (y.train - y.train.hat)^2
  Res.test.rf[m]   =     (y.test - y.test.hat)^2
  
  # fit lasso and calculate and record the train and test R squares
  cv.fit.las       =     cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
  fit              =     glmnet(X.train, y.train, alpha = 1, lambda = cv.fit.las$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.las[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.las[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  Res.train.las[m]  =     (y.train - y.train.hat)^2
  Res.test.las[m]   =     (y.test - y.test.hat)^2
  
  # fit ridge and calculate and record the train and test R squares
  cv.fit.rid       =     cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
  fit              =     glmnet(X.train, y.train, alpha = 0, lambda = cv.fit.rid$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.rid[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rid[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2) 
  Res.train.rid[m]  =     (y.train - y.train.hat)^2
  Res.test.rid[m]   =     (y.test - y.test.hat)^2
  
  
  cat(sprintf("m=%3.f| Rsq.test.rf=%.2f,  Rsq.test.en=%.2f,  Rsq.test.las=%.2f,  Rsq.test.rid=%.2f| Rsq.train.rf=%.2f,  Rsq.train.en=%.2f,  Rsq.train.las=%.2f,  Rsq.train.rid=%.2f| \n", 
              m,  Rsq.test.rf[m], Rsq.test.en[m], Rsq.test.las[m], Rsq.test.rid[m],  Rsq.train.rf[m], Rsq.train.en[m],  Rsq.train.las[m], Rsq.train.rid[m]))
  
}
plot(cv.fit, main = 'Elastic Net Cross Validation Curve ')
plot(cv.fit.las, main = 'Lasso Cross Validation Curve ')
plot(cv.fit.rid, main = 'Ridge Cross Validation Curve ')



boxplot(clean(Rsq.train.rf),clean(Rsq.train.en),clean(Rsq.train.las),clean(Rsq.train.rid), main = 'Rsquared Train' ,names =c('RandomForest','Elastic Net','Lasso','Ridge'))
boxplot(clean(Rsq.test.rf),clean(Rsq.test.en),clean(Rsq.test.las),clean(Rsq.test.rid), main = 'Rsquared Test' ,names =c('RandomForest','Elastic Net','Lasso','Ridge'))

boxplot(Res.train.rf,Res.train.en,Res.train.las,Res.train.rid, main = 'Residual Train',names =c('RandomForest','Elastic Net','Lasso','Ridge'))
boxplot(Res.test.rf,Res.test.en,Res.test.las,Res.test.rid, main = 'Residual Test',names =c('RandomForest','Elastic Net','Lasso','Ridge'))


bootstrapSamples =     30
beta.rf.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.en.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)
beta.las.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.rid.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    

for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # fit bs rf
  rf               =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m]   =     as.vector(rf$importance[,1])
  # fit bs en
  a                =     0.5 # elastic-net
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m]   =     as.vector(fit$beta)
  
  # fit bs las
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = 1, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = 1, lambda = cv.fit$lambda.min)  
  beta.las.bs[,m]   =     as.vector(fit$beta)
  
  # fit bs ridge
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = 0, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = 0, lambda = cv.fit$lambda.min)  
  beta.rid.bs[,m]   =     as.vector(fit$beta)
  cat(sprintf("Bootstrap Sample %3.f \n", m))
  
}
# calculate bootstrapped standard errors / alternatively you could use qunatiles to find upper and lower bounds
rf.bs.sd    = apply(beta.rf.bs, 1, "sd")
en.bs.sd    = apply(beta.en.bs, 1, "sd")
las.bs.sd    = apply(beta.las.bs, 1, "sd")
rid.bs.sd    = apply(beta.rid.bs, 1, "sd")


# fit rf to the whole data
rf               =     randomForest(X, y, mtry = sqrt(p), importance = TRUE)

# fit en to the whole data
a=0.5 # elastic-net
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.en              =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)

# fit las to the whole data
cv.fit           =     cv.glmnet(X, y, alpha = 1, nfolds = 10)
fit.las              =     glmnet(X, y, alpha = 1, lambda = cv.fit$lambda.min)

# fit rid to the whole data
cv.fit           =     cv.glmnet(X, y, alpha = 0, nfolds = 10)
fit.rid              =     glmnet(X, y, alpha = 0, lambda = cv.fit$lambda.min)


betaS.rf               =     data.frame(c(1:p), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")

betaS.en               =     data.frame(c(1:p), as.vector(fit.en$beta), 2*en.bs.sd)
colnames(betaS.en)     =     c( "feature", "value", "err")

betaS.las               =     data.frame(c(1:p), as.vector(fit.las$beta), 2*las.bs.sd)
colnames(betaS.las)     =     c( "feature", "value", "err")

betaS.rid               =     data.frame(c(1:p), as.vector(fit.rid$beta), 2*rid.bs.sd)
colnames(betaS.rid)     =     c( "feature", "value", "err")

rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle('BootStrap Error Bars\n Random Forest')


enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle('Elastic Net')

lasPlot =  ggplot(betaS.las, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle('Lasso')

ridPlot =  ggplot(betaS.rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle('Ridge')

grid.arrange(rfPlot, enPlot, lasPlot,ridPlot, nrow = 4)

# we need to change the order of factor levels by specifying the order explicitly.
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.las$feature     =  factor(betaS.las$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.rid$feature     =  factor(betaS.rid$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])



rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle('BootStrap Error Bars\n Random Forest')

enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle('Elastic Net')

lasPlot =  ggplot(betaS.las, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle('Lasso')

ridPlot =  ggplot(betaS.rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + ggtitle('Ridge')

grid.arrange(rfPlot, enPlot, lasPlot,ridPlot, nrow = 4)

