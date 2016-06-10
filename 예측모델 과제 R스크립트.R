library(caret)
library(C50)
library(ROCR)
library(devtools)
library(randomForest)

start_data <- read.csv("D:/BigData MBA/데이터 마이닝/예측모델 과제/transactions.csv", stringsAsFactors = FALSE) 
data = start_data[,-1]
head(data)
data$churn <- factor(data$churn)
set.seed(1)
inTrain = createDataPartition(y=data$churn, p=0.7, list=FALSE)  
data.train = data[inTrain,] 
data.test = data[-inTrain,]
dim(data.train)
dim(data.test)

##--------C5.0 모델링
c5_options <- C5.0Control(winnow = FALSE, noGlobalPruning = FALSE) 
c5_model <- C5.0(churn ~., data=data.train, control=c5_options, rules=FALSE) 

summary(c5_model)

plot(c5_model)
data.test$c5_pred <- predict(c5_model, data.test, type="class") 
data.test$c5_pred_prob <- predict(c5_model, data.test, type="prob")
confusionMatrix(data.test$c5_pred, data.test$churn)

varImp(c5_model)


c5_pred <- prediction(data.test$c5_pred_prob[,2],data.test$churn) 
c5_model.perf1 <- performance(c5_pred, "tpr", "fpr") 
c5_model.perf2 <- performance(c5_pred, "lift", "rpp") 



##--------신경망 모델링
nn_model = nnet(churn ~., data=data.train, size=7, maxit=2460)
summary(nn_model)

source_url('https://gist.githubusercontent.com/Peque/41a9e20d6687f2f3108d/raw/85e14f3a292e126f1454864427e3a189c2fe33f3/nnet_plot_update.r') 
plot.nnet(nn_model)
confusionMatrix(predict(nn_model, newdata=data.test, type="class"),
                data.test$churn)

nn_pred <- prediction(predict(nn_model, newdata=data.test, type="raw"), data.test$churn) 
nn_model.perf1 <- performance(nn_pred, "tpr", "fpr")
nn_model.perf2 <- performance(nn_pred, "lift", "rpp")


##--------RandomForest
rf_model = randomForest(churn ~., data=data.train, ntree=21)

rf_model_var = varImp(rf_model)  ## 설명변수의 중요도 
var_rf = order(rf_model_var, decreasing=T)[1:5]

row.names(rf_model_var)[c(37, 113, 40, 3, 64)]



plot(rf_model)
data.test$rf_pred <- predict(rf_model, data.test, type="class") 
data.test$rf_pred_prob <- predict(rf_model, data.test, type="prob")

confusionMatrix(predict(rf_model, newdata=data.test, type="class"),
                data.test$churn)

rf_pred <- prediction(data.test$rf_pred_prob[,2],data.test$churn) 
rf_model.perf1 <- performance(rf_pred, "tpr", "fpr")
rf_model.perf2 <- performance(rf_pred, "lift", "rpp")

##-------------모델 평가---------##
## ROC
plot(c5_model.perf1, col="red", main ='ROC')
plot(nn_model.perf1, col="blue", main ='ROC', add=T)
plot(rf_model.perf1, col="green", main ='ROC', add=T)
legend("topright",legend=c("C5.0","nnet","RForest"), fill=c("red", "blue", "green"))
??legend

## Lift
plot(c5_model.perf2, col="red", main ='Lift')
plot(nn_model.perf2, col="blue", main ='Lift', add=T)
plot(rf_model.perf2, col="green", main ='Lift', add=T)
legend("topright",legend=c("C5.0","nnet","RForest"), fill=c("red", "blue", "green"))
