Traindataset <- read.csv("C:/Users/harde/OneDrive/Belgeler/R Projects/tensor/TrainDataset.txt" , header=TRUE, sep=",")
Testdataset <- read.csv("C:/Users/harde/OneDrive/Belgeler/R Projects/tensor/TestDataset.txt" , header=TRUE, sep=",")
library(neuralnet)

sigmoid = function(x) {
  1 / (1 + exp(-x))
}


index <- sample(1:nrow(Traindataset),40000)
index2 <- sample(1:nrow(Testdataset),19000)

nn <- neuralnet(TARGET ~ ., Traindataset[index,],linear.output = FALSE,err.fct = "ce",hidden = c(4,2), act.fct = sigmoid,threshold = 0.8)

plot(nn)

temp_test <- subset(Testdataset[index2,])

nn.results <- predict(nn, temp_test)

results <- data.frame(actual = temp_test$TARGET, prediction = nn.results)

roundedresults<-sapply(results,round,digits=0)

roundedresultsdf=data.frame(roundedresults)

confusion_matrix <- table(roundedresultsdf$actual,roundedresultsdf$prediction)

accuracy <- ((confusion_matrix[1,1]+confusion_matrix[2,2])/(sum(confusion_matrix)))


TP <- confusion_matrix[1]
FP <- confusion_matrix[2]
FN <- confusion_matrix[3]
TN <- confusion_matrix[4]

RECALL <- TP/(TP+FN)
SPECIFICITY <- TN/(TN+FP)
PRECISION <- TP/(TP+FP)
F1 <- 2*PRECISION*RECALL/(PRECISION+RECALL)

temp <- roundedresultsdf$prediction - roundedresultsdf$actual
temp <- temp^2
temp_sum <- sum(temp)

MSE <- temp_sum / length(temp)
RMSE <- sqrt(MSE)


cat("Accuracy:", accuracy)
cat("Recall:", RECALL)
cat("Spesificity:", SPECIFICITY)
cat("Precision:", PRECISION)
cat("F1-Score:", F1)
cat("MSE:",MSE)
cat("RMSE:",RMSE)
cat("MAE:",MSE)


accs <- numeric()

cv <-Traindataset[index,]

for (i in 1:5) {
  sub <- sample(1:nrow(cv),size=nrow(cv)*0.80)
  fit <-  neuralnet(TARGET ~ ., cv[sub,],linear.output = FALSE,err.fct = "ce",hidden = c(4,2), act.fct = sigmoid,threshold = 0.8)
  
  test_predict <- predict(fit, cv[-sub,])
  rm(results)
  results <- data.frame(actual = cv[-sub,]$TARGET, prediction = test_predict)
  
  roundedresults<-sapply(results,round,digits=0)
  
  roundedresultsdf=data.frame(roundedresults)
  
  confusion_matrix <- table(roundedresultsdf$actual,roundedresultsdf$prediction)
  
  accuracy <- ((confusion_matrix[1,1]+confusion_matrix[2,2])/(sum(confusion_matrix)))
  
  accs <- append(accs,accuracy)
}


plot(accs,type = "o",xlab = "Iteration",ylab = "Accuracy",main = "Accuracy changes over 5 iterations in Cross Validation")
cat(("Average accuracy is: "),mean(accs))




test_comment <- read.csv("C:/Users/harde/PycharmProjects/TensorFlowDeneme/TestComment.txt",header = FALSE,sep = ",")

comment_predict <- predict(nn,test_comment)

comment_predict


