## Library
library(caret)
library(skimr)
library(rlang)
library(RANN)
library(dplyr)

## Pipeline
pipeline <- function(dataset, preprocess, dataForPrediction) {
  
  #Descriptive statistics
  skimmed <- skim(dataset)
  print(skimmed[, c(1:5, 9:11, 13, 15:16)])
  
  set.seed(100)
  
  #X-y Split
  x_train = dataset[,-ncol(dataset)]
  y_train = as.factor(dataset[,ncol(dataset)])
  
  str(x_train)
  
  
  # Add test data to train
  x_train <- rbind(x_train, dataForPrediction)
  
  #Missing Data
  print("--Missing Data--")
  preProcess_missingdata_model <- preProcess(x_train, method='knnImpute')
  x_train <- predict(preProcess_missingdata_model, newdata = x_train)
  
  #One hot encoding
  print("--Encoding--")
  dummies_model <- dummyVars( ~ .,data=x_train)
  trainData_mat <- predict(dummies_model, newdata = x_train)
  x_train <- data.frame(trainData_mat)
  str(x_train)
  
  #PreProcess
  print("--Preprocess--")
  preProcess_range_model <- preProcess(x_train, method=preprocess)
  x_train <- predict(preProcess_range_model, newdata = x_train)
  
  # Train and test row size 
  row_test <- nrow(dataForPrediction)
  row_train <- nrow(x_train)
  
  # Split Train and Test data 
  dataForPrediction <- tail(x_train, n=row_test)
  dataForPrediction <- data.frame(lapply(dataForPrediction, type.convert), stringsAsFactors=FALSE)
  x_train <- head(x_train, n=row_train-row_test)
  
  
  #Data Exploration
  print("--Exploration--")
  plotFeature <- featurePlot(x = x_train, 
                             y = y_train, 
                             plot = "box",
                             strip=strip.custom(par.strip.text=list(cex=.7)),
                             scales = list(x = list(relation="free"), 
                                           y = list(relation="free")),
                             environment = environment())
  print(plotFeature)
  
  plotFeature2 <- featurePlot(x = x_train, 
                              y = y_train, 
                              plot = "strip",
                              jitter = TRUE ,
                              strip=strip.custom(par.strip.text=list(cex=.7)),
                              scales = list(x = list(relation="free"), 
                                            y = list(relation="free")),
                              environment = environment())
  print(plotFeature2)
  
  #For train control
  fitControl <- trainControl(
    method = 'cv',                   # k-fold cross validation
    number = 10                      # number of folds
  ) 
  print("--Train--")
  # Train the model using knn
  print("--KNN--")
  set.seed(100)
  model_knn = train(x_train, y_train, method='knn', trControl=fitControl)
  
  # Train the model using svm linear
  print("--SVM Linear--")
  set.seed(100)
  model_svmL = train(x_train, y_train, method='svmLinear', trControl=fitControl)
  
  # Train the model using plr
  print("--PLR--")
  set.seed(100)
  model_plr = train(x_train, y_train, method='plr', trControl=fitControl)
  
  # Train the model using naive bayes
  print("--Naive Bayes--")
  set.seed(100)
  model_nb = train(x_train, y_train, method='naive_bayes', trControl=fitControl)
  
  # Compare model performances using resample()
  print("--Model Compare and Choose Best--")
  models_compare <- resamples(list(KNN=model_knn, PLR=model_plr, SVML=model_svmL, NB=model_nb))
  
  # Draw box plots to compare models
  scales <- list(x=list(relation="free"), y=list(relation="free"))
  plotM <- bwplot(models_compare, scales=scales, environment = environment())
  print(plotM)
  
  # Summary of the models performances
  summary_model <- summary(models_compare)
  summary_model <- summary_model$statistics$Accuracy
  summary_model <- sort(summary_model[,c('Mean')], decreasing = TRUE)
  bestModel <- attr(summary_model[1], 'names')
  
  #Model Chooser
  if (bestModel == "KNN") {
    fittedTest <- predict(model_knn, dataForPrediction)
  } else if (bestModel == "PLR") {
    fittedTest <- predict(model_plr, dataForPrediction)
  } else if (bestModel == "SVML") {
    fittedTest <- predict(model_svmL, dataForPrediction)
  } else if (bestModel == "NB") {
    fittedTest <- predict(model_nb, dataForPrediction)
  } 
  return(fittedTest)
}
## Result
#Read Dataset
data <- read.csv("online_shoppers_intention.csv")
data$Revenue[data$Revenue == FALSE] <- 0
data$Revenue[data$Revenue == TRUE] <- 1

random_row <- sample(1:nrow(data), 10, replace=FALSE)

testData <- data[c(random_row),1:17]
testData_y <- data[c(random_row), ncol(data)]
predict <- pipeline(dataset = data, preprocess = 'scale', testData)
#Predict result
predict

# Conf matrix
expected <- factor(testData_y)
conf<- confusionMatrix(data=predict, reference=expected)
conf
