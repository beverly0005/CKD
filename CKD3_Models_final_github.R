rm(list=ls())

#IMPORT LIBRARIES-----------------
library(ggplot2)
library(caret)
library(pROC)
library(xgboost)
library(e1071)
library(Formula)
library(backports)
library(Hmisc)
library(caTools)


#MODEL DEVELOPMENT---------
All_Data<-read.csv('CKD3_Data/CKD3_8Aug2018.csv', header = TRUE, na.strings=c("","NA"))

#Process data
colnames(All_Data)[colnames(All_Data)=="ï..CDS_PAT_ID"] <- "CDS_PAT_ID"
All_Data2 <- subset(All_Data, select = c(LATEST_EGFR_RESULT,PRIOR_EGFR_RESULT,AGE2012,LATEST_HAE_RESULT,LATEST_ALBURI_RESULT,LATEST_UACR_RESULT, PRIOR_UACR_RESULT,
                                         NUM_UNQ_GRPC,LATEST_ALBSERUM_RESULT, LATEST_A1C_RESULT, DIABETES_DURATION_DAYS,
                                         LATEST_GLU_RESULT,LATEST_TRI_RESULT,LATEST_EGFR_FACILITY,
                                         PRIOR_A1C_RESULT,NUM_UNQ_GRPA,LATEST_ALT_RESULT,HRM_GENDER_VALUE,HRM_RACE_VALUE,
                                         LATEST_HDL_RESULT,PRIOR_HAE_RESULT,PRIOR_ALBSERUM_RESULT,
                                         LATEST_TC_RESULT,LATEST_LDL_RESULT,CKD3_TARGET))


All_Data2$HRM_RACE_VALUE <- gsub("INDIAN", "Indian", All_Data2$HRM_RACE_VALUE)
All_Data2$HRM_RACE_VALUE <- gsub("MALAY", "Malay", All_Data2$HRM_RACE_VALUE)
All_Data2$HRM_RACE_VALUE <- as.factor(All_Data2$HRM_RACE_VALUE)
All_Data2$LATEST_EGFR_FACILITY <- gsub("TAN TOCK SENG HOSPITAL", "Tan Tock Seng Hospital", All_Data2$LATEST_EGFR_FACILITY)
All_Data2$CKD3_TARGET <- as.factor(All_Data2$CKD3_TARGET)
All_Data2$CKD3_TARGET <- gsub("No        ","No",All_Data2$CKD3_TARGET)
All_Data2$CKD3_TARGET <- gsub("Yes       ","Yes",All_Data2$CKD3_TARGET)

#Create dummy variables for categorical variables
All_Data2_Catvar <- subset(All_Data2, select=c(HRM_GENDER_VALUE,HRM_RACE_VALUE,LATEST_EGFR_FACILITY))

All_Data2_Numvar <- subset(All_Data2, select=-c(HRM_GENDER_VALUE,HRM_RACE_VALUE,LATEST_EGFR_FACILITY,CKD3_TARGET))

All_Data2_Label <- subset(All_Data2, select=c(CKD3_TARGET))

All_Data2_Catvar = as.matrix(All_Data2_Catvar)
All_Data2_Catvar[is.na(All_Data2_Catvar)] <- "Missing"
All_Data2_Catvar = as.data.frame(All_Data2_Catvar)

dummies <- dummyVars(~ ., data = All_Data2_Catvar, fullRank = F)
All_Data2_Catvar <- data.frame(predict(dummies, newdata = All_Data2_Catvar))
All_Data2_Catvar <- subset(All_Data2_Catvar, select=-c(HRM_GENDER_VALUE.Female,HRM_RACE_VALUE.Missing, LATEST_EGFR_FACILITY.Missing))

All_Data2_Catvar <- data.frame(lapply(All_Data2_Catvar, as.numeric))
All_Data2_Numvar <- data.frame(lapply(All_Data2_Numvar, as.numeric))
dt <- cbind(All_Data2_Catvar,All_Data2_Numvar,All_Data2_Label)
dt$CKD3_TARGET <- as.factor(dt$CKD3_TARGET)

#Impute missing value
dt[is.na(dt)] <- 99999

#Split into train and test dataset
set.seed(1234)
split = sample.split(dt$CKD3_TARGET, SplitRatio = 0.75)

traindt <- subset(dt, split==TRUE)
validatedt <- subset(dt, split==FALSE)
ncol(traindt)
nrow(traindt)

labelname <- 'CKD3_TARGET'
predictors1<-names(traindt)[names(traindt)!=labelname]
length(predictors1)

## Train model##
set.seed(1234)
mycontrol<- trainControl(method='cv', number=3, classProbs = TRUE, summaryFunction = twoClassSummary)

xgbGrid <-  expand.grid(max_depth = (3:5), eta = 0.1, gamma = 0,  
                        min_child_weight = (7:9), nrounds = (4:6)*100,
                        colsample_bytree = 0.6, subsample=1)

labelname <- 'CKD3_TARGET'
features <- colnames(traindt)

predictors <- features[sapply(features, function(x)x %in% c("LATEST_EGFR_RESULT","PRIOR_EGFR_RESULT","AGE2012","LATEST_HAE_RESULT","LATEST_ALBURI_RESULT","LATEST_UACR_RESULT",
                                                            "NUM_UNQ_GRPC","LATEST_ALBSERUM_RESULT", "LATEST_A1C_RESULT", "DIABETES_DURATION_DAYS",
                                                            "LATEST_GLU_RESULT","LATEST_TRI_RESULT","LATEST_EGFR_FACILITY.SINGAPORE.GENERAL.HOSPITAL",
                                                            "PRIOR_A1C_RESULT","NUM_UNQ_GRPA","LATEST_ALT_RESULT","HRM_GENDER_VALUE.Male","HRM_RACE_VALUE.Malay",
                                                            "LATEST_HDL_RESULT","PRIOR_HAE_RESULT","PRIOR_ALBSERUM_RESULT","HRM_RACE_VALUE.Chinese","HRM_RACE_VALUE.Indian",
                                                            "LATEST_TC_RESULT","LATEST_LDL_RESULT"))]


# Attention that the order of variables in predictors is critical to the xgboost performance
predictors <- c("HRM_GENDER_VALUE.Male","HRM_RACE_VALUE.Chinese","HRM_RACE_VALUE.Indian","HRM_RACE_VALUE.Malay",	
                "LATEST_EGFR_FACILITY.SINGAPORE.GENERAL.HOSPITAL","AGE2012","DIABETES_DURATION_DAYS","LATEST_A1C_RESULT",
                "PRIOR_A1C_RESULT","LATEST_ALBURI_RESULT","LATEST_GLU_RESULT","LATEST_HAE_RESULT","PRIOR_HAE_RESULT",
                "LATEST_UACR_RESULT","LATEST_ALBSERUM_RESULT","PRIOR_ALBSERUM_RESULT","LATEST_EGFR_RESULT","PRIOR_EGFR_RESULT",
                "LATEST_ALT_RESULT","LATEST_TRI_RESULT","LATEST_HDL_RESULT","LATEST_LDL_RESULT","LATEST_TC_RESULT",
                "NUM_UNQ_GRPA","NUM_UNQ_GRPC")

length(predictors)

# modelxgboost<-train(traindt[,predictors],traindt[,labelname],method='xgbTree', trControl=mycontrol,  tuneGrid = xgbGrid, metric="ROC")


#MODEL REFERENCING --------------------
#Once the model is well-trained
modelxgboost <- readRDS('xgboostmodel.rds')

modelxgboost
# max_depth  gamma min_child_weight  nrounds    ROC        Sens       Spec     
# 3          9                       500        0.8922925  0.9378561  0.5699529

#variance importance
importantvar = varImp(modelxgboost,scale=FALSE)
plot(importantvar)


test_results <- predict(modelxgboost, validatedt[, predictors], type="prob")
test_results$pred <- predict(modelxgboost, validatedt[,predictors])
test_results$obs <- validatedt[,labelname]
test_results$pred = Hmisc::cut2(test_results$Yes, c(0, 0.1, 1))
levels(test_results$pred) <- c("No","Yes")
test_results$obs <- as.factor(test_results$obs)
confusionMatrix(test_results$pred, test_results$obs, positive = "Yes")
auc(validatedt[, labelname], test_results$Yes)

