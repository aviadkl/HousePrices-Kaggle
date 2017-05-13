
# Predictive model for Housing Prices -  data from Kaggle #

# Load raw data
House.train <- read.csv("House Prices/train.csv", header = TRUE)
House.test <- read.csv("House Prices/test.csv", header = TRUE)

House.all <- rbind(House.train[,1:80],House.test)

SalePrice = House.train$SalePrice
summary(SalePrice)

library(ggplot2)
ggplot(House.train) +
  geom_histogram(aes(x = SalePrice), binwidth=5000)

# Review data
str(House.train)



#============= Check for uninformative variables =============

# Remove int variables
library(caret)
nums <- sapply(House.all, is.integer)
int.predictors <- House.all[,nums]
int.predictors <- int.predictors[,!names(int.predictors) %in% c("Id")]
zeroVars <- nearZeroVar(int.predictors)
int.predictors <- int.predictors[,-zeroVars]

# Remove factor variables
facs <- sapply(House.all, is.factor)
factor.predictors <- House.all[,facs]
# function: if one level is 90% and more, then variable is uninformative
zeroVar.factor <- function(x){
                      temp <- as.data.frame(table(x))
                      temp$pct <- temp[2]/sum(temp[2])*100
                      ifelse(temp$pct > 90.0, return(TRUE),return(FALSE))
                      ifelse(sum(temp$pct > 90.0) == 1,1,0)
                  }


zeroFacs <- which(sapply(factor.predictors,zeroVar.factor))
factor.predictors <- factor.predictors[-zeroFacs]



#============= Check for missing values =============

#== numeric ==
missing.int <- colSums(is.na(int.predictors))

# LotFrontage has 259 NA's, too much (17% of training data). removed.

# GarageYrBlt has 81 NA's. plot shows it is not informative. removed.
ggplot(int.predictors[1:1460,]) +
  geom_point(aes(x = GarageYrBlt, y = SalePrice)) +
  xlab("GarageYrBlt") +
  ylab("SalePrice")

int.predictors <- int.predictors[,-which(names(int.predictors) %in% c("LotFrontage", "GarageYrBlt"))]

# impute missing values
library(mice)
imp.int.predictors <- mice(int.predictors, m=1, method='cart')
imp.int.predictors <- complete(imp.int.predictors)


#== factor ==
missing.factor <- colSums(is.na(factor.predictors))

# NA's of some variables actually has meaning of "None" (data discription).

index <- which(colnames(factor.predictors) %in% c("Alley", "BsmtQual", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "PoolQC", "Fence"))

add.none.level <- function(x){
                      levels(x) <- c(levels(x),"None")
                      x[is.na(x)] <- "None"
                      return(x)
                  }

temp <- as.data.frame(lapply(factor.predictors[index],add.none.level))

factor.predictors[index] <- temp

# impute missing values
imp.factor.predictors <- mice(factor.predictors, m=1, method='cart')
imp.factor.predictors <- complete(imp.factor.predictors)
imp.factor.predictors<- as.data.frame(sapply(imp.factor.predictors,as.factor)) #remove contrasts


#============= Inspect possible predictors =============

summary(imp.int.predictors[1:1460,])

# fix misclassified variables
imp.int.predictors$OverallQual <- as.factor(imp.int.predictors$OverallQual)
imp.int.predictors$OverallCond <- as.factor(imp.int.predictors$OverallCond)

imp.factor.predictors <- cbind(imp.factor.predictors,
                               imp.int.predictors$OverallQual,
                               imp.int.predictors$OverallCond)

colnames(imp.factor.predictors)[names(imp.factor.predictors) %in% 
                              c("imp.int.predictors$OverallQual", "imp.int.predictors$OverallCond")] <- c("OverallQual", "OverallCond")

# remove the two new factors
nums <- which(sapply(imp.int.predictors, is.integer))
imp.int.predictors <- imp.int.predictors[,nums]

       
# Check correlation with SalePrice
cor.int <- cor(imp.int.predictors[1:1460,],SalePrice,method = "pearson")

cor.int.predictors <- imp.int.predictors[which(cor.int > 0.6)]
  

ggplot(imp.int.predictors[1:1460,]) +
  geom_point(aes(x = YearBuilt, y = SalePrice))
  #+scale_x_continuous(limits = c(0,2500))

ggplot(imp.int.predictors[1:1460,]) +
  geom_histogram(aes(x = MoSold))




summary(imp.factor.predictors[1:1460,])

# Check Mutual Information with SalePrice
library(infotheo)

info.factor <- sapply(imp.factor.predictors[1:1460,],function(x) mutinformation(SalePrice, x))

info.int <- sapply(imp.int.predictors[1:1460,],function(x) mutinformation(SalePrice, discretize(x)))



ggplot(imp.factor.predictors[1:1460,]) +
  geom_boxplot(aes(x = Neighborhood, y = SalePrice)) + 
  scale_y_continuous(limits = c(0,300000))


info.factor.predictors <- imp.factor.predictors[which(cor.factor > 0.85)]
info.int.predictors <- imp.int.predictors[which(info.int > 1.00)]

new.train1 <- cbind(cor.int.predictors[1:1460,],info.factor.predictors[1:1460,])
new.train2 <- cbind(info.int.predictors[1:1460,],info.factor.predictors[1:1460,])


info.factor.predictors <- imp.factor.predictors[which(cor.factor > 1.00)]
info.int.predictors <- imp.int.predictors[which(info.int > 1.50)]

new.train3 <- cbind(info.int.predictors[1:1460,],info.factor.predictors[1:1460,])

#==============================================================================

# לנסות
svmRadial
knn



#==================== Test for predictive models ====================
## Linear Regression ##
library(caret)
set.seed(2222)

# Train model with all meaningful variables
lin.train <- new.train2

ctrl <- trainControl(method = "repeatedcv", number = 10 , repeats = 10)


lin <- train(lin.train,
                  y = log(SalePrice),
                  method = "lm",
                  trControl = ctrl)

lin




## Random Forest ##

set.seed(2222)

# Train model with all meaningful variables
rf.train <- new.train2

ctrl <- trainControl(method = "repeatedcv", number = 10 , repeats = 10)


rf <- train(rf.train,
               y = log(SalePrice),
              method = "rf",
              ntree = 100,
               trControl = ctrl)

rf


## Neural Networks ##
library(nnet)
set.seed(2222)


# Train model with all meaningful variables
nn.train <- new.train2

nn <- nnet(formula = log(SalePrice)~.,
             data = nn.train,
             MaxNWts = 748,
             linout = TRUE,
             size = 9)
nn


# Train model with all variables
nn.train.2 <- cbind(int.predictors,factor.predictors)

ctrl <- trainControl(method = "repeatedcv", number = 5 , repeats = 10)


nn.2 <- train(nn.train.2,
              y = log(SalePrice),
              method = "nnet",
              tuneGrid = expand.grid(size=c(50), decay=c(0.1)),
              MaxNWts = 11251,
              trControl = ctrl)

nn.2





## Decision Tree ##
library(rpart)
library(rpart.plot)
set.seed(2222)

# Train model with all meaningful variables
dt.train <- new.train2

ctrl <- trainControl(method = "repeatedcv", number = 5 , repeats = 10)

dt <- train(x = dt.train, 
                  y = log(SalePrice), 
                  method = "rpart", 
                  trControl = ctrl)
dt

prp(dt.1$finalModel, type = 0, extra = 1, under = TRUE)



## Support Vector Machines ##

set.seed(2222)

# Train model with all meaningful variables
svm.train <- cbind(new.train2, SalePrice)

ctrl <- trainControl(method = "repeatedcv", number = 5 , repeats = 10)

svm <- train(log(SalePrice) ~ .,
            data = svm.train,
            method = "svmRadial",
            trControl = ctrl)

svm


## K-nearest neighbours ##

set.seed(2222)

# Train model with all meaningful variables
knn.train <- cbind(new.train2, SalePrice)

ctrl <- trainControl(method = "repeatedcv", number = 5 , repeats = 10)


knn <- train(log(SalePrice) ~ .,
             data = knn.train,
             method = "knn",
             trControl = ctrl)

knn


# Prepare Test Set

Final.test1 <- cbind(cor.int.predictors[1461:2919,],cor.factor.predictors[1461:2919,])

Final.test2 <- cbind(info.int.predictors[1461:2919,],info.factor.predictors[1461:2919,])



# Make predictions

preds <- exp(predict(rf, Final.test2))
table(preds)

# Write out a CSV file for submission to Kaggle
submit.df <- data.frame(Id = rep(1461:2919), SalePrice = preds)

write.csv(submit.df, file = "Houses_3_rf.csv", row.names = FALSE)


