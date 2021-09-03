#Script to examine bike hire data and predict usage.

#load libraries
library(ggplot2)
library(dplyr)
library(caret)
library(corrplot)

#Read in source file (hourly bike hire data)
#Dataset read from github.  Original source:  https://www.kaggle.com/codebreaker619/bike-sharing-dataset

Hourly_Data <- read.csv("https://raw.githubusercontent.com/rwp-fp/Bike_Hire/master/hour.csv")

#Examine Dataset
Hourly_Data %>% group_by(weathersit) %>% summarise(total = mean(cnt)) %>% ggplot(aes(weathersit, total)) + geom_line()
Hourly_Data %>% group_by(atemp) %>% summarise(total = mean(cnt)) %>% ggplot(aes(atemp, total)) + geom_line()
Hourly_Data %>% group_by(weekday) %>% summarise(total = mean(cnt)) %>% ggplot(aes(weekday, total)) + geom_line()
Hourly_Data %>% group_by(workingday) %>% summarise(total = mean(cnt)) %>% ggplot(aes(workingday,total)) + geom_col()
Hourly_Data %>% group_by(holiday) %>% summarise(total = mean(cnt)) %>% ggplot(aes(holiday, total)) + geom_col()
corr <- Hourly_Data %>% select(-dteday, -instant) %>% cor()
corrplot(corr, type = "upper")

#Save plot for use in RMarkdown
png(file="./first_cor_plot.png",
    width=900, height=500)
print(corrplot(corr, type = "upper"))
dev.off()
rm(corr)

######################################################
#Split Dataset into training, test, and validation 
#Validation = 20% of dataset
#Training and Test datasets split 70/30 of remainder
######################################################

#Make Validation dataset
set.seed(1, sample.kind="Rounding")
ind <- createDataPartition(Hourly_Data$instant,0.2, times=1)
validation <- Hourly_Data[ind$Resample1,]
bike <- Hourly_Data[-ind$Resample1,]

#Split bike dataset into test and train
set.seed(1, sample.kind="Rounding")
ind <- createDataPartition(bike$instant,0.3, times=1)
bike_train <- bike[-ind$Resample1,]
bike_test <- bike[ind$Resample1,]
rm(bike, ind)

######################################################
# Create measure of error - RMSE function
######################################################

# Define function to measure RMSE
RMSE <- function(actual_hires, predicted_hires){
  sqrt(mean((actual_hires - predicted_hires)^2))
}

######################################################
# Manually predict - using linear regression
######################################################

#Calculate mean number of hires

bike_train <- bike_train %>% mutate(mean_hires = mean(bike_train$cnt))
bike_test <- bike_test %>% mutate(mean_hires = mean(bike_train$cnt))

#Calculate RMSE and put in data.frame to track results
Results_Tracker <- data.frame(Method = "Mean Only", RMSE = RMSE(bike_test$cnt,bike_test$mean_hires))


#Incude hour of day as predictor
hour_predictor <- bike_train %>% group_by(hr) %>% summarise(hour_pred = mean(cnt - mean_hires))
bike_train <- left_join(bike_train, hour_predictor, by = "hr")
bike_test <- left_join(bike_test, hour_predictor, by = "hr")

#Add RMSE to results tracker
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Hour", RMSE(bike_test$cnt,(bike_test$mean_hires + bike_test$hour_pred))))

rm(hour_predictor)


#Incude day as predictor
day_predictor <- bike_train %>% group_by(weekday) %>% summarise(day_pred = mean(cnt - mean_hires - hour_pred))
bike_train <- left_join(bike_train, day_predictor, by = "weekday")
bike_test <- left_join(bike_test, day_predictor, by = "weekday")

#Add RMSE to results tracker
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Hour + Day", RMSE(bike_test$cnt,(bike_test$mean_hires + bike_test$hour_pred + bike_test$day_pred))))

rm(day_predictor)


#Incude temp as predictor

#Create new variable based on rounding temp to 1 sig digit
bike_train <- bike_train %>% mutate(temp_rounded = signif(bike_train$temp, 1))
bike_test <- bike_test %>% mutate(temp_rounded = signif(bike_test$temp, 1))

#Predict based on rounded temperature
temp_predictor <- bike_train %>% group_by(temp_rounded) %>% summarise(temp_pred = mean(cnt - mean_hires - hour_pred - day_pred))
bike_train <- left_join(bike_train, temp_predictor, by = "temp_rounded")
bike_test <- left_join(bike_test, temp_predictor, by = "temp_rounded")

#Add RMSE to results tracker
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Hour + Day + Temp", RMSE(bike_test$cnt,(bike_test$mean_hires + bike_test$hour_pred + bike_test$day_pred + bike_test$temp_pred))))

rm(temp_predictor)

#Incude humidity as predictor

#Create new variable based on rounding humidity to 1 sig digit
bike_train <- bike_train %>% mutate(hum_rounded = signif(bike_train$hum, 1))
bike_test <- bike_test %>% mutate(hum_rounded = signif(bike_test$hum, 1))

#Predict based on rounded humidity
hum_predictor <- bike_train %>% group_by(hum_rounded) %>% summarise(hum_pred = mean(cnt - mean_hires - hour_pred - day_pred - temp_pred))
bike_train <- left_join(bike_train, hum_predictor, by = "hum_rounded")
bike_test <- left_join(bike_test, hum_predictor, by = "hum_rounded")

#Add RMSE to results tracker
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Hour + Day + Temp + Hum", RMSE(bike_test$cnt,(bike_test$mean_hires + bike_test$hour_pred + bike_test$day_pred + bike_test$temp_pred + bike_test$hum_pred))))

rm(hum_predictor)

#Incude year as predictor

year_predictor <- bike_train %>% group_by(yr) %>% summarise(year_pred = mean(cnt - mean_hires - hour_pred - day_pred - temp_pred - hum_pred))
bike_train <- left_join(bike_train, year_predictor, by = "yr")
bike_test <- left_join(bike_test, year_predictor, by = "yr")

#Add RMSE to results tracker
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Hour + Day + Temp + Hum + Year", RMSE(bike_test$cnt,(bike_test$mean_hires + bike_test$hour_pred + bike_test$day_pred + bike_test$temp_pred + bike_test$hum_pred +
                                                                                                              bike_test$year_pred))))

rm(year_predictor)

#Incude season as predictor

season_predictor <- bike_train %>% group_by(season) %>% summarise(season_pred = mean(cnt - mean_hires - hour_pred - day_pred - temp_pred - hum_pred - year_pred))
bike_train <- left_join(bike_train, season_predictor, by = "season")
bike_test <- left_join(bike_test, season_predictor, by = "season")

#Add RMSE to results tracker
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Hour + Day + Temp + Hum + Year + Season", RMSE(bike_test$cnt,(bike_test$mean_hires + bike_test$hour_pred + bike_test$day_pred + bike_test$temp_pred + bike_test$hum_pred +
                                                                                                              bike_test$year_pred + bike_test$season_pred))))

rm(season_predictor)

#Incude workingday as predictor

workday_predictor <- bike_train %>% group_by(workingday) %>% summarise(workday_pred = mean(cnt - mean_hires - hour_pred - day_pred - temp_pred - hum_pred - year_pred - season_pred))
bike_train <- left_join(bike_train, workday_predictor, by = "workingday")
bike_test <- left_join(bike_test, workday_predictor, by = "workingday")

#Add RMSE to results tracker
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Hour + Day + Temp + Hum + Year + Season + Working Day", RMSE(bike_test$cnt,(bike_test$mean_hires + bike_test$hour_pred + bike_test$day_pred + bike_test$temp_pred + bike_test$hum_pred +
                                                                                                                       bike_test$year_pred + bike_test$season_pred + bike_test$workday_pred))))

rm(workday_predictor)

#Incude weathersit as predictor

weathersit_predictor <- bike_train %>% group_by(weathersit) %>% summarise(weathersit_pred = mean(cnt - mean_hires - hour_pred - day_pred - temp_pred - hum_pred - year_pred - season_pred - workday_pred))
bike_train <- left_join(bike_train, weathersit_predictor, by = "weathersit")
bike_test <- left_join(bike_test, weathersit_predictor, by = "weathersit")

#Add RMSE to results tracker
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Hour + Day + Temp + Hum + Year + Season + Working Day + Weathersit", RMSE(bike_test$cnt,(bike_test$mean_hires + bike_test$hour_pred + bike_test$day_pred + bike_test$temp_pred + bike_test$hum_pred +
                                                                                                                                     bike_test$year_pred + bike_test$season_pred + bike_test$workday_pred + bike_test$weathersit_pred))))

rm(weathersit_predictor)

# Add final prediction to dataset
bike_test <- bike_test %>% mutate(estimate = bike_test$mean_hires + bike_test$hour_pred + bike_test$day_pred + bike_test$temp_pred + bike_test$hum_pred +
                                    bike_test$year_pred + bike_test$season_pred + bike_test$workday_pred + bike_test$weathersit_pred)

# Change any negative predictions to zero
bike_test$estimate <- ifelse(bike_test$estimate<0 , 0, bike_test$estimate)

# Add final RMSE
Results_Tracker <- Results_Tracker %>% rbind(c("Final Estimate (No negative estimates)", RMSE(bike_test$cnt,bike_test$estimate)))
                                               
######################
# Examine Errors
######################

# Add error figure to bike_test 
bike_test <- bike_test %>% mutate(error = cnt - estimate)

# Plot errors
bike_test %>% ggplot(aes(as.factor(weathersit), error)) + geom_boxplot()
bike_test %>% ggplot(aes(as.factor(signif(atemp,2)), error)) + geom_boxplot() 
bike_test %>% ggplot(aes(as.factor(weekday), error)) + geom_boxplot()
bike_test %>% ggplot(aes(as.factor(workingday),error)) + geom_boxplot()
bike_test %>% ggplot(aes(as.factor(holiday), error)) + geom_boxplot()
bike_test %>% ggplot(aes(as.factor(mnth), error)) + geom_boxplot()
bike_test %>% ggplot(aes(as.factor(season), error)) + geom_boxplot()

bike_test %>% ggplot(aes(cnt,error)) + geom_point() + geom_smooth()

corr <- bike_test %>% select(3:14,17,29:30) %>% cor()
corr %>% corrplot(type = "upper")
rm(corr)

###################################################
# Use machine learning algorithms
###################################################

# Make version of training dataset that removes unused fields
bike_train_machine <- bike_train %>% select(3:14,17)

# Try KNN
set.seed(123)
train_knn <- train(cnt ~ . , method = "knn", data = bike_train_machine)
predict_knn <- predict(train_knn, bike_test, type = "raw")
ggplot(train_knn, highlight = TRUE)

#Calculate RMSE and add to tracker
Results_Tracker <- Results_Tracker %>% rbind(c("KNN", RMSE(bike_test$cnt, predict_knn)))


# Try glm
set.seed(123)
train_glm <- train(cnt ~ . , method = "glm", data = bike_train_machine)
predict_glm <- predict(train_glm, bike_test, type = "raw")

#Calculate RMSE and add to tracker
Results_Tracker <- Results_Tracker %>% rbind(c("GLM", RMSE(bike_test$cnt, predict_glm)))

# Try svmLinear
set.seed(123)
train_svmLinear <- train(cnt ~ . , method = "svmLinear", data = bike_train_machine)
predict_svmLinear <- predict(train_svmLinear, bike_test, type = "raw")

#Calculate RMSE and add to tracker
Results_Tracker <- Results_Tracker %>% rbind(c("svmLinear", RMSE(bike_test$cnt, predict_svmLinear)))

# Try gamLoess
set.seed(123)
train_gamLoess <- train(cnt ~ . , method = "gamLoess", data = bike_train_machine)
predict_gamLoess <- predict(train_gamLoess, bike_test, type = "raw")

#Calculate RMSE and add to tracker
Results_Tracker <- Results_Tracker %>% rbind(c("gamLoess", RMSE(bike_test$cnt, predict_gamLoess)))

# Try kknn
set.seed(123)
train_kknn <- train(cnt ~ . , method = "kknn", data = bike_train_machine)
predict_kknn <- predict(train_kknn, bike_test, type = "raw")

#Calculate RMSE and add to tracker
Results_Tracker <- Results_Tracker %>% rbind(c("kknn", RMSE(bike_test$cnt, predict_kknn)))

# Try gam
set.seed(123)
train_gam <- train(cnt ~ . , method = "gam", data = bike_train_machine)
predict_gam <- predict(train_gam, bike_test, type = "raw")

#Calculate RMSE and add to tracker
Results_Tracker <- Results_Tracker %>% rbind(c("gam", RMSE(bike_test$cnt, predict_gam)))

# Try rf
set.seed(123)
train_rf <- train(cnt ~ . , method = "rf", data = bike_train_machine)
predict_rf <- predict(train_rf, bike_test, type = "raw")

#Calculate RMSE and add to tracker
Results_Tracker <- Results_Tracker %>% rbind(c("rf", RMSE(bike_test$cnt, predict_rf)))

# Try avNNet
set.seed(123)
train_avNNet <- train(cnt ~ . , method = "avNNet", data = bike_train_machine)
predict_avNNet <- predict(train_avNNet, bike_test, type = "raw")

#Calculate RMSE and add to tracker
Results_Tracker <- Results_Tracker %>% rbind(c("avNNet", RMSE(bike_test$cnt, predict_avNNet)))

# Try monmlp
set.seed(123)
train_monmlp <- train(cnt ~ . , method = "monmlp", data = bike_train_machine)
predict_monmlp <- predict(train_monmlp, bike_test, type = "raw")

#Calculate RMSE and add to tracker
Results_Tracker <- Results_Tracker %>% rbind(c("monmlp", RMSE(bike_test$cnt, predict_monmlp)))

# Try gbm
set.seed(123)
train_gbm <- train(cnt ~ . , method = "gbm", data = bike_train_machine)
predict_gbm <- predict(train_gbm, bike_test, type = "raw")

#Calculate RMSE and add to tracker
Results_Tracker <- Results_Tracker %>% rbind(c("gbm", RMSE(bike_test$cnt, predict_gbm)))

# Try svmRadial
set.seed(123)
train_svmRadial <- train(cnt ~ . , method = "svmRadial", data = bike_train_machine)
predict_svmRadial <- predict(train_svmRadial, bike_test, type = "raw")

#Calculate RMSE and add to tracker
Results_Tracker <- Results_Tracker %>% rbind(c("svmRadial", RMSE(bike_test$cnt, predict_svmRadial)))

# Try svmRadialCost
set.seed(123)
train_svmRadialCost <- train(cnt ~ . , method = "svmRadialCost", data = bike_train_machine)
predict_svmRadialCost <- predict(train_svmRadialCost, bike_test, type = "raw")

#Calculate RMSE and add to tracker
Results_Tracker <- Results_Tracker %>% rbind(c("svmRadialCost", RMSE(bike_test$cnt, predict_svmRadialCost)))

# Try svmRadialSigma
set.seed(123)
train_svmRadialSigma <- train(cnt ~ . , method = "svmRadialSigma", data = bike_train_machine)
predict_svmRadialSigma <- predict(train_svmRadialSigma, bike_test, type = "raw")

#Calculate RMSE and add to tracker
Results_Tracker <- Results_Tracker %>% rbind(c("svmRadialSigma", RMSE(bike_test$cnt, predict_svmRadialSigma)))

#Examine the importance of the variables identified by the RF approach
importance <- varImp(train_rf)
#plot the output
importance_plot <- importance %>% ggplot(aes(importance)) + geom_line()
#Save plot for use in RMarkdown
png(file="./importance_plot.png",
    width=900, height=500)
print(importance_plot)
dev.off()


Results_Tracker %>% ggplot(aes(factor(Method, levels = Method), round(as.numeric(RMSE),4))) + 
  geom_col(fill = "dark orange") + 
  theme(axis.text.x = element_text(angle = 55, vjust = 1, hjust=1, size =6)) + 
  labs(title = "Predictions and Associated RMSE", x= "Predictions", y= "RMSE", size =8) 


#Write files to disk for use in RMarkdown report
write.csv(Results_Tracker, file = "Results_Tracker.csv")




######################################################
# Manually predict - using linear regression and 
# importance criteria from RF training
######################################################

#Make second copy of bike test and train datasets

set.seed(1, sample.kind="Rounding")
ind <- createDataPartition(Hourly_Data$instant,0.2, times=1)
bike <- Hourly_Data[-ind$Resample1,]

set.seed(1, sample.kind="Rounding")
ind <- createDataPartition(bike$instant,0.3, times=1)
bike_train_2 <- bike[-ind$Resample1,]
bike_test_2 <- bike[ind$Resample1,]
rm(bike, ind)

#Use hour of day as predictor
hour_predictor <- bike_train_2 %>% group_by(hr) %>% summarise(hour_pred = mean(cnt))
bike_train_2 <- left_join(bike_train_2, hour_predictor, by = "hr")
bike_test_2 <- left_join(bike_test_2, hour_predictor, by = "hr")

#Add RMSE to results tracker
Results_Tracker_2 <- data.frame(Method = "Hour Only", RMSE = RMSE(bike_test_2$cnt,bike_test_2$hour_pred))
#Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Hour", RMSE(bike_test$cnt,(bike_test$mean_hires + bike_test$hour_pred))))

rm(hour_predictor)

#Incude temp as predictor

#Create new variable based on rounding temp to 1 sig digit
bike_train_2 <- bike_train_2 %>% mutate(temp_rounded = signif(bike_train_2$temp, 1))
bike_test_2 <- bike_test_2 %>% mutate(temp_rounded = signif(bike_test_2$temp, 1))

#Predict based on rounded temperature
temp_predictor <- bike_train_2 %>% group_by(temp_rounded) %>% summarise(temp_pred = mean(cnt - hour_pred))
bike_train_2 <- left_join(bike_train_2, temp_predictor, by = "temp_rounded")
bike_test_2 <- left_join(bike_test_2, temp_predictor, by = "temp_rounded")

#Add RMSE to results tracker
Results_Tracker_2 <- Results_Tracker_2 %>% rbind(c("Hour + Temp", RMSE(bike_test_2$cnt,(bike_test_2$hour_pred + bike_test_2$temp_pred))))

rm(temp_predictor)

#Incude year as predictor

year_predictor <- bike_train_2 %>% group_by(yr) %>% summarise(year_pred = mean(cnt - hour_pred - temp_pred))
bike_train_2 <- left_join(bike_train_2, year_predictor, by = "yr")
bike_test_2 <- left_join(bike_test_2, year_predictor, by = "yr")

#Add RMSE to results tracker
Results_Tracker_2 <- Results_Tracker_2 %>% rbind(c("Hour + Temp + Year", RMSE(bike_test_2$cnt,(bike_test_2$hour_pred + bike_test_2$temp_pred +
                                                                                                              bike_test_2$year_pred))))
rm(year_predictor)

#Incude workingday as predictor

workday_predictor <- bike_train_2 %>% group_by(workingday) %>% summarise(workday_pred = mean(cnt - hour_pred - temp_pred - year_pred))
bike_train_2 <- left_join(bike_train_2, workday_predictor, by = "workingday")
bike_test_2 <- left_join(bike_test_2, workday_predictor, by = "workingday")

#Add RMSE to results tracker
Results_Tracker_2 <- Results_Tracker_2 %>% rbind(c("Hour + Temp + Year + Working Day", RMSE(bike_test_2$cnt,(bike_test_2$hour_pred + bike_test_2$temp_pred +
                                                                                                                                     bike_test_2$year_pred + bike_test_2$workday_pred))))
rm(workday_predictor)

#Incude humidity as predictor

#Create new variable based on rounding humidity to 1 sig digit
bike_train_2 <- bike_train_2 %>% mutate(hum_rounded = signif(bike_train_2$hum, 1))
bike_test_2 <- bike_test_2 %>% mutate(hum_rounded = signif(bike_test_2$hum, 1))

#Predict based on rounded temperature
hum_predictor <- bike_train_2 %>% group_by(hum_rounded) %>% summarise(hum_pred = mean(cnt - hour_pred - temp_pred - year_pred - workday_pred))
bike_train_2 <- left_join(bike_train_2, hum_predictor, by = "hum_rounded")
bike_test_2 <- left_join(bike_test_2, hum_predictor, by = "hum_rounded")

#Add RMSE to results tracker
Results_Tracker_2 <- Results_Tracker_2 %>% rbind(c("Hour + Temp + Year + Working Day + Humidity", RMSE(bike_test_2$cnt,(bike_test_2$hour_pred + bike_test_2$temp_pred + bike_test_2$year_pred + bike_test_2$workday_pred + bike_test_2$hum_pred))))

rm(hum_predictor)

# Add final prediction to dataset
bike_test_2 <- bike_test_2 %>% mutate(estimate = bike_test_2$hour_pred + bike_test_2$temp_pred + bike_test_2$year_pred +
                                        bike_test_2$workday_pred + bike_test_2$hum_pred)

# Change any negative predictions to zero
bike_test_2$estimate <- ifelse(bike_test_2$estimate<0 , 0, bike_test_2$estimate)

# Add final RMSE
Results_Tracker_2 <- Results_Tracker_2 %>% rbind(c("Final Estimate (No negative estimates)", RMSE(bike_test_2$cnt,bike_test_2$estimate)))

######################
# Examine Errors
######################

# Add error figure to bike_test 
bike_test_2 <- bike_test_2 %>% mutate(error = cnt - estimate)

# Plot errors
bike_test_2 %>% ggplot(aes(as.factor(weathersit), error)) + geom_boxplot()
bike_test_2 %>% ggplot(aes(as.factor(signif(atemp,2)), error)) + geom_boxplot() 
bike_test_2 %>% ggplot(aes(as.factor(weekday), error)) + geom_boxplot()
bike_test_2 %>% ggplot(aes(as.factor(workingday),error)) + geom_boxplot()
bike_test_2 %>% ggplot(aes(as.factor(holiday), error)) + geom_boxplot()
bike_test_2 %>% ggplot(aes(as.factor(mnth), error)) + geom_boxplot()
bike_test_2 %>% ggplot(aes(as.factor(season), error)) + geom_boxplot()

bike_test_2 %>% ggplot(aes(cnt,error)) + geom_point() + geom_smooth()

corr <- bike_test_2 %>% select(3:14,17,26) %>% cor()
corr %>% corrplot(type = "upper")
rm(corr)



#Incude day as predictor
day_predictor <- bike_train %>% group_by(weekday) %>% summarise(day_pred = mean(cnt - mean_hires - hour_pred))
bike_train <- left_join(bike_train, day_predictor, by = "weekday")
bike_test <- left_join(bike_test, day_predictor, by = "weekday")

#Add RMSE to results tracker
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Hour + Day", RMSE(bike_test$cnt,(bike_test$mean_hires + bike_test$hour_pred + bike_test$day_pred))))

rm(day_predictor)






#Incude season as predictor

season_predictor <- bike_train %>% group_by(season) %>% summarise(season_pred = mean(cnt - mean_hires - hour_pred - day_pred - temp_pred - hum_pred - year_pred))
bike_train <- left_join(bike_train, season_predictor, by = "season")
bike_test <- left_join(bike_test, season_predictor, by = "season")

#Add RMSE to results tracker
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Hour + Day + Temp + Hum + Year + Season", RMSE(bike_test$cnt,(bike_test$mean_hires + bike_test$hour_pred + bike_test$day_pred + bike_test$temp_pred + bike_test$hum_pred +
                                                                                                                       bike_test$year_pred + bike_test$season_pred))))

rm(season_predictor)

#Incude workingday as predictor

workday_predictor <- bike_train %>% group_by(workingday) %>% summarise(workday_pred = mean(cnt - mean_hires - hour_pred - day_pred - temp_pred - hum_pred - year_pred - season_pred))
bike_train <- left_join(bike_train, workday_predictor, by = "workingday")
bike_test <- left_join(bike_test, workday_predictor, by = "workingday")

#Add RMSE to results tracker
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Hour + Day + Temp + Hum + Year + Season + Working Day", RMSE(bike_test$cnt,(bike_test$mean_hires + bike_test$hour_pred + bike_test$day_pred + bike_test$temp_pred + bike_test$hum_pred +
                                                                                                                                     bike_test$year_pred + bike_test$season_pred + bike_test$workday_pred))))

rm(workday_predictor)

#Incude weathersit as predictor

weathersit_predictor <- bike_train %>% group_by(weathersit) %>% summarise(weathersit_pred = mean(cnt - mean_hires - hour_pred - day_pred - temp_pred - hum_pred - year_pred - season_pred - workday_pred))
bike_train <- left_join(bike_train, weathersit_predictor, by = "weathersit")
bike_test <- left_join(bike_test, weathersit_predictor, by = "weathersit")

#Add RMSE to results tracker
Results_Tracker <- Results_Tracker %>% rbind(c("Mean + Hour + Day + Temp + Hum + Year + Season + Working Day + Weathersit", RMSE(bike_test$cnt,(bike_test$mean_hires + bike_test$hour_pred + bike_test$day_pred + bike_test$temp_pred + bike_test$hum_pred +
                                                                                                                                                  bike_test$year_pred + bike_test$season_pred + bike_test$workday_pred + bike_test$weathersit_pred))))

rm(weathersit_predictor)

#Write files to disk for use in RMarkdown report
write.csv(Results_Tracker, file = "Results_Tracker.csv")



#####################################################
# Apply trained RF to the validation dataset
#####################################################

set.seed(123)
predict_rf_validation <- predict(train_rf, validation, type = "raw")

#Calculate RMSE 
RMSE_RF_Validation <- RMSE(validation$cnt, predict_rf_validation)

# Add error measurement to validation dataset then examine 

validation <- validation %>% mutate(error = cnt - predict_rf_validation)

# Plot errors
validation %>% ggplot(aes(as.factor(weathersit), error)) + geom_boxplot()
validation %>% ggplot(aes(as.factor(signif(atemp,2)), error)) + geom_boxplot() 
validation %>% ggplot(aes(as.factor(weekday), error)) + geom_boxplot()
validation %>% ggplot(aes(as.factor(workingday),error)) + geom_boxplot()
validation %>% ggplot(aes(as.factor(holiday), error)) + geom_boxplot()
validation %>% ggplot(aes(as.factor(mnth), error)) + geom_boxplot()
validation %>% ggplot(aes(as.factor(season), error)) + geom_boxplot()

validation %>% ggplot(aes(cnt,error)) + geom_point() + geom_smooth()

corr <- validation %>%  select(-2) %>% cor()
corr %>% corrplot(type = "upper")
rm(corr)

#save training, test and validation datasets for use in RMarkdown
write.csv(validation, file = "Validation.csv")
write.csv(bike_train, file = "Bike_Train.csv")
write.csv(bike_test, file = "Bike_Test.csv")

