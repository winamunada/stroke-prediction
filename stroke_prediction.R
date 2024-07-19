---
title: "Build and deploy a stroke prediction model using R"
output: html_document
author: "Wina Munada"
---
  
  # About Data Analysis Report
  
  This RMarkdown file contains the report of the data analysis done for the project on building and deploying a stroke prediction model in R. It contains analysis such as data exploration, summary statistics and building the prediction models. The final report was completed on `r date()`. 

**Data Description:**
  
  According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.

This data set is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relevant information about the patient.


# Task One: Import data and data preprocessing

## Load data and install packages

```{r}
library('tidyverse')
library('ggplot2')
library('dplyr')
library('caret')
library('randomForest')
library('skimr')
library('gridExtra')
library('caTools')
library('corrplot')
library('ggcorrplot')
library('naniar')

setwd("C:\Project")
Data_Stroke <- read.csv('healthcare-dataset-stroke-data.csv')

## Describe and explore the data

summary(Data_Stroke)
glimpse(Data_Stroke)
skim(Data_Stroke)
miss_scan_count(data = Data_Stroke, search = list("Unknown","N/A","Other"))

##Convert NA to median in BMI
Data_Stroke$bmi <- as.numeric(Data_Stroke$bmi)
Data_Stroke$bmi[is.na(Data_Stroke$bmi)] <- median(Data_Stroke$bmi, na.rm = TRUE)

##Check duplicates
sum(duplicated(Data_Stroke))
colSums(Data_Stroke == 'N/A')
colSums(Data_Stroke == '')
Data_Stroke %>% count(gender)

##Remove ID and filter out 'Other' values in Gender
Data_Stroke <- Data_Stroke %>% 
  select(-c(id)) %>% 
  filter(gender != "Other")
str(Data_Stroke)

##Convert non-numeric variables to factors
Data_Stroke$stroke <- factor(Data_Stroke$stroke, levels = c(0,1), labels = c("No", "Yes"))
Data_Stroke$hypertension <- factor(Data_Stroke$hypertension, levels = c(0,1), labels = c("No", "Yes"))
Data_Stroke$heart_disease <- factor(Data_Stroke$heart_disease, levels = c(0,1), labels = c("No", "Yes"))


# Task Two: Build prediction models

d1 <- Data_Stroke %>%
  ggplot(aes(x = gender, fill = gender)) +
  geom_bar(fill = c("chocolate4", "bisque4")) +
  ggtitle("Gender Distribution") +
  geom_text(aes(label=..count..), stat = "Count", vjust = 1.0)

d2 <- Data_Stroke %>%
  ggplot(aes(x = hypertension, fill = hypertension)) +
  geom_bar(fill = c("chocolate4", "bisque4")) +
  ggtitle("Hypertenstion Distribution") +
  geom_text(aes(label=..count..), stat = "Count", vjust = 1.0)


d3 <- Data_Stroke %>%
  ggplot(aes(x = heart_disease, fill = heart_disease)) +
  geom_bar(fill = c("chocolate4", "bisque4")) +
  ggtitle("Heart Disease Distribution") +
  geom_text(aes(label=..count..), stat = "Count", vjust = 1.0)

d4 <- Data_Stroke %>%
  ggplot(aes(x = ever_married, fill = ever_married)) +
  geom_bar(fill = c("chocolate4","bisque4")) +
  ggtitle("Married distribution") +
  geom_text(aes(label=..count..), stat = "Count", vjust = 1.0)

d5 <- Data_Stroke %>%
  ggplot(aes(x = work_type, fill = work_type)) +
  geom_bar(fill = c("chocolate4", "bisque4","cornsilk2","darkorange3","burlywood3")) +
  ggtitle("Work type distribution") +
  geom_text(aes(label=..count..), stat = "Count", vjust = 1.0)

d6 <- Data_Stroke %>%
  ggplot(aes(x = stroke, fill = stroke)) +
  geom_bar(fill = c("chocolate4", "bisque4")) +
  ggtitle("Stroke distribution") +
  geom_text(aes(label=..count..), stat = "Count", vjust = 1.0)

d7 <- Data_Stroke %>%
  ggplot(aes(x = Residence_type, fill = Residence_type)) +
  geom_bar(fill = c("chocolate4", "bisque4")) +
  ggtitle("Residence distribution") +
  geom_text(aes(label=..count..), stat = "Count", vjust = 1.0)


grid.arrange(d1,d2,d3,d4,d5,d6,d7, ncol=2)


Data_Stroke %>%
  ggplot(aes(x = gender, fill = stroke)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values=c("chocolate4",
                             "bisque4")) +
  ggtitle("Gender vs. Stroke") 

Data_Stroke %>%
  ggplot(aes(x = hypertension, fill = stroke)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values=c("chocolate4",
                             "bisque4")) +
  ggtitle("Hypertension vs. Stroke")

Data_Stroke %>%
  ggplot(aes(x = heart_disease, fill = stroke)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values=c("chocolate4",
                             "bisque4")) +
  ggtitle("Heart disease vs. Stroke") 

Data_Stroke %>%
  ggplot(aes(x = Residence_type, fill = stroke)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values=c("chocolate4",
                             "bisque4")) +
  ggtitle("Residence type vs. Stroke")

Data_Stroke %>%
  ggplot(aes(x = smoking_status, fill = stroke)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values=c("chocolate4",
                             "bisque4")) +
  ggtitle("Smoking status vs. Stroke")


Data_Stroke %>%
  ggplot(aes(x = work_type, fill = stroke)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values=c("chocolate4",
                             "bisque4"
  )) +
  ggtitle("Type of Work vs. Stroke")


Data_Stroke %>%
  ggplot(aes(x = avg_glucose_level, fill = stroke)) +
  geom_density(alpha = 0.7) +
  scale_fill_manual(values=c("chocolate4",
                             "bisque4"
  )) +
  ggtitle("Average Glucose level vs. Stroke") 

Data_Stroke %>% filter(between(bmi, 0, 60)) %>%
  ggplot(aes(x = bmi, fill = stroke)) +
  geom_density(alpha = 0.7) +
  scale_fill_manual(values=c("chocolate4",
                             "bisque4"
  )) +
  ggtitle("Body Mass Index vs. Stroke")


# Task Three: Evaluate and select prediction models

set.seed(123)
split_tag <- sample.split(Data_Stroke$stroke, SplitRatio = 0.8)
train <- subset(Data_Stroke, split_tag == TRUE)
test <- subset(Data_Stroke, split_tag == FALSE)
dim(train)
dim(test)

# Task Four: Deploy the prediction model

# Ensure there are no missing values in the training set
train <- na.omit(train)

# Hyperparameter tuning using caret with expanded parameters
control <- trainControl(method = "cv", number = 5, search = "random")

# Define the tuning grid with a wider range of mtry and ntree
tuneGrid <- expand.grid(.mtry = c(2, 3, 4, 5, 6, 7, 8, 9, 10))

# Train the random forest model with hyperparameter tuning
set.seed(123)
rf_model <- train(stroke ~ ., 
                  data = train, 
                  method = "rf", 
                  trControl = control, 
                  tuneGrid = tuneGrid, 
                  ntree = 500)  # Setting a higher number of trees

# Print and plot the model
print(rf_model)
plot(rf_model)

# Evaluate the model
predictions <- predict(rf_model, newdata = test)
confusionMatrix(predictions, test$stroke)


# Task Five: Findings and Conclusions
As depicted above, our model boasts an accuracy rate exceeding 95%, indicating that it underwent effective training.

