url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart_data <- read.table(url, sep = ",", header = FALSE)
colnames(heart_data) <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target")
str(heart_data)#give data types of column
head(heart_data)#print first five rows
print(heart_data)#print all
View(heart_data)


#install.packages("reshape2")
library(reshape2)

#install.packages("ggplot2")
library(ggplot2)

# Histogram of age distribution colored by heart disease status
#install.packages('plotly')
library(plotly)
age_histogram <- ggplot(heart_data, aes(x = age, fill = factor(target))) +
  geom_histogram(binwidth = 5, position = "dodge", alpha = 0.7) +
  labs(title = "Age Distribution by Heart Disease Status",
       x = "Age", y = "Count") +
  scale_fill_discrete(name = "Heart Disease", labels = c("No", "Yes", "Safe", "Moderate", "Severe")
print(levels(heart_data$target))
interactive_age_histogram <- ggplotly(age_histogram)
interactive_age_histogram


# Histogram of sex distribution colored by heart disease status(ggplot)
sex_distribution <- ggplot(heart_data, aes(x = factor(sex), fill = factor(target))) +
  geom_bar(position = "dodge", alpha = 0.7) +
  labs(title = "Sex Distribution by Heart Disease Status",
       x = "Sex", y = "Count") +
  scale_fill_discrete(name = "Heart Disease", labels = c("No", "Yes", "Safe", "Moderate", "Severe"))
interactive_sex_distribution <- ggplotly(sex_distribution)
print(interactive_sex_distribution)

# Histogram of cholesterol levels distribution colored by heart disease status(ggplot)
cholesterol_distribution <- ggplot(heart_data, aes(x = chol, fill = factor(target))) +
  geom_histogram(binwidth = 50, position = "dodge", alpha = 0.7) +
  labs(title = "Cholesterol Levels Distribution by Heart Disease Status",
       x = "Cholesterol Levels", y = "Count") +
  scale_fill_discrete(name = "Heart Disease", labels = c("No", "Yes", "Safe", "Moderate", "Severe"))
interactive_cholesterol_distribution <- ggplotly(cholesterol_distribution)
interactive_cholesterol_distribution


#summary of age 
sum(is.na(heart_data))
print("Summary of age:")
summary(heart_data$age)
summary(heart_data$chol)



#correlation Analysis with target
correlation_matrix <- cor(heart_data[, c("age", "trestbps", "chol", "thalach", "oldpeak", "target")])
correlation_with_target <- correlation_matrix["target", ]
print(correlation_with_target)


# Visualize correlation with target(ggplot)
correlation_df <- as.data.frame(correlation_with_target)
correlation_graph <- ggplot(correlation_df, aes(x = rownames(correlation_df), y = correlation_with_target)) +
  geom_bar(stat = "identity", fill = "skyblue", width = 0.5) +
  labs(title = "Correlation with Target Variable",
       x = "Variable", y = "Correlation(heart disease)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
interactive_correlation_graph <- ggplotly(correlation_graph)
interactive_correlation_graph

# Correlation with other variables

#install.packages("reshape2")
library(reshape2)
correlation_matrix <- cor(heart_data[, c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "target")])
print(correlation_matrix)


# Visualize correlation with with other variables(ggplot)
correlation_long <- melt(correlation_matrix)
correlation_long_graph <- ggplot(correlation_long, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0, limits = c(-1, 1), name = "Correlation") +
  theme_minimal() +
  labs(title = "Correlation Matrix Heatmap",
       x = "Variables",
       y = "Variables")
interactive_correlation_long <- ggplotly(correlation_long_graph)
interactive_correlation_long

# Summary statistics for numerical variables

summary_stats_numeric <- sapply(heart_data[, c("age", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal")], summary)
print(summary_stats_numeric)

# Summary statistics for numerical variables(ggplot)

#install.packages("gridExtra")
library(gridExtra)
numeric_vars <- c("age", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal")

histograms <- lapply(numeric_vars, function(var) {
  ggplot(heart_data, aes(x = !!sym(var))) +
    geom_histogram(binwidth = 5, fill = "skyblue", color = "black", stat = "count") +
    labs(title = paste("Histogram of", var),
         x = var, y = "Frequency(people)")
})

grid.arrange(grobs = histograms, ncol = 2)




# Summary statistics for categorical variables

summary_stats_categorical <- sapply(heart_data[, c("sex", "cp", "fbs", "restecg", "exang", "slope", "target")], table)
print(summary_stats_categorical)


#Summary statistics for categorical variables(ggplot)

categorical_vars <- c("sex", "cp", "fbs", "restecg", "exang", "slope", "target")
bar_plots <- lapply(categorical_vars, function(var) {
  ggplot(heart_data, aes(x = factor(!!sym(var)))) +
    geom_bar(fill = "skyblue", color = "black") +
    labs(title = paste("Bar Plot of", var),
         x = var, y = "Frequency(people)") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
})
gridExtra::grid.arrange(grobs = bar_plots, ncol = 2)






#data modelling


heart_data$ca <- as.numeric(heart_data$ca)
heart_data$thal <- as.numeric(heart_data$thal)
heart_data$ca[is.na(heart_data$ca)] <- 0.0

# Fill missing values in 'thal' column with 0.0
heart_data$thal[is.na(heart_data$thal)] <- 0.0
print(heart_data)
sum(is.na(heart_data))




#install.packages("caret")
#install.packages("randomForest")
#install.packages("e1071")
#Load necessary libraries
#install.packages("MLmetrics")
library(caret)
library(randomForest)
library(e1071)
library(MLmetrics)
set.seed(123)

# Convert target variable to factor
heart_data$target <- as.factor(heart_data$target)
# Inspect current levels of the target variable
levels(heart_data$target)

# If needed, modify the levels of the target variable
levels(heart_data$target) <- c("No", "Yes", "Safe", "Moderate", "Severe")
str(heart_data)
set.seed(123)  # for reproducibility
split <- createDataPartition(heart_data$target, p = 0.8, list = FALSE)
train_data <- heart_data[split, ]
test_data <- heart_data[-split, ]
# Example: Random Forest
model_rf <- train(target ~ ., data = train_data, method = "rf",
                  trControl = trainControl(method = "cv", number = 5),
                  tuneGrid = expand.grid(mtry = c(2, 3, 4)))
# Evaluate model
predictions <- predict(model_rf, newdata = test_data)
conf_matrix <- confusionMatrix(data = predictions, reference = test_data$target)
print(conf_matrix)
# Check if the model is predicting only one class
table(predictions)

conf_matrix_df <- as.data.frame(conf_matrix$table)
# Rename columns 
colnames(conf_matrix_df) <- c("Predicted", "Actual", "Count")

#ggplot2 confusion materix modelling
confusion_matric_graph <- ggplot(data = conf_matrix_df, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), color = "black", size = 12) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix",
       x = "Predicted",
       y = "Actual")

#printing the interactive confusion matrix
interactive_confusion_matric <- ggplotly(confusion_matric_graph)
interactive_confusion_matric

#clean
heart_data_clean <- heart_data[!apply(heart_data == "?", 1, any), ]
heart_data_clean$sex <- ifelse(heart_data_clean$sex == 0, "female", "male")
heart_data_clean$target <- factor(heart_data_clean$target, levels = c(0, 1,2,3,4), labels = c("No", "Yes", "Safe", "Moderate", "Severe"))
str(heart_data_clean)
head(heart_data_clean)