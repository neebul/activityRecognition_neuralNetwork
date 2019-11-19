# Libraries needed and working directory 
library(keras)
library(tidyverse)
library(feather)
library(caret)
library(fastDummies)
library(e1071)
library(plyr)

setwd('your_working_directory')



# Load the data
df <- read_feather('your_file') %>% 
  mutate(activity_column = fct_drop(activity_column))

data <- df %>% select(-notXorYcategory, -notXorYcategory)



# Create evaluation variable 
evaluation <- matrix(0, nrow = length(unique(data$id)), ncol = 1)



# Create, train and evaluate the models
step <- 0

for (i in unique(data$id)) {
  
  step <- step + 1
  cat('Testing step:', step, " ")
  
  
  # Select train and test data
  train <- data %>% filter(id != i) %>% select(-id)
  test <- data %>% filter(id == i) %>% select(-id)
  
  
  # Select x and y data
  x_train <- train[, 2:ncol(train)]
  y_train <- train[, 1]
  
  x_test <- test[, 2:ncol(test)]
  y_test <- test[, 1]
  
  
  # Reshape y as binary elements (factor -> binary)
  y_train <- dummy_cols(y_train, select_columns = "activity")
  y_train <- y_train[, 2:ncol(y_train)]
  
  y_test <- dummy_cols(y_test, select_columns = "activity")
  y_test <- y_test[, 2:ncol(y_test)]
  
  
  # Transform every variable into clean matrices 
  x_train <- as.matrix(x_train)
  dimnames(x_train) <- NULL
  
  y_train <- as.matrix(y_train)
  dimnames(y_train) <- NULL
  
  x_test <- as.matrix(x_test)
  dimnames(x_test) <- NULL
  
  y_test <- as.matrix(y_test)
  dimnames(y_test) <- NULL
  
 
  # Feature scaling 
  mean_train <- matrix(0, nrow = ncol(x_train), ncol = 1)
  sd_train <- matrix(0, nrow = ncol(x_train), ncol = 1)
  
  for (i in 1:ncol(x_train)) {
    mean_train[i] <- mean(x_train[,i])
    sd_train[i] <- sd(x_train[,i])
  }
  
  for (i in 1:ncol(x_train)) {
    x_train[,i] <- (x_train[,i] - mean_train[i]) / sd_train[i]
  }
  
  for (i in 1:ncol(x_test)) {
    x_test[,i] <- (x_test[,i] - mean_train[i]) / sd_train[i]
  }
  
  
  
  # NN - Create the model 
  model.nn <- keras_model_sequential() 
  model.nn %>% 
    layer_dense(units = 50, activation = 'relu', input_shape = c(number_of_features), kernel_regularizer = regularizer_l2(l = 0)) %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(units = 50, activation = 'relu', kernel_regularizer = regularizer_l2(l = 0)) %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = 5, activation = 'softmax')
  
  model.nn %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )
  
  
  
  # NN - Train the model 
  history <- model.nn %>% fit(
    x_train, y_train, 
    epochs = 30, 
    batch_size = 1500,
    validation_split = 0.1,
    shuffle = TRUE
  )
  
  
  
  # Observe probabilities of predicted outcomes 
  y_nn_predict <- model.nn %>% predict(x_test)
  
  category_nn_predict <- max.col(y_nn_predict)
  category_nn_predict <- as.factor(category_nn_predict)
  category_nn_predict <- revalue(category_nn_predict, c("1"="Sitting", "2"="Lying", "3"="Standing", "4"="Walking", "5"="Running"))
  levels(category_nn_predict) <- c("Sitting", "Lying", "Standing", "Walking", "Running")
  
  category_test <- max.col(y_test)
  category_test <- as.factor(category_test)
  category_test <- revalue(category_test, c("1"="Sitting", "2"="Lying", "3"="Standing", "4"="Walking", "5"="Running"))
  levels(category_test) <- c("Sitting", "Lying", "Standing", "Walking", "Running")
  
  accuracy_nn <- 1 - length(which(category_nn_predict != category_test)) / nrow(test)
 
  evaluation[step] <- accuracy_nn
  
  errors_nn <- cbind(which(category_nn_predict != category_test), category_nn_predict[which(category_nn_predict != category_test)], 
                     category_test[which(category_nn_predict != category_test)], y_nn_predict[which(category_nn_predict != category_test),])

  }
