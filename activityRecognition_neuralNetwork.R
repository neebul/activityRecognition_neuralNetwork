# Libraries needed
library(keras)
library(tidyverse)
library(feather)
library(caret)
library(fastDummies)
library(e1071)
library(plyr)



# Load the data
df <- read_feather('your_file.feather') %>% 
  mutate(activity_column = fct_drop(activity_column))

data <- df %>% select(-notXorYcategory, -notXorYcategory)   # just keep y and x values in the dataset



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
  y_test_categories <- test[, 1]
  
  
  # Reshape y as one-hot matrices
  y_train <- dummy_cols(y_train, select_columns = "activity")
  y_train <- y_train[, 2:ncol(y_train)]
  
  y_test <- dummy_cols(y_test_categories, select_columns = "activity")
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
  
  
  
  # Neural network - Create the model 
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
  
  
  
  # Neural network - Train the model 
  history <- model.nn %>% fit(
    x_train, y_train, 
    epochs = 30, 
    batch_size = 1500,
    validation_split = 0.1,
    shuffle = TRUE
  )
  
  
  
  # Analyze the efficiency of the model
  y_nn_predict <- model.nn %>% predict(x_test)
  y_nn_predict_categories <- max.col(y_nn_predict)
  y_nn_predict_categories <- as.factor(category_nn_predict_categories)
  category_nn_predict <- revalue(category_nn_predict, c("1"="activity1", "2"="activity2", "3"="activity3", "4"="activity4", "5"="activity5"))
  levels(category_nn_predict) <- c("activity1", "activity2", "activity3", "activity4", "activity5")
  
  caret::confusionMatrix(y_test_categories, y_nn_predict_categories)
  

  
  # Find the errors made by the model
  indices_errors <- which(category_nn_predict != category_test)
  errors_categories <- category_nn_predict[indices_errors]
  true_categories <- category_test[indices_errors]
  probabilities_errors <- y_nn_predict[indices_errors,]
  
  errors_nn <- cbind(indices_errors, errors_categories, true_categories, probabilities_errors)
  colnames(errors_nn) <- c("index", "error_cat", "true_cat", "p-sitting", "p_lying", "p_standing", "p_walking", "p_running")
  
}
