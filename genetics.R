# Gideon Vos 20 Oct 2020
# Implementation of paper: Deep learning to predict the lab-of-origin of engineered DNA
# Nielsen, A and Voigt, C (Voigt Labs)
# URL: https://www.nature.com/articles/s41467-018-05378-z.pdf?origin=ppub
# Paper results in predictive accuracy of 48%, my model delivers 70% on validation set (10%)

# NOTE: The dataset used here provided from this URL:
# https://www.drivendata.org/competitions/63/genetic-engineering-attribution
# 60,000 DNA sequences with lab of origin as label

# Training time: 157 mins for 25 epochs

library(data.table)
library(stringr)
library(keras)
library(dplyr)
library(tidyverse)

# DNA sequence one-hot encoding routines:
library(XalBio)  # devtools::install_github("https://github.com/xalentis/XalBio")

# compress one-hot vector and decompress as required, reduces memory
# footprint from 29G down to 7G
library(XalUtil) # devtools::install_github("https://github.com/xalentis/XalUtil")

# data from: https://www.drivendata.org/competitions/63/genetic-engineering-attribution/
data <- fread("train_values.csv", data.table = TRUE)
labels <- fread("train_labels.csv", data.table = TRUE)

# convert data to dataframe, clean it up, convert to category
labels$sequence_id <-NULL
j1 <- max.col(labels, "first")
labels <- names(labels)[j1]
data$lab_id <- labels
rownames(data) <- NULL
rm(labels, j1)
gc()

data <- data %>% group_by(lab_id) %>% mutate(lab_count = n())
data <- data %>% filter(lab_count > 9) # remove labs with less than 9 samples as per paper
classes_data <- data %>% select(lab_count, lab_id)
data$lab_count <- NULL
labels <- data$lab_id
data$lab_id <- NULL

label_factors <- factor(labels)
Y <- to_categorical(as.numeric(label_factors), num_classes = length(levels(label_factors))+1)

# calculate class weights for unbalanced data
calculate_weights <- FALSE

if (calculate_weights == TRUE)
{
  names(classes_data) <- c("Count","Label")
  class_labels <- list()
  class_weight <- list()
  
  index <- 1
  max_count <- 0
  for (row in 1:nrow(classes_data)) {
    if (index == 1)
    {
      class_labels[index] <- classes_data[row, 'Label']
      class_weight[index] <- 1
      max_count <-classes_data[row, 'Count']
      index <- index + 1
    }
    else
    {
      class_labels[index] <- classes_data[row, 'Label']
      class_weight[index] <- max_count / classes_data[row, 'Count']
      index <- index + 1
    }
  }
  # save to disk when done so next time we can avoid and
  # just load the prepared features
  saveRDS(class_labels, file = "class_labels.rds")
  saveRDS(class_weight, file = "class_weights.rds")
} else
{
  class_labels <- readRDS(file = "class_labels.rds")
  class_weight <- readRDS(file = "class_weights.rds")
}

############################### DNA encoding to one-hot ###############################
one_hot_encode<-function(data, start, block_size)
{
  data_matrix <- array(0,dim = c(block_size, 16048))

  for (row in start:(start + (block_size - 1))) 
  {
    sequence <- data[row,"sequence"][[1]]

    if (nchar(sequence) > 8000)
    {
      sequence <- substring(sequence, first = 1, last = 8000)
    }

    reverse <- reverse_compliment(sequence)
    reverse <- str_pad(reverse, 8000, side = c("right"), pad = "N")
    sequence <- str_pad(sequence, 8000, side = c("right"), pad = "N")
    one_hot <- encode_sequence(sequence, reverse,48)
    data_matrix[row,] <- compress_block(one_hot, 16048, 4)
  }
  
  return(data_matrix)
}

total <- nrow(data)
classes <- dim(Y)[2]
block_size <- nrow(data) # paper used 6 chunks, we use 1 single large chunk
must_encode <- FALSE

# one-hot encode DNA sequences and save to file for re-use
# takes 10 mins for 60k sequences of length 8000
# save to disk when done so next time we can avoid and
# just load the prepared features
if (must_encode == TRUE)
{
  x_train <- one_hot_encode(data, 1, block_size) # 9 mins approx
  y_train <- Y
  saveRDS(x_train, file = "x_chunk.rds")
  saveRDS(y_train, file = "y_chunk.rds")
} else
{
  x_train <- readRDS(file = "x_chunk.rds")
  y_train <- readRDS(file = "y_chunk.rds")
}

# clean up
rm(data, classes_data, Y, labels, label_factors, calculate_weights, must_encode, block_size)
gc()

###################################################################################################################################
# Model starts here
###################################################################################################################################
set.seed(42)
epochs <- 40 # 100 in paper
batch_size <- 64 # 8 in paper

# generator to sample a random batch of training data with labels
sampling_generator <- function(X_data, Y_data, batch_size) {
  function() {
    rows <- sample(1:nrow(X_data), batch_size, replace = TRUE)
    data_matrix <- array(0,dim = c(length(rows), 16048, 4))
    index <- 1
    for (row in rows)
    {
      decompressed <- decompress_block(X_data[row,], 16048, 4)
      data_matrix[index,,] <- decompressed
      index <- index + 1
    }
    return (list(data_matrix, Y_data[rows,]))
  }
}

# keep 10% for validation
ind <- sample(nrow(x_train), round(nrow(x_train) * 0.90, 0), replace = FALSE)
x_valid <- x_train[-ind,]
x_train <- x_train[ind,]
y_valid <- y_train[-ind,]
y_train <- y_train[ind,]

train_generator <- sampling_generator(x_train, y_train, batch_size = batch_size)
valid_generator <- sampling_generator(x_train, y_train, batch_size = batch_size)

rm(ind)
gc()

model <- keras_model_sequential()

# model below follows paper, except for doubling of network capacity
# at convolution and dense layers and kernel size
# Train on full batch where the paper used 6 mini-batches
# kernel size set to 24 based on results from this paper that
# finds ideal kmer length of 18 to 24:
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0228070
model %>% 
  layer_conv_1d(filters=256, input_shape=c(16048,4), kernel_size = 24, activation="relu", padding="same") %>%
  layer_max_pooling_1d(pool_size = 16048) %>%
  layer_batch_normalization() %>%
  layer_flatten() %>%
  layer_dense(units=128, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dense(units = classes, activation = "softmax")

# review neural network model
summary(model)

model %>% compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = c("accuracy"))

# train with 70% random samples, test on 30% random samples
hist <- model %>% fit_generator(train_generator,epochs = epochs,
  steps_per_epoch = as.integer(total / batch_size),
  validation_data = valid_generator,
  validation_steps = as.integer((total * 0.3) / batch_size), # 0.3 = test on 30% after each epoch
  verbose=1,
  class_weight=list(class_labels, class_weight),
  callbacks = list(
    # save best model after every epoch
    callback_model_checkpoint("checkpoint.h5", save_best_only = TRUE),
    # stop training when val loss stops improving after 2 epochs
    callback_early_stopping(monitor = "val_loss",patience = 2,verbose = 1,mode = c("min"))
  )
) # 6 minutes per epoch

model %>% save_model_hdf5("final_model.h5")

model <- load_model_hdf5("checkpoint.h5") # load best model

rm(x_train, y_train)
gc()

# decompress validation data to score
index <- 1
data_matrix <- array(0,dim = c(dim(x_valid)[1], 16048, 4))
for (row in 1:dim(x_valid)[1])
{
  decompressed <- decompress_block(x_valid[row,], 16048, 4)
  data_matrix[index,,] <- decompressed
  index <- index + 1
}
x_valid <- data_matrix
rm(data_matrix)
gc()

score <- model %>% evaluate(x_valid, y_valid, batch_size = batch_size) # 70%

# Top-n accuracy scorer
topn <- function(vector, n){
  maxs <- c()
  ind <- c()
  for (i in 1:n){
    biggest <- match(max(vector), vector)
    ind[i] <- biggest
    maxs[i] <- max(vector)
    vector <- vector[-biggest]
  }
  mat <- cbind(maxs, ind)
  return(mat)
}

# Top-n accuracy scorer
top_scorer <- function(x, y, model, n)
{
  scores <- array(0, dim = c(dim(x_valid)[1], 1)) 
  probas <- predict_proba(model, x_valid, batch_size = batch_size)
  for (pred in 1:(dim(x_valid)[1]))
  {
    # get top n probabilities
    b<-topn(probas[pred,], n)[,2]
    # check if y truth is in the top n
    top_n <- (which.max(y_valid[pred,]) %in% b)
    scores[pred] <- top_n
  }
  # return overall mean as result
  return (mean(scores))
}

top_scorer(x_valid, y_valid, model, 10) # 76.4% in top 10
top_scorer(x_valid, y_valid, model, 3)  # 75.2% in top 3
