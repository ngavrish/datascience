dataset = read.csv('Data.csv')
# dataset = dataset[, 2:3]

# splitting the dataset into training set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])
