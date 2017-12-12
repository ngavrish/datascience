# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)

# test_set = scale(test_set)

# Simple Linear Regression
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
y_pred = predict(regressor, newdata = test_set)

# Visualizing
library(ggplot2)
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, 
               y = test_set$Salary),
           color = 'red') +
  geom_line(aes(x = test_set$YearsExperience, 
                y =predict(regressor, newdata = test_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience (Trainig Set') +
  xlab('Exp') +
  ylab('Salary')



