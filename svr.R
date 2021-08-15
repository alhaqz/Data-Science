# Regrerssion Template

# Importing the dataset
dataset <- read.csv('Position_Salaries.csv')
dataset <- dataset[2:3]

# # Feature Scaling
# training_set <- scale(training_set)
# test_set <- scale(test_set)


# Fitting SVR to the dataset
library(e1071)
regressor <- svm(formula = Salary ~ .,
                 data = dataset,
                 type = 'eps-regression')

# Predicting a new result with SVR
y_pred <- predict(regressor, data.frame(Level = 6.5))

# Visualising the SVR result
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
             colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')



