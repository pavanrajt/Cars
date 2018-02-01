#import the required packages
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn import preprocessing

#laod the car dataset
car = pd.read_csv('E:/PGDM/Assignment/BA03 -1 Assignments/Car/car1.csv', sep=",")
car.describe()

#delete the missing values
car1 = car.dropna()

#create a new dataset with the numeric variables
car2 = car1[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']]

#The variable horsepower has been set as a string variable. Convert this variable to numeric.
car2['horsepower'] = car2['horsepower'].convert_objects(convert_numeric=True)

#check for missing values and save all the rows with missing values into a new dataset and then drop the missing values
null_data = car2[car2.isnull().any(axis=1)]
car3 = car2.dropna()

#apply normalisation on this new dataset and create a new dataset with the results
normal_car3 = preprocessing.normalize(car3)
car4 = pd.DataFrame(normal_car3)
car4.columns = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']

#create new dataset with all the non numeric variables and delete the rows that were deleted in the normalised datset
car5 = car1[['cylinders', 'modelyr', 'origin', 'car name', 'car_company']]
car6 = car5.drop(car5.index[[null_data.index]])

#join the numeric and non-numeric datasets to create the final dataset
car7 = car4.join(car6)

#apply logistic resgression model on the final data by declaring the categorical variables
ols_car = smf.ols(formula='mpg~displacement+horsepower+weight+acceleration+C(cylinders)+C(modelyr)+C(origin)+C(car_company)',data=car7).fit()

#check the summary to verify the R-squared value and the co-efficient values and intercept
ols_car.summary()

#predict mpg using the model
ols_car.predict()
