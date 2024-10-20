''' House price prediction model using multiple linear Regression,ridge Regression and lasso Regression'''
from hashlib import algorithms_available
import numpy as np
import pandas as pd
import sklearn
import sklearn
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score,mean_squared_error

class PREDICT:
    def __init__(self,path):
        try:
            # Read the data
            self.df = pd.read_csv(path)

            # Mapping categorical values first
            self.df['Location'] = self.df['Location'].map({'Downtown': 0, 'Suburban': 1, 'Urban': 2, 'Rural': 3}).astype(int)
            self.df['Condition'] = self.df['Condition'].map({'Excellent': 0, 'Good': 1, 'Fair': 2, 'Poor': 3}).astype(int)
            self.df['Garage'] = self.df['Garage'].map({'Yes': 0, 'No': 1}).astype(int)

            # Dropping 'Id' and splitting the data into X and y
            self.df = self.df.drop(['Id'], axis=1)
            self.X = self.df.iloc[:, :-1]  # independent variables
            self.y = self.df.iloc[:, -1]  # dependent variable (price or target)
            #splitting the data
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=20)
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> {error_msg}')
    def mul_reg(self):
        #using multiple linear regression
        try:
            self.mul_reg=LinearRegression()
            self.mul_reg.fit(self.X_train,self.y_train)  #training the algorithm
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> {error_msg}')

    def rid_reg(self):
        #using Ridge Regression
        try:
            self.rid_reg=Ridge()
            self.rid_reg.fit(self.X_train,self.y_train)  #training the algorithm
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> {error_msg}')
    def las_reg(self):
        #using Lasso Regression
        try:
            self.las_reg=Lasso()
            self.las_reg.fit(self.X_train,self.y_train)  #training the algorithm
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> {error_msg}')


    def train_perfom(self):
        #checking model performance with trained data
        try:
            self.y_train_pred = self.mul_reg.predict(self.X_train)
            print('Trained data:')
            print(f'Loss by mean_squared_error : {mean_squared_error(self.y_train, self.y_train_pred)}')
            print(f'training perfomance accuracy(MLR) : {r2_score(self.y_train,self.y_train_pred)}')
            print(f'training perfomance accuracy(Ridge) : {self.rid_reg.score(self.X_train,self.y_train)}')
            print(f'training perfomance accuracy(Lasso) : {self.las_reg.score(self.X_train,self.y_train)}')
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> {error_msg}')

    def test_data(self):
        # checking model performance with test data
        try:
            self.y_test_pred = self.mul_reg.predict(self.X_test)
            print('Test data:')
            print(f'Loss by mean_squared_error : {mean_squared_error(self.y_test, self.y_test_pred)}')
            print(f'test perfomance accuracy(MLR) : {r2_score(self.y_test,self.y_test_pred)}')
            print(f'test perfomance accuracy(Ridge) : {self.rid_reg.score(self.X_test,self.y_test)}')
            print(f'test perfomance accuracy(Lasso) : {self.las_reg.score(self.X_test,self.y_test)}')
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> {error_msg}')


if __name__ == "__main__":
    try:
        Tnobj = PREDICT('C:\\Users\\Lenovo\\Downloads\\House_price_project\\House_price_pred\\House Price Prediction Dataset.csv')
        Tnobj.mul_reg()
        Tnobj.rid_reg()
        Tnobj.las_reg()
        Tnobj.train_perfom()
        Tnobj.test_data()


    except Exception as e:
        error_type, error_msg, err_line = sys.exc_info()
        print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> {error_msg}')