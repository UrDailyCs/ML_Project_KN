import sys
sys.path.insert(0, '/Belajar/UDEMY/Krish_Naik/Complete_Machine_Learning_NLP_Bootcamp/Section48_MLProject')

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
# from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor


@dataclass
class modelTrainerConfig:
    trained_model_file_path=os.path.join('artifact','model.pkl')
class modelTrainer:
    def __init__(self):
        self.model_trainer_config = modelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                # "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            model_report:dict=evaluate_model(X_train= X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            logging.info(f"here is the model report {model_report}")

            # to get best model score
            best_model_score = max(sorted(model_report.values()))
            logging.info(f"here is the model report {best_model_score}")
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            logging.info(f"here is the model report {(best_model_score,best_model_name)}")
            best_model = models[best_model_name]
            logging.info(f"111Best model found on both training and testing. name is {best_model_name} with score {best_model_score}")
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and testing. name is {best_model_name} with score {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            logging.info(f"r2 score is {r2_square}")
            return r2_square
        except Exception as e:
            print(e)
            CustomException(e,sys)