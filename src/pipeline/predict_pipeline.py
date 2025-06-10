import os

import sys
sys.path.insert(0, '/Belajar/UDEMY/Krish_Naik/Complete_Machine_Learning_NLP_Bootcamp/Section48_MLProject')
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import  load_object
import dill

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds = model.predict(data_scaled)
        except Exception as e:
            raise CustomException(e,sys)
        return preds
    
class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e :
            raise CustomException(e,sys)
            pass
def load_object(file_path):
        try:
            with open(file_path,"rb") as file_obj:
                return dill.load(file_obj)
        except Exception as e:
            raise CustomException(e,sys)