###Flask
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts\\model.pkl'
            preprocessor_path='artifacts\\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

             # Ensure "Serial No." is dropped before preprocessing
           # features = features.drop(columns=['Serial No.'], errors='ignore') 

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def  __init__ (self,
        GRE_Score:int,
        TOEFL_Score:int,
        University_Rating:int, 
        SOP:float,
        LOR:float,
        CGPA:float,
        Research:bool):

        self.GRE_Score=GRE_Score

        self.TOEFL_Score=TOEFL_Score

        self.University_Rating=University_Rating

        self.SOP=SOP

        self.LOR=LOR

        self.CGPA=CGPA

        self.Research=Research

    #def to_dict(self):
     #   return {
      #      "GRE Score": self.GRE_Score,
       #     "TOEFL Score": self.TOEFL_Score,
        #    "University Rating": self.University_Rating,
         #   "SOP": self.SOP,
          #  "LOR": self.LOR,
           # "CGPA": self.CGPA,
            #"Research": bool(self.Research),  # Convert boolean to int if the model expects it
        #}

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "GRE Score": [self.GRE_Score],
                "TOEFL Score": [self.TOEFL_Score],
                "University Rating": [self.University_Rating],
                "SOP": [self.SOP],
                "LOR": [self.LOR],
                "CGPA": [self.CGPA],
                "Research": [self.Research],

            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)

