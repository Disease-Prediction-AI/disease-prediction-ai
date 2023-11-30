import pandas as pd
import matplotlib.pyplot as plt
import joblib


class LungCancerModel:
    def __init__(self):
        self.model = None
        

if __name__ == "__main__":
    user_data = pd.DataFrame({
            'GENDER': ['0'],
            'AGE': [74],
            'SMOKING': ['YES'],
            'YELLOW_FINGERS': ['NO'],
            'ANXIETY': ['NO'],
            'PEER_PRESSURE': ['NO'],
            'CHRONICDISEASE': ['YES'],
            'FATIGUE': ['YES'],
            'ALLERGY': ['YES'],
            'WHEEZING': ['NO'],
            'ALCOHOLCONSUMING': ['NO'],
            'COUGHING': ['NO'],
            'SHORTNESSOFBREATH': ['YES'],
            'SWALLOWINGDIFFICULTY': ['YES'],
            'CHESTPAIN': ['YES']
        })
    
    lung_cancer_model = joblib.load('model/model_lung_cancer.sav')

    #Map string values to numeric
    user_data.replace({'NO': 1, 'YES': 2}, inplace=True)

    # Strip leading and trailing whitespaces from column names
    user_data.columns = user_data.columns.str.strip()

    # Perform prediction
    cancer_prediction = lung_cancer_model.predict(user_data)

    print('prediction = ',cancer_prediction )



