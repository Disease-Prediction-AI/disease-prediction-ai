import pandas as pd
import joblib
import sys
import json

class LungCancerModel:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def preprocess_data(self, user_input):
        # Convert user input to a DataFrame
        user_data = pd.DataFrame({
            'GENDER': [user_input[0]],
            'AGE': [user_input[1]],
            'SMOKING': [user_input[2]],
            'YELLOW_FINGERS': [user_input[3]],
            'ANXIETY': [user_input[4]],
            'PEER_PRESSURE': [user_input[5]],
            'CHRONICDISEASE': [user_input[6]],
            'FATIGUE': [user_input[7]],
            'ALLERGY': [user_input[8]],
            'WHEEZING': [user_input[9]],
            'ALCOHOLCONSUMING': [user_input[10]],
            'COUGHING': [user_input[11]],
            'SHORTNESSOFBREATH': [user_input[12]],
            'SWALLOWINGDIFFICULTY': [user_input[13]],
            'CHESTPAIN': [user_input[14]]
        })
        # Map string values to numeric
        user_data['GENDER'].replace({'M': "0", 'F': "1"}, inplace=True)
        user_data.replace({'NO': 1, 'YES': 2}, inplace=True)

        # Strip leading and trailing whitespaces from column names
        user_data.columns = user_data.columns.str.strip()

        return user_data
    
    def predict_cancer(self, user_data):
        # Perform prediction
        cancer_prediction = self.model.predict(user_data)
        return cancer_prediction


if __name__ == "__main__":

    user_input = sys.argv[1:]

    lung_cancer_model = LungCancerModel()
    
    lung_cancer_model.load_model('model/model_lung_cancer.sav')

    # Preprocess data
    preprocessed_data = lung_cancer_model.preprocess_data(user_input)

    cancer_prediction = lung_cancer_model.predict_cancer(preprocessed_data)

    # print('prediction:', cancer_prediction)

    response_dict = {
        "cancer_prediction": cancer_prediction[0]
    }
    # Convert the dictionary to a JSON-formatted string
    json_response = json.dumps(response_dict)

    # Print the JSON response
    print(json_response)