import pickle
import pandas as pd
def kyphosis_predict(age:int, number:int,start:int)->int:
    """This tool would be used to predict whether kyphosis is absent or present in a patient.
    It takes Age, Number and start values from user
    age: age of the child in MONTHS.
    number: number of vertebrae involved in the operation.
    start: number of the first vertebra operated on."""
    model = pickle.load(open("model.pkl", "rb"))
    preprocessing = pickle.load(open("preprocessing.pkl", "rb"))

    patient = pd.DataFrame({"Age": [15], "Number": [12], "Start": [7]})
    patient[['Age']] = preprocessing["age_scaler"].transform(patient[['Age']])
    prediction = model.predict(patient)
    
    return prediction