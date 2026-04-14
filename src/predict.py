import pandas as pd

def predict_machine(model, scaler, input_data, columns):
    input_df = pd.DataFrame([input_data], columns=columns)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    return prediction