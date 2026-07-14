from flask import Flask, request, jsonify
import onnxruntime as rt
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

session = rt.InferenceSession("Models/model.onnx")

EXPECTED_COLUMNS = [
    'Hour', 'Temperature_C', 'Humidity_pct', 'Wind_Speed_ms', 'Visibility_m', 
    'Solar_Radiation_MJm2', 'Rainfall_mm', 'Month', 'Day', 'Is_Peak_Hour', 
    'Is_Weekend', 'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos', 
    'Season_Spring', 'Season_Summer', 'Season_Winter', 'Holiday_Yes', 
    'Library_Branch_Al Rawdah Branch', 'Library_Branch_Corniche Kiosk', 
    'Library_Branch_Downtown Central', 'Library_Branch_University Branch', 
    'Top_Category_Business', 'Top_Category_Children', 'Top_Category_Fiction', 
    'Top_Category_History', 'Top_Category_Non-Fiction', 'Top_Category_Science', 
    'Top_Category_Technology', 'Membership_Type_Regular', 'Membership_Type_Student', 
    'Membership_Type_Walk-In', 'Day_of_Week_Monday', 'Day_of_Week_Saturday', 
    'Day_of_Week_Sunday', 'Day_of_Week_Thursday', 'Day_of_Week_Tuesday', 
    'Day_of_Week_Wednesday', 'Temperature_Bin_Warm', 'Temperature_Bin_Hot', 
    'Time_of_Day_Evening', 'Time_of_Day_Morning', 'Temp_Humidity_Interaction'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        
        df['Temp_Humidity_Interaction'] = df['Temperature_C'] * df['Humidity_pct']
        
        df = df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
        
        input_data = df.values.astype(np.float32)
        
        input_name = session.get_inputs()[0].name
        prediction = session.run(None, {input_name: input_data})[0]
        
        return jsonify({"status": "success", "predicted_rentals": float(prediction[0][0])}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)