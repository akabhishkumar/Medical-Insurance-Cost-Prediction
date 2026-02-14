import pickle
import numpy as np
import os
from flask import Flask, request, render_template, jsonify



try:
    with open('svr_model.pkl', 'rb') as f:
       
        svr = pickle.load(f)

    with open('X_scaler.pkl', 'rb') as f:
        X_scaler = pickle.load(f)

    with open('y_scaler.pkl', 'rb') as f:
        y_scaler = pickle.load(f)
        
except FileNotFoundError:
    print("Error: Model or Scaler files not found. Ensure svr_model.pkl, X_scaler.pkl, and y_scaler.pkl are in the same directory.")
    svr = None
    X_scaler = None
    y_scaler = None
except Exception as e:
    print(f"Error loading model assets: {e}")
    svr = None


app = Flask(__name__)

@app.route('/')
def home():
    
    return render_template('home.html') 

@app.route('/predict', methods=['POST'])
def predict():
    if svr is None:
        return render_template('op.html', pred="Error: Model assets failed to load.")

    try:
       
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['bmi']),
            float(request.form['children']),
            float(request.form['smoker']),
            float(request.form['region'])
        ]
        
        features_array = np.array([features])

        scaled_features = X_scaler.transform(features_array)

        pred_scaled = svr.predict(scaled_features)

        pred_real = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        return render_template('op.html', pred=f"Rs. {round(pred_real,2):,}")
        
    except KeyError as e:
        return render_template('op.html', pred=f"Error! Missing form field: {e}. Check your HTML input names.")
    except Exception as e:
        return render_template('op.html', pred=f"Prediction Error: {e}")

if __name__ == "__main__":
    
    app.run(debug=True)