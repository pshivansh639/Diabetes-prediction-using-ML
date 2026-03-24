import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# ✅ Safe loading of model and scaler
base_dir = os.path.dirname(__file__)

scaler_path = os.path.join(base_dir, "diabetes_scaler.pkl")
model_path = os.path.join(base_dir, "diabetes_model.pkl")

loaded_scaler = pickle.load(open(scaler_path, "rb"))
model = pickle.load(open(model_path, "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    
    pregnancies = int(request.form['pregnancy'])
    glucose = int(request.form['glucose'])
    blood_pressure = int(request.form['bloodpressure'])
    skin_thickness = int(request.form['skinthickness'])
    insulin = int(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree = float(request.form['diabetesPedigreeFunction'])
    age = int(request.form['age'])

    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age)
    
    nparray = np.asarray(input_data)
    reshaped = nparray.reshape(1, -1)
    stddata = loaded_scaler.transform(reshaped)

    pred = model.predict(stddata)

    if pred[0] == 1:
        prediction = "DIABETIC"
    else:
        prediction = "NON-DIABETIC"

    return render_template("result.html", prediction_text=f"THE PERSON MAY BE {prediction}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)