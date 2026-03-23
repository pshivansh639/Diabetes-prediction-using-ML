import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

#Creating FLASK app

app=Flask(__name__)

#Load the pickle model
loaded_scaler = pickle.load(open('diabetes_scaler.pkl', 'rb'))
model=pickle.load(open("Diabetes_model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/predict",methods=["POST"])
def predict():
    
    pregnancies = int(request.form['pregnancy'])
    glucose = int(request.form['glucose'])
    blood_pressure = int(request.form['bloodpressure'])
    skin_thickness = int(request.form['skinthickness'])
    insulin = int(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree = float(request.form['diabetesPedigreeFunction'])
    age = int(request.form['age'])

    input_data=(pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree,age)
    nparray=np.asarray(input_data)
    reshaped=nparray.reshape(1,-1)
    stddata=loaded_scaler.transform(reshaped)
    pred=model.predict(stddata)
    if(pred==1):
        prediction="DIABETIC"
        print(prediction)
    else:
        prediction="NON-DIABETIC"
        print(prediction)

    return render_template("result.html", prediction_text=f"THE PERSON MAY BE {prediction}")
    
if __name__=="__main__":
    app.run(debug=True)