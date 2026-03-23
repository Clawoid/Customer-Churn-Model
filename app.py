from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    f = request.form

    features = [
        int(f["SeniorCitizen"]),
        int(f["tenure"]),
        float(f["MonthlyCharges"]),
        float(f["TotalCharges"]),
        1 if f["gender"] == "Male" else 0,
        1 if f["Partner"] == "Yes" else 0,
        1 if f["Dependents"] == "Yes" else 0,
        1 if f["PhoneService"] == "Yes" else 0,
        1 if f["MultipleLines"] == "No phone service" else 0,
        1 if f["MultipleLines"] == "Yes" else 0,
        1 if f["InternetService"] == "Fiber optic" else 0,
        1 if f["InternetService"] == "No" else 0,
        1 if f["OnlineSecurity"] == "No internet service" else 0,
        1 if f["OnlineSecurity"] == "Yes" else 0,
        1 if f["OnlineBackup"] == "No internet service" else 0,
        1 if f["OnlineBackup"] == "Yes" else 0,
        1 if f["DeviceProtection"] == "No internet service" else 0,
        1 if f["DeviceProtection"] == "Yes" else 0,
        1 if f["TechSupport"] == "No internet service" else 0,
        1 if f["TechSupport"] == "Yes" else 0,
        1 if f["StreamingTV"] == "No internet service" else 0,
        1 if f["StreamingTV"] == "Yes" else 0,
        1 if f["StreamingMovies"] == "No internet service" else 0,
        1 if f["StreamingMovies"] == "Yes" else 0,
        1 if f["Contract"] == "One year" else 0,
        1 if f["Contract"] == "Two year" else 0,
        1 if f["PaperlessBilling"] == "Yes" else 0,
        1 if f["PaymentMethod"] == "Credit card (automatic)" else 0,
        1 if f["PaymentMethod"] == "Electronic check" else 0,
        1 if f["PaymentMethod"] == "Mailed check" else 0,
    ]

    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)[0]

    result = "This customer is likely to CHURN." if prediction == 1 else "This customer is NOT likely to churn."
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)