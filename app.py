from flask import Flask, request, render_template
from sklearn.externals import joblib
from numpy import array

app = Flask(__name__)
model = joblib.load('heart.pkl')

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        sex = int(request.form["sex"])
        trestbps = int(request.form["trestbps"])
        chol = int(request.form["chol"])
        oldpeak = float(request.form["oldpeak"])
        thalach = int(request.form["thalach"])
        fbs = int(request.form["fbs"])
        exang = int(request.form["exang"])
        slope = int(request.form["slope"])
        cp = int(request.form["cp"])
        thal = int(request.form["thal"])
        ca = int(request.form["ca"])
        restecg = int(request.form["restecg"])

        arr = array([[age, sex, cp, trestbps,
                      chol, fbs, restecg, thalach,
                      exang, oldpeak, slope, ca,
                      thal]])

        pred = model.predict(arr)
        prediction_labels = {0: 'NO HEART PROBLEM', 1: 'HEART PROBLEM'}
        res_val = prediction_labels.get(pred.item(), 'UNKNOWN')

        return render_template('index.html', prediction_text='PATIENT HAS {}'.format(res_val))

    except ValueError:
        return render_template('index.html', prediction_text='Invalid input. Please enter valid values.')


if __name__ == '__main__':
    app.run(debug=True)
