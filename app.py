from flask import render_template, request, Flask
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


filename = "model/finalized_model.sav"
loaded_model = pickle.load(open(filename, "rb"))

@app.route("/", methods = ['GET','POST'])
def home():
    if request.method == "GET":
        return render_template("home.html")
    if request.method == "POST":

        # data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
        data = []
        for i in request.form:
            if i != "oldpeak":
                data.append(int(request.form[i]))
            else:
                data.append(float(request.form[i]))

        data_array = np.asarray(data)
        reshaped_data = data_array.reshape(1,-1)
        result1 = loaded_model.predict(reshaped_data)
        return render_template("index.html",value = result1)
        

if __name__ == "__main__":
    app.run(debug = True)