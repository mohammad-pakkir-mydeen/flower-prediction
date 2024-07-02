from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))


@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    f=[float(x) for x in request.form.values()]
    m=[np.array(f)]
    prediction=model.predict(m)
    print(prediction)
    return render_template("index.html",prediction_test="the flower is {}".format(prediction))
    
if __name__=="__main__":
    app.run(debug=True)
