# importing necessary libraries and functions
import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)  # Initialize the flask App

@app.route('/')  # Homepage
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model=open('/home/yifan_li/StockApp/stock_model.pkl','rb')

    # retrieving values from form
    if request.method=='POST': 
        user_features=request.form.get('prediction')
        user_input=[np.array(user_features)]
        prediction=model.predict([[user_input]])
        print(prediction)
    return render_template ('output.html')


if __name__ == "__main__":
    app.run(debug=True)
