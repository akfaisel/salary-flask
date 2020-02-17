import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

flask_app_obj = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@flask_app_obj.route('/')
def home():
    return render_template('form.html')

@flask_app_obj.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('form.html', prediction_text='Employee Salary should be Rs. {}'.format(output))


if __name__ == "__main__":
    flask_app_obj.run(debug=True)