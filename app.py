
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
# create instance of Flask app
app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))
#define route 1
@app.route('/')
#content of 1 route
def home():
    return render_template('index.html')
#  define route 2
@app.route('/predict',methods=['POST'])
# content of it
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Odessa real estate price prediction ${}'.format(output))
# 3rd route and it's content
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
#Running and Controlling the script
if __name__ == "__main__":
    app.run(debug=True)