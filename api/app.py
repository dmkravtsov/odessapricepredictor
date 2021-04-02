
import numpy as np
import math
from flask import Flask, request, jsonify, render_template
import pickle
from loguru import logger  # Дебажить на стороне сервара.

# 1  Проверить типы данных. ПОльзователи что попало отправляют
# 2 Try Eccept

# create instance of Flask app
app = Flask(__name__)
# model = pickle.load(open('api/models/finalized_model.pkl', 'rb'))
model = pickle.load(open('models/finalized_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    try:
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = (np.expm1(model.predict(final_features))).round()

        output = round(prediction[0], 2)
    except Exception(e):
        logger.error(e)

    return render_template('index.html', prediction_text='Predicted price for this apartments ${}'.format(output))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


# Рекомендация
@app.route('/update_model', methods=['POST'])
def update_model():
    '''
    Обновление модели.
    1. СОбрать данные
    2. Обучить модель
    3. Сохранить
    '''
    pass


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
