from flask import Flask, render_template, request
import base64
import json
from io import BytesIO
import cnn2 as cnn

app = Flask(__name__, template_folder='template')
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    b = base64.urlsafe_b64decode(request.form['operation'])
    operation = BytesIO(b)
    result = cnn.predict(operation)
    return json.dumps({
        'solution': result
    })

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True, host='0.0.0.0', port=5000)