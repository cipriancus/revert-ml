import pickle
from flask import Flask
from flask import request
import os

port = os.getenv('VCAP_APP_PORT', '5000')

app = Flask(__name__)


@app.route("/", methods=['POST'])
def hello():
    with open('model.pkl', 'rb') as f:
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            json = request.json
            clf2 = pickle.load(f)
            prediction_list = []
            for iterator in json.keys():
                prediction_list.append(json[iterator])
            prediction_list.pop()
            prediction = clf2.predict([prediction_list])
            return str(bool(prediction[0]))
        else:
            return 'Content-Type not supported!'


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(port), debug=True)