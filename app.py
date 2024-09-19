from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Load dataset and train model
iris = load_iris()
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

# Save model
joblib.dump(model, "model.joblib")

@app.route('/')
def home():
    return "ML Model API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
