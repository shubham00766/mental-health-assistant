import flask
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS # Line 1

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize the Flask app
app = Flask(__name__)
CORS(app) # Line 2: This allows requests from any origin

# Create the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text_to_predict = data['text']

        # Vectorize the text
        text_vec = vectorizer.transform([text_to_predict])

        # Make prediction
        prediction = model.predict(text_vec)
        sentiment = prediction[0]

        return jsonify({'sentiment': sentiment})

    except Exception as e:
        return jsonify({'error': str(e)})

# Main entry point for the application
if __name__ == '__main__':
    app.run(port=5000, debug=True)