from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

# Load the saved model, vectorizer, and label encoder
model_path = 'best_emotion_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'
label_encoder_path = 'label_encoder.pkl'

try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with open(label_encoder_path, 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)
except FileNotFoundError as e:
    raise FileNotFoundError("Required files (model, vectorizer, or label encoder) not found: {}".format(e))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided or text is empty.'}), 400

        # Preprocess and predict
        text_tfidf = vectorizer.transform([text])
        predicted_label = model.predict(text_tfidf)
        emotion = label_encoder.inverse_transform(predicted_label)[0]  # Decode the predicted label

        return jsonify({'emotion': emotion})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
