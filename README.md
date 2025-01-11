# Emotion-Detection-From-Text

## Overview
This project implements an emotion prediction system using natural language processing (NLP) to predict emotions from text input. The model is based on a logistic regression classifier and uses a TF-IDF vectorizer for text feature extraction. The system processes English text input and classifies it into one of three main emotion categories: Positive, Negative, and Neutral. 

## Accuracy
The model achieved an accuracy of **57.075%** on the test dataset. The accuracy is moderate, as the model is still in the early stages of development and was trained with basic parameters. 

## Dataset
The dataset used for training is from [Emotion Detection from Text](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text/data).

## Example
Here is an example of how the system works:
1. The user enters a text such as "How cute you are"
2. The model predicts the emotion as **Positive**.
   
![Testcase](https://github.com/pathanin-kht/Emotion-Detection-From-Text/blob/c9c8f46de01477fd9b82cc3c2aa8ae882d665d6a/TestCase.png)

## Installation
1. Clone or download this repository.
2. Install the required dependencies:
   
   ```bash
   pip install -r requirements.txt
4. Download the dataset and place it in the same directory as the script, or update the dataset path in the code.
5. Run the model:
   
   ```bash
   python emotion_train_model.py
   ```
7. Start the server
   
   ```bash
   python app.py

9. Open your browser and navigate to http://127.0.0.1:5000.

## License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Contact
For feedback or inquiries, feel free to reach out via [pathanin.kht@gmail.com](pathanin.kht@gmail.com).


