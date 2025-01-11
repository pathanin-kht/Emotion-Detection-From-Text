import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle

# Load dataset
data = pd.read_csv('tweet_emotions.csv')

# Preview dataset
print("Dataset Preview:")
print(data.head())

# Drop missing values
data.dropna(inplace=True)

# Map emotions to broader categories
emotion_map = {
    'empty': 'negative',
    'sadness': 'negative',
    'enthusiasm': 'positive',
    'neutral': 'neutral',
    'worry': 'negative',
    'surprise': 'neutral',
    'love': 'positive',
    'fun': 'positive',
    'happiness': 'positive',
    'hate': 'negative',
    'relief': 'positive',
    'boredom': 'negative',
    'anger': 'negative'
}

data['sentiment'] = data['sentiment'].map(emotion_map)

# Encode target labels
label_encoder = LabelEncoder()
data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

# Separate features and target
X = data['content']
y = data['sentiment']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Improved TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,  # Increase features
    stop_words='english',
    ngram_range=(1, 2),  # Include bigrams
    min_df=5,           # Ignore very rare words
    max_df=0.8          # Ignore very common words
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Hyperparameter tuning with Logistic Regression
param_grid = {
    'C': [0.1, 1, 10],  # Regularization strength
    'max_iter': [500, 1000]  # Number of iterations
}
grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_tfidf, y_train)

# Best model from grid search
best_model = grid.best_estimator_
print("\nBest Parameters:", grid.best_params_)

# Evaluate the model
y_pred = best_model.predict(X_test_tfidf)
print("\nImproved Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the best model, label encoder, and vectorizer
with open('best_emotion_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)
with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nImproved Model and Vectorizer have been saved successfully!")
