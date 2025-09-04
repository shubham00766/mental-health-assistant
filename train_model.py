import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Step 1: Data Load Karna
print("Loading data...")
try:
    df = pd.read_csv('Suicide_Detection.csv')
except FileNotFoundError:
    print("Error: 'Suicide_Detection.csv' not found. Please check the file path.")
    exit()

# Step 2: Data Clean Karna
print("Cleaning data...")
df['text'] = df['text'].apply(clean_text)

# Step 3: Data ko Train aur Test sets mein baantna
X = df['text']
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Text ko numbers mein badalna (Vectorization)
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Model ko train karna
print("Training the model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Step 6: Model ki performance check karna
print("Evaluating the model...")
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 7: Model aur Vectorizer ko save karna
print("Saving the model and vectorizer...")
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved successfully!")
# Custom data for training
new_data = {
    'text': [
        "marne ja rha hu",
        "I want to end my life",
        "I can't go on anymore",
        "this is the end for me",
        "I have nothing to live for",
        "I am going to kill myself",
        "suicide is the only option",
        "I feel like dying"
    ],
    'class': [
        'suicide',
        'suicide',
        'suicide',
        'suicide',
        'suicide',
        'suicide',
        'suicide',
        'suicide'
    ]
}

new_df = pd.DataFrame(new_data)

# Existing data ke saath naya data jodo
df = pd.concat([df, new_df], ignore_index=True)