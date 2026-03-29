import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load data
data = pd.read_csv("../dataset/personality_dataset.csv")
data = pd.read_csv("../dataset/mbti_1.csv")
X = data["text"]
y = data["personality"]

# Vectorize
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

# Save results
os.makedirs("../results", exist_ok=True)
with open("../results/accuracy.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# Save model
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/model.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")

print(f" Accuracy: {accuracy * 100:.2f}%")
print(" Model saved!")
print(" Results saved to results/accuracy.txt")