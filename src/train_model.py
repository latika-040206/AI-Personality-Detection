import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

data = pd.read_csv("../dataset/personality_dataset.csv")

X = data["text"]
y = data["personality"]

vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

joblib.dump(model, "../models/personality_model.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")

print("Model trained successfully")
print("Accuracy:", accuracy)

with open("../results/accuracy.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy}")
