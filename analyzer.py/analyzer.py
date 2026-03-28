
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("dataset.csv")

# Text and labels
X = data["text"]
y = data["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vectorized, y)

print("Model trained successfully!")
print("Type a caption to predict personality.\n")

# User input loop
while True:
    user_input = input("Enter caption: ")
    
    if user_input.lower() == "exit":
        print("Program ended.")
        break

    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)

    print("Predicted Personality:", prediction[0])
