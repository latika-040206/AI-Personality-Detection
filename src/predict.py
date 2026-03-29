import joblib

model = joblib.load("../models/personality_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

text = input("Enter a sentence: ")

text_vector = vectorizer.transform([text])

prediction = model.predict(text_vector)

print("Predicted Personality:", prediction[0])
