import joblib
import os

model = joblib.load("../models/model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

MBTI_DESCRIPTIONS = {
    "INTJ": "The Architect – Strategic, independent thinker",
    "INTP": "The Thinker – Logical, analytical problem solver",
    "ENTJ": "The Commander – Bold, confident leader",
    "ENTP": "The Debater – Clever, curious innovator",
    "INFJ": "The Advocate – Insightful, principled idealist",
    "INFP": "The Mediator – Empathetic, creative dreamer",
    "ENFJ": "The Protagonist – Charismatic, inspiring leader",
    "ENFP": "The Campaigner – Enthusiastic, creative free spirit",
    "ISTJ": "The Logistician – Reliable, detail-oriented planner",
    "ISFJ": "The Defender – Caring, loyal protector",
    "ESTJ": "The Executive – Organized, rule-following manager",
    "ESFJ": "The Consul – Warm, social caretaker",
    "ISTP": "The Virtuoso – Practical, hands-on problem solver",
    "ISFP": "The Adventurer – Gentle, artistic free spirit",
    "ESTP": "The Entrepreneur – Bold, action-oriented doer",
    "ESFP": "The Entertainer – Spontaneous, fun-loving performer",
}

os.makedirs("../results", exist_ok=True)

print("🧠 === AI Personality Detection (16 MBTI Types) ===")
print("Type 'quit' to exit\n")

results = []

while True:
    user_input = input("Describe yourself or enter a sentence: ")

    if user_input.lower() == "quit":
        break

    if user_input.strip() == "":
        print("Please enter some text.\n")
        continue

    text_vec = vectorizer.transform([user_input])
    personality = model.predict(text_vec)[0]
    description = MBTI_DESCRIPTIONS.get(personality, "Unknown type")

    print(f"\n🎯 Predicted Type : {personality}")
    print(f"📖 Description   : {description}\n")

    results.append(f"Input: {user_input}\nType: {personality} - {description}\n")

if results:
    with open("../results/predictions.txt", "w") as f:
        f.write("\n".join(results))
    print("✅ Predictions saved to results/predictions.txt")