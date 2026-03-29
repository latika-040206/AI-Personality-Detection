#  AI Personality Detection

A machine learning project that predicts **MBTI Personality Types** from text input using Natural Language Processing (NLP).

---

##  About the Project

This project explores whether personality traits can be identified from short text descriptions using machine learning techniques. The system classifies text into one of the **16 MBTI personality types** such as INTJ, ENFP, INFJ, and more.

Built as part of my BTech journey at **VIT Bhopal University**.

---

##  Project Structure

```
AI-Personality-Detection/
├── Dataset/
│   └── personality_dataset.csv   # Sample dataset
├── Models/
│   ├── model.pkl                 # Trained ML model
│   └── vectorizer.pkl            # TF-IDF Vectorizer
├── results/
│   ├── accuracy.txt              # Model accuracy report
│   └── predictions.txt           # Prediction results
├── src/
│   ├── train_model.py            # Script to train the model
│   └── predict.py                # Script to predict personality
└── README.md
```

---

##  Tech Stack

- **Language:** Python 3.13
- **Libraries:** Pandas, Scikit-learn, Joblib
- **Model:** Logistic Regression with TF-IDF Vectorization
- **IDE:** Visual Studio Code

---

##  Dataset

This project uses the **MBTI Personality Type Dataset** from Kaggle.

Downloaded it  from here: [https://www.kaggle.com/datasets/datasnaek/mbti-type](https://www.kaggle.com/datasets/datasnaek/mbti-type)


---

##  How to Run

### Step 1: Clone the repository
```bash
git clone https://github.com/latika-040206/AI-Personality-Detection.git
cd AI-Personality-Detection
```

### Step 2: Install dependencies
```bash
pip install pandas scikit-learn joblib
```

### Step 3: Download the dataset
Download `mbti_1.csv` from Kaggle and place it in the `Dataset/` folder.

### Step 4: Train the model
```bash
cd src
python train_model.py
```

### Step 5: Predict personality
```bash
python predict.py
```

---

##  MBTI Personality Types Supported

| Type | Description |
|------|-------------|
| INTJ | The Architect – Strategic, independent thinker |
| INTP | The Thinker – Logical, analytical problem solver |
| ENTJ | The Commander – Bold, confident leader |
| ENTP | The Debater – Clever, curious innovator |
| INFJ | The Advocate – Insightful, principled idealist |
| INFP | The Mediator – Empathetic, creative dreamer |
| ENFJ | The Protagonist – Charismatic, inspiring leader |
| ENFP | The Campaigner – Enthusiastic, creative free spirit |
| ISTJ | The Logistician – Reliable, detail-oriented planner |
| ISFJ | The Defender – Caring, loyal protector |
| ESTJ | The Executive – Organized, rule-following manager |
| ESFJ | The Consul – Warm, social caretaker |
| ISTP | The Virtuoso – Practical, hands-on problem solver |
| ISFP | The Adventurer – Gentle, artistic free spirit |
| ESTP | The Entrepreneur – Bold, action-oriented doer |
| ESFP | The Entertainer – Spontaneous, fun-loving performer |

---

##  Model Performance

- **Algorithm:** Logistic Regression
- **Vectorizer:** TF-IDF (5000 features)
- **Accuracy:** Check `results/accuracy.txt` after training

---

##  Example Usage

```
  === AI Personality Detection (16 MBTI Types) ===
Type 'quit' to exit

Describe yourself or enter a sentence: I love analyzing complex problems alone

 Predicted Type : INTP
 Description   : The Thinker – Logical, analytical problem solver
```

---

##  Author

**Latika Shekhawat**
- 📍 VIT Bhopal University
- 🎓 B.Tech Computer Science Engineering (1st Year)
- 🐙 GitHub: [@latika-040206](https://github.com/latika-040206)

---

##  License

This project is open source and available under the [MIT License](LICENSE).
