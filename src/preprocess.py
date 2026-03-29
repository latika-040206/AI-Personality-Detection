import pandas as pd

def load_data():
    data = pd.read_csv("../dataset/personality_dataset.csv")
    X = data["text"]
    y = data["personality"]
    return X, y

if __name__ == "__main__":
    X, y = load_data()
    print("Dataset Loaded")
    print(X.head())
