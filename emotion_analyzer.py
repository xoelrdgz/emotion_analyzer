import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from colorama import Fore, Style, init

init(autoreset=True)

def load_model(model_name, model_dir):
    if not os.path.exists(model_dir):
        print(f"{model_name} model not found locally. Downloading from Hugging Face...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"{model_name} model downloaded and saved.")
    else:
        print(f"{model_name} model found locally. Loading...")
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

print("Loading models...")
sentiment_model_dir = "./sentiment_model"
emotion_model_dir = "./emotion_model"

try:
    sentiment_model, sentiment_tokenizer = load_model("nlptown/bert-base-multilingual-uncased-sentiment", sentiment_model_dir)
    emotion_model, emotion_tokenizer = load_model("bhadresh-savani/bert-base-uncased-emotion", emotion_model_dir)
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_model.to(device)
emotion_model.to(device)

sentiment_classifier = pipeline("text-classification", model=sentiment_model, tokenizer=sentiment_tokenizer, device=0 if torch.cuda.is_available() else -1)
emotion_classifier = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer, device=0 if torch.cuda.is_available() else -1, top_k=None)

emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

def analyze(text):
    print("\nSentiment Analysis:")
    sentiment_result = sentiment_classifier(text)[0]
    label = sentiment_result["label"]
    score = sentiment_result["score"] * 100

    color = Fore.GREEN if label.lower() == "positive" else Fore.RED if label.lower() == "negative" else Fore.YELLOW
    print(f"{color}Sentiment: {label.upper()} ({score:.2f}% confidence)")

    print("\nEmotion Analysis:")
    emotion_result = emotion_classifier(text)[0]
    sorted_emotions = sorted(emotion_result, key=lambda x: x["score"], reverse=True)[:5]
    for emo in sorted_emotions:
        percent = emo["score"] * 100
        print(f"{emo['label']:<15}: {percent:.2f}%")

def main():
    print("\nEnter a sentence to analyze sentiment and emotions (or 'q' to quit):")
    while True:
        text = input("> ")
        if text.strip().lower() in ["q", "quit", "exit"]:
            break
        analyze(text)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
