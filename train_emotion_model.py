from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Configuration
model_name = "distilbert-base-uncased"
batch_size = 8
epochs = 3
num_labels = 28  # 27 emotions + neutral

print("📥 Loading GoEmotions dataset...")
dataset = load_dataset("go_emotions", "simplified")

# We'll use only one label per input (the first one)
def simplify_labels(example):
    return {"label": example["labels"][0]}

dataset = dataset.map(simplify_labels)

# Tokenization
print("🔤 Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], truncation=True)

tokenized = dataset.map(tokenize, batched=True)

# Split into train/test
split = tokenized["train"].train_test_split(test_size=0.1)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Model
print("🧠 Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./emotion_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs_emotions",
)

print("🚀 Training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save model
print("💾 Saving emotion model to ./emotion_model")
trainer.save_model("./emotion_model")
tokenizer.save_pretrained("./emotion_model")
