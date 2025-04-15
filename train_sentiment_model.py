from datasets import load_dataset, concatenate_datasets, Features, ClassLabel, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch

# Config
model_name = "distilbert-base-uncased"
num_labels = 3
batch_size = 8
epochs = 3

imdb = load_dataset("imdb", split="train").shuffle(seed=42).select(range(10000))
amazon = load_dataset("amazon_polarity", split="train").shuffle(seed=42).select(range(10000))
yelp = load_dataset("yelp_polarity", split="train").shuffle(seed=42).select(range(10000))

# Definir features comunes
common_features = Features({
    'text': Value('string'),
    'label': ClassLabel(names=['negative', 'positive'])
})

# Función para estandarizar el formato
def standardize_dataset(example, is_amazon=False):
    if is_amazon:
        text = example["content"]
    else:
        text = example["text"]
    return {
        "text": text,
        "label": 0 if example["label"] == 1 else 1  # Invertimos para que sea 0=negative, 1=positive
    }

# Estandarizar cada dataset
imdb = imdb.map(standardize_dataset, remove_columns=imdb.column_names).cast(common_features)
amazon = amazon.map(
    lambda x: standardize_dataset(x, is_amazon=True), 
    remove_columns=amazon.column_names
).cast(common_features)
yelp = yelp.map(standardize_dataset, remove_columns=yelp.column_names).cast(common_features)

# Concatenar los datasets
dataset = concatenate_datasets([imdb, amazon, yelp]).shuffle(seed=42)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], truncation=True)

tokenized = dataset.map(tokenize, batched=True)

# Split into train/test
split = tokenized.train_test_split(test_size=0.1)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./sentiment_model",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
)

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

# Save final model
trainer.save_model("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")
