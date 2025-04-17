"""Emotion Detection Model Training Script.

This script handles the fine-tuning of a pre-trained BERT model for emotion detection
using the bhadresh-savani/bert-base-uncased-emotion dataset. It implements a complete
training pipeline including data preprocessing, model configuration, training loop,
and model evaluation.

The trained model will be saved in the ./emotion_model directory and can be used
by the Emotion Analyzer application for inference.

Requirements:
    - PyTorch
    - Transformers
    - Datasets
    - NumPy
    - scikit-learn
    - tqdm

Model Architecture:
    Base: bert-base-uncased
    Task: Multi-class emotion classification
    Output Classes: joy, sadness, anger, fear, love, surprise, etc.
    
Usage:
    python train_emotion_model.py [--epochs N] [--batch_size N] [--learning_rate N]
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
import argparse
import logging
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionDataset(Dataset):
    """Custom dataset for emotion classification task.
    
    This class handles the preprocessing of text data and conversion to
    tensor format required by PyTorch.
    
    Attributes:
        texts (list): List of input text samples
        labels (list): List of corresponding emotion labels
        tokenizer: BERT tokenizer for text preprocessing
        max_length (int): Maximum sequence length for padding/truncation
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize and prepare for BERT
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    """Load and preprocess the emotion dataset.
    
    Returns:
        tuple: Training and validation datasets
    """
    logger.info("Loading emotion dataset...")
    dataset = load_dataset("bhadresh-savani/bert-base-uncased-emotion")
    
    return dataset['train'], dataset['validation']

def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train the model for one epoch.
    
    Args:
        model: The BERT model instance
        train_loader: DataLoader for training data
        optimizer: Optimization algorithm
        scheduler: Learning rate scheduler
        device: Device to run training on (CPU/GPU)
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)

def evaluate(model, eval_loader, device):
    """Evaluate the model on validation data.
    
    Args:
        model: The BERT model instance
        eval_loader: DataLoader for validation data
        device: Device to run evaluation on (CPU/GPU)
        
    Returns:
        tuple: (validation loss, predicted labels, true labels)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return (
        total_loss / len(eval_loader),
        all_preds,
        all_labels
    )

def save_model(model, tokenizer, output_dir="./emotion_model"):
    """Save the trained model and tokenizer.
    
    Args:
        model: Trained BERT model
        tokenizer: Associated tokenizer
        output_dir: Directory to save model files
    """
    logger.info(f"Saving model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def main(args):
    """Main training function.
    
    Args:
        args: Command line arguments containing training parameters
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=6  # Number of emotion classes
    ).to(device)
    
    # Load datasets
    train_dataset, val_dataset = load_data()
    
    # Create data loaders
    train_loader = DataLoader(
        EmotionDataset(
            train_dataset['text'],
            train_dataset['label'],
            tokenizer
        ),
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        EmotionDataset(
            val_dataset['text'],
            val_dataset['label'],
            tokenizer
        ),
        batch_size=args.batch_size
    )
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, preds, labels = evaluate(model, val_loader, device)
        
        logger.info(f"Train loss: {train_loss:.4f}")
        logger.info(f"Validation loss: {val_loss:.4f}")
        logger.info("\nClassification Report:")
        print(classification_report(labels, preds))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, tokenizer)
            logger.info("New best model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train emotion detection model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    main(args)
