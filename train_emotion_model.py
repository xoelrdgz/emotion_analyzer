"""Emotion Detection Model Training Script.

This script handles the fine-tuning of RoBERTa model for emotion detection
using the go_emotions dataset. It implements a complete training pipeline 
including data preprocessing, model configuration, training loop, and model evaluation.

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
    Base: roberta-base
    Task: Multi-label emotion classification
    Output Classes: 28 emotions from go_emotions dataset
    
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
            'labels': torch.tensor(label, dtype=torch.float)
        }

def load_data():
    """Load and preprocess the emotion dataset."""
    logger.info("Loading go_emotions dataset...")
    dataset = load_dataset("go_emotions")
    
    return dataset['train'], dataset['validation']

def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
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
    
    return total_loss / len(train_loader)

def evaluate(model, eval_loader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            preds = (outputs.logits > 0).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(eval_loader), all_preds, all_labels

def save_model(model, tokenizer, output_dir="./emotion_model"):
    """Save the trained model and tokenizer"""
    logger.info(f"Saving model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
    model = AutoModelForSequenceClassification.from_pretrained(
        "SamLowe/roberta-base-go_emotions",
        problem_type="multi_label_classification"
    )
    model.to(device)
    
    # Load datasets
    train_data, val_data = load_data()
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataset = EmotionDataset(
        train_data['text'],
        train_data['labels'],
        tokenizer
    )
    val_dataset = EmotionDataset(
        val_data['text'],
        val_data['labels'],
        tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size
    )
    
    # Initialize optimizer and scheduler
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
