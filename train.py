"""Fine-tune VinAI's PhoBERT for Vietnamese sentiment classification.

3-class sentiment: positive, neutral, negative on Vietnamese product reviews.

Usage:
    python train.py
    python train.py --epochs 5 --batch-size 16 --lr 2e-5
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


NUM_LABELS = 3
LABEL_NAMES = ["negative", "neutral", "positive"]


class VietnameseReviewDataset(Dataset):
    """PyTorch Dataset for Vietnamese product reviews."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
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
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_data(data_dir="data"):
    """Load train/val/test splits from CSV files."""
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Label distribution (train): {train_df['label'].value_counts().sort_index().to_dict()}")

    return train_df, val_df, test_df


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate model on validation/test data."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            total_loss += outputs.loss.item() * input_ids.size(0)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, accuracy, f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT for Vietnamese sentiment")
    parser.add_argument("--model-name", type=str, default="vinai/phobert-base")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Check if data exists, generate if not
    if not os.path.exists(os.path.join(args.data_dir, "train.csv")):
        print("Data not found. Generating synthetic reviews...")
        from data.prepare_data import generate_synthetic_reviews, save_splits
        df = generate_synthetic_reviews()
        save_splits(df, args.data_dir)

    # Load data
    train_df, val_df, test_df = load_data(args.data_dir)

    # Load tokenizer and model
    print(f"\nLoading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS,
        problem_type="single_label_classification",
    ).to(device)

    # Create datasets
    train_dataset = VietnameseReviewDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, args.max_length
    )
    val_dataset = VietnameseReviewDataset(
        val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, args.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    best_val_f1 = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>10} {'Val F1':>10}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        print(f"{epoch:>6d} {train_loss:>12.4f} {train_acc:>10.4f} {val_loss:>10.4f} {val_acc:>10.4f} {val_f1:>10.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  -> Saved best model (F1: {val_f1:.4f})")

    print(f"\nBest validation F1: {best_val_f1:.4f}")
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
