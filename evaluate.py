"""Evaluation script for the fine-tuned PhoBERT sentiment model.

Generates classification report, confusion matrix, and per-class metrics.

Usage:
    python evaluate.py
    python evaluate.py --model-dir model --data-dir data
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

from train import VietnameseReviewDataset, LABEL_NAMES


def evaluate_model(model_dir="model", data_dir="data", batch_size=32, max_length=256):
    """Run full evaluation on test set.

    Args:
        model_dir: Path to saved model directory.
        data_dir: Path to data directory.
        batch_size: Evaluation batch size.
        max_length: Max token length.

    Returns:
        Dictionary with all metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    # Load test data
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_dataset = VietnameseReviewDataset(
        test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_length
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Predict
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    print("=" * 60)
    print("VIETNAMESE SENTIMENT ANALYSIS — EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"\n{report}")
    print("Confusion Matrix:")
    print(f"{'':>12} {'Pred Neg':>10} {'Pred Neu':>10} {'Pred Pos':>10}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name:>12} {cm[i, 0]:>10d} {cm[i, 1]:>10d} {cm[i, 2]:>10d}")

    return {
        "accuracy": accuracy,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": np.array(all_probs),
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PhoBERT sentiment model")
    parser.add_argument("--model-dir", type=str, default="model")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    evaluate_model(args.model_dir, args.data_dir, args.batch_size)
