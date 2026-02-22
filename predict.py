"""Inference script for Vietnamese sentiment prediction.

Predict sentiment of individual Vietnamese text inputs using the fine-tuned model.

Usage:
    python predict.py --text "Sản phẩm rất tốt, tôi rất hài lòng"
    python predict.py --file reviews.txt
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL_EMOJIS = {"negative": "👎", "neutral": "😐", "positive": "👍"}


def load_model(model_dir="model"):
    """Load fine-tuned PhoBERT model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer, device


def predict_sentiment(text, model, tokenizer, device, max_length=256):
    """Predict sentiment for a single text.

    Args:
        text: Vietnamese text input.
        model: Fine-tuned PhoBERT model.
        tokenizer: PhoBERT tokenizer.
        device: torch device.
        max_length: Maximum token length.

    Returns:
        dict with label, confidence, and all probabilities.
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    label = LABEL_NAMES[pred_idx]
    confidence = probs[pred_idx].item()

    return {
        "text": text,
        "label": label,
        "confidence": confidence,
        "probabilities": {
            name: probs[i].item() for i, name in enumerate(LABEL_NAMES)
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Vietnamese sentiment")
    parser.add_argument("--text", type=str, help="Single text to classify")
    parser.add_argument("--file", type=str, help="File with one text per line")
    parser.add_argument("--model-dir", type=str, default="model")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_dir)

    if args.text:
        result = predict_sentiment(args.text, model, tokenizer, device)
        emoji = LABEL_EMOJIS[result["label"]]
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['label']} {emoji} (confidence: {result['confidence']:.2%})")
        print(f"Probabilities:")
        for label, prob in result["probabilities"].items():
            bar = "█" * int(prob * 30)
            print(f"  {label:>10}: {prob:.4f} {bar}")

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"\nClassifying {len(texts)} texts...\n")
        for text in texts:
            result = predict_sentiment(text, model, tokenizer, device)
            emoji = LABEL_EMOJIS[result["label"]]
            print(f"{emoji} [{result['label']:>8}] ({result['confidence']:.2f}) {text[:80]}")

    else:
        # Interactive mode
        print("Vietnamese Sentiment Analyzer (PhoBERT)")
        print("Type a Vietnamese text and press Enter. Type 'quit' to exit.\n")
        while True:
            text = input(">>> ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            result = predict_sentiment(text, model, tokenizer, device)
            emoji = LABEL_EMOJIS[result["label"]]
            print(f"  {emoji} {result['label']} ({result['confidence']:.2%})")
            for label, prob in result["probabilities"].items():
                print(f"     {label}: {prob:.4f}")
            print()


if __name__ == "__main__":
    main()
