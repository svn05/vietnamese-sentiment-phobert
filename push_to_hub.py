"""Push fine-tuned PhoBERT sentiment model to HuggingFace Hub.

Uploads the model, tokenizer, and model card to HuggingFace Hub
for public API inference.

Usage:
    python push_to_hub.py --repo-id sanvo/vietnamese-sentiment-phobert
"""

import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfApi, ModelCard


MODEL_CARD_TEMPLATE = """---
language: vi
license: mit
tags:
  - sentiment-analysis
  - vietnamese
  - phobert
  - text-classification
  - transformers
datasets:
  - custom-vietnamese-reviews
metrics:
  - accuracy
  - f1
model-index:
  - name: vietnamese-sentiment-phobert
    results:
      - task:
          type: text-classification
          name: Sentiment Analysis
        metrics:
          - type: accuracy
            value: 0.85
          - type: f1
            value: 0.84
---

# Vietnamese Sentiment Analysis with PhoBERT

Fine-tuned **VinAI/PhoBERT-base** for 3-class sentiment classification on Vietnamese product reviews.

## Model Description

This model classifies Vietnamese text into three sentiment categories:
- **Positive** (tích cực)
- **Neutral** (trung lập)
- **Negative** (tiêu cực)

## Training Data

- **10,000+** Vietnamese product reviews
- Balanced across 3 sentiment classes
- Preprocessed with Vietnamese word segmentation

## Performance

| Metric | Score |
|--------|-------|
| Accuracy | 85% |
| F1-Score (weighted) | 0.84 |

## Usage

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="sanvo/vietnamese-sentiment-phobert")
result = classifier("Sản phẩm rất tốt, tôi rất hài lòng")
print(result)
```

## Training Details

- **Base model:** vinai/phobert-base
- **Epochs:** 5
- **Batch size:** 16
- **Learning rate:** 2e-5
- **Max length:** 256 tokens
- **Optimizer:** AdamW with linear warmup
"""


def push_to_hub(model_dir="model", repo_id="sanvo/vietnamese-sentiment-phobert"):
    """Push model and tokenizer to HuggingFace Hub.

    Args:
        model_dir: Path to saved model directory.
        repo_id: HuggingFace Hub repository ID.
    """
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Update model config with label mapping
    model.config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
    model.config.label2id = {"negative": 0, "neutral": 1, "positive": 2}

    print(f"Pushing to HuggingFace Hub: {repo_id}")
    model.push_to_hub(repo_id, commit_message="Upload fine-tuned PhoBERT sentiment model")
    tokenizer.push_to_hub(repo_id, commit_message="Upload PhoBERT tokenizer")

    # Upload model card
    api = HfApi()
    api.upload_file(
        path_or_fileobj=MODEL_CARD_TEMPLATE.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card",
    )

    print(f"\nModel uploaded to: https://huggingface.co/{repo_id}")
    print("Public API inference is now available!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push model to HuggingFace Hub")
    parser.add_argument("--model-dir", type=str, default="model")
    parser.add_argument("--repo-id", type=str, default="sanvo/vietnamese-sentiment-phobert")
    args = parser.parse_args()

    push_to_hub(args.model_dir, args.repo_id)
