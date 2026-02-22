# Vietnamese Sentiment Analysis with PhoBERT

Fine-tuned **VinAI/PhoBERT-base** for 3-class sentiment classification on **10,000+ Vietnamese product reviews**, achieving **85% accuracy**. Model deployed to **HuggingFace Hub** for public API inference.

## Overview

Classifies Vietnamese text into three sentiment categories:
- **Positive** (tích cực) — satisfied, good quality, recommend
- **Neutral** (trung lập) — average, okay, nothing special
- **Negative** (tiêu cực) — dissatisfied, poor quality, complaints

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 85.0% |
| F1-Score (weighted) | 0.84 |
| Precision (macro) | 0.85 |
| Recall (macro) | 0.84 |

## Quick Start

### Install
```bash
git clone https://github.com/svn05/vietnamese-sentiment-phobert.git
cd vietnamese-sentiment-phobert
pip install -r requirements.txt
```

### Use pre-trained model from HuggingFace Hub
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="svn05/vietnamese-sentiment-phobert")
result = classifier("Sản phẩm rất tốt, tôi rất hài lòng")
print(result)  # [{'label': 'positive', 'score': 0.96}]
```

### Train from scratch
```bash
# 1. Prepare data
python data/prepare_data.py --n-samples 10000

# 2. Fine-tune PhoBERT
python train.py --epochs 5 --batch-size 16 --lr 2e-5

# 3. Evaluate
python evaluate.py

# 4. Push to HuggingFace Hub
python push_to_hub.py --repo-id your-username/your-model-name
```

### Run Gradio demo
```bash
python app.py
```

### Predict single text
```bash
python predict.py --text "Sản phẩm rất tốt, tôi rất hài lòng"
python predict.py --file reviews.txt
python predict.py  # Interactive mode
```

## Training Details

- **Base model:** [vinai/phobert-base](https://huggingface.co/vinai/phobert-base)
- **Dataset:** 10,000+ Vietnamese product reviews (3 classes, balanced)
- **Epochs:** 5 with linear LR warmup
- **Batch size:** 16
- **Learning rate:** 2e-5 (AdamW)
- **Max sequence length:** 256 tokens
- **Gradient clipping:** max norm 1.0

## Project Structure

```
vietnamese-sentiment-phobert/
├── train.py            # Fine-tune PhoBERT
├── evaluate.py         # Classification report, confusion matrix
├── predict.py          # Single text / batch / interactive inference
├── app.py              # Gradio web demo
├── push_to_hub.py      # Upload to HuggingFace Hub
├── data/
│   └── prepare_data.py # Data preparation and preprocessing
├── model/              # Saved model (generated after training)
├── requirements.txt
└── README.md
```

## Tech Stack

- **PhoBERT** (VinAI) — Vietnamese pre-trained language model
- **HuggingFace Transformers** — Model fine-tuning and inference
- **PyTorch** — Deep learning framework
- **Gradio** — Interactive web demo
- **scikit-learn** — Evaluation metrics

## HuggingFace Hub

Model available at: [svn05/vietnamese-sentiment-phobert](https://huggingface.co/svn05/vietnamese-sentiment-phobert)
