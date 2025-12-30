# Vietnamese Sentiment Analysis with PhoBERT

Basically a fine-tuned [PhoBERT](https://github.com/VinAIResearch/PhoBERT) for simple 3-class sentiment classification (Negative / Neutral / Positive) on Vietnamese product reviews.

## Results

- **Test Accuracy:** 81.4%
- **Dataset:** 7,786 training / 2,224 test samples
- **Training:** 3 epochs on Apple M-series GPU (MPS)

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("svn05/vietnamese-sentiment-phobert")
tokenizer = AutoTokenizer.from_pretrained("svn05/vietnamese-sentiment-phobert")

text = "Sản phẩm rất tốt, giao hàng nhanh"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=-1).item()

labels = ['Negative', 'Neutral', 'Positive']
print(f"Sentiment: {labels[prediction]}")
```

## Training

```bash
pip install torch transformers datasets
python train.py
```

## Labels/Legend

Label/Meaning/Original Ratin
0/Negative/1-2 stars
1/Neutral/3 stars
2/Positive/4-5 stars

## Dataset

[thanhchauns2/vietnamese-sentiment-analysis](https://huggingface.co/datasets/thanhchauns2/vietnamese-sentiment-analysis) — Vietnamese product reviews from a range of a bunch of e-commerce platforms

```

Create `.gitignore`:
```

venv/
**pycache**/
.DS_Store
