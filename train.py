import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from datasets import load_dataset
from tqdm import tqdm

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device}")

# Load tokenizer and model (3 classes: neg, neu, pos)
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.to(device)
print("✓ Model loaded")

# Load dataset
dataset = load_dataset("thanhchauns2/vietnamese-sentiment-analysis")
print(f"✓ Dataset loaded: {len(dataset['train'])} train samples")

# Convert 5-star to 3-class: 1-2 = negative(0), 3 = neutral(1), 4-5 = positive(2)
def convert_label(example):
    label = example['label']
    if label <= 2:
        example['label'] = 0
    elif label == 3:
        example['label'] = 1
    else:
        example['label'] = 2
    return example

dataset = dataset.map(convert_label)
print("✓ Labels converted")

# Tokenize
def tokenize(examples):
    return tokenizer(examples['comment'], truncation=True, padding='max_length', max_length=256)

dataset = dataset.map(tokenize, batched=True, remove_columns=['comment'])
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
print("✓ Tokenized")

# Quick sanity check
print(f"Sample batch shape: {dataset['train'][0]['input_ids'].shape}")

# DataLoaders
train_loader = DataLoader(dataset['train'], batch_size=16, shuffle=True)
test_loader = DataLoader(dataset['test'], batch_size=16)
print(f"✓ DataLoaders ready: {len(train_loader)} train batches")

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
print(f"\nTraining for {num_epochs} epochs...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch in progress:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(input_ids=batch['input_ids'], 
                       attention_mask=batch['attention_mask'],
                       labels=batch['label'])
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        progress.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

# Evaluation
print("\nEvaluating...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == batch['label']).sum().item()
        total += len(batch['label'])

accuracy = correct / total
print(f"\n✓ Test Accuracy: {accuracy:.1%}")

# Save model
model.save_pretrained("phobert-sentiment")
tokenizer.save_pretrained("phobert-sentiment")
print("✓ Model saved to phobert-sentiment/")