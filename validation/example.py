import numpy as np
from datasets import load_dataset
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from torch.utils.data import DataLoader

MODEL_DIR = 'C:/Users/looij/PycharmProjects/masterScriptie/bertmodel/bertje'

# Check if GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the tokenizer for multilingual BERT
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

# Load only the Dutch part of the MultiEURLEX dataset
dataset = load_dataset('coastalcph/multi_eurlex', 'nl', trust_remote_code=True)

# Tokenize the Dutch dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert labels to a fixed-size tensor for multi-label classification
def multi_hot_encode_labels(batch):
    num_labels = len(dataset['train'].features['labels'].feature.names)
    encoded_labels = torch.zeros((len(batch['labels']), num_labels), dtype=torch.float32)
    for i, labels in enumerate(batch['labels']):
        encoded_labels[i, labels] = 1.0  # Multi-hot encoding
    batch['labels'] = encoded_labels
    return batch

tokenized_datasets = tokenized_datasets.map(multi_hot_encode_labels, batched=True)
num_labels = len(dataset['train'].features['labels'].feature.names)

# Set up the train, validation, and test splits
train_dataset = tokenized_datasets['train'].shuffle(seed=42)
eval_dataset = tokenized_datasets['validation']
test_dataset = tokenized_datasets['test']

# Remove unnecessary columns and set format for PyTorch tensors
train_dataset = train_dataset.remove_columns(['text']).with_format('torch')
eval_dataset = eval_dataset.remove_columns(['text']).with_format('torch')
test_dataset = test_dataset.remove_columns(['text']).with_format('torch')

# Create DataLoader instances for each dataset split
batch_size = 16  # Adjust batch size as needed

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Model setup: Binary Cross Entropy Loss and metrics calculation
class MultiLabelBERT(nn.Module):
    def __init__(self, num_labels):
        super(MultiLabelBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits

# Instantiate model, loss function, and optimizer
model = MultiLabelBERT(num_labels=num_labels).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

# Scheduler setup for learning rate decay
epochs = 5
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training and Evaluation Loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    true_labels = []
    pred_labels = []

    # Training loop
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).float()  # Labels should be in float for BCE loss

        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

        # Collect predictions for metrics calculation
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        pred_labels.extend((preds > 0.5).astype(int))  # Threshold for multi-label classification
        true_labels.extend(labels.cpu().numpy())

    # Calculate epoch metrics
    train_loss /= len(train_dataloader)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Training Loss: {train_loss:.4f} | Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")

    # Validation step
    model.eval()
    val_loss = 0.0
    val_true_labels = []
    val_pred_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            preds = torch.sigmoid(logits).cpu().numpy()
            val_pred_labels.extend((preds > 0.5).astype(int))
            val_true_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_dataloader)
    val_true_labels = np.array(val_true_labels)
    val_pred_labels = np.array(val_pred_labels)
    val_accuracy = accuracy_score(val_true_labels, val_pred_labels)
    val_f1 = f1_score(val_true_labels, val_pred_labels, average='weighted')

    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f} | Validation F1 Score: {val_f1:.4f}")
