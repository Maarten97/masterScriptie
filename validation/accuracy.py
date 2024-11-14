import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from datasets import load_from_disk
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from torch.utils.data import DataLoader

# Setup logging configuration
logging.basicConfig(
    filename='training.log',  # Log file name
    filemode='a',  # Append mode
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
)
logger = logging.getLogger()

MODEL_DIR = './bertje'

# Check if GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the tokenizer for multilingual BERT
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

# Load only the Dutch part of the MultiEURLEX dataset
dataset = load_from_disk('./library')


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
train_dataset = train_dataset.remove_columns(['celex_id', 'text']).with_format('torch')
eval_dataset = eval_dataset.remove_columns(['celex_id', 'text']).with_format('torch')
test_dataset = test_dataset.remove_columns(['celex_id', 'text']).with_format('torch')

# Create DataLoader instances for each dataset split
batch_size = 16  # Adjust batch size as needed

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# Model setup: Binary Cross Entropy Loss and metrics calculation
class MultiLabelBERT(nn.Module):
    def __init__(self, num_labels):
        super(MultiLabelBERT, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_DIR)
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

# Metrics calculation function
def calculate_metrics(true_labels, pred_labels):
    results = {}
    results['accuracy'] = accuracy_score(true_labels, pred_labels)
    results['precision'] = precision_score(true_labels, pred_labels, average=None)
    results['recall'] = recall_score(true_labels, pred_labels, average=None)
    results['micro_f1'] = f1_score(true_labels, pred_labels, average='micro')
    results['macro_f1'] = f1_score(true_labels, pred_labels, average='macro')
    return results

# Test set evaluation function
def evaluate_test_set(model, test_dataloader):
    model.eval()
    test_true_labels = []
    test_pred_labels = []
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            test_loss += loss.item()

            preds = torch.sigmoid(logits).cpu().numpy()
            test_pred_labels.extend((preds > 0.5).astype(int))
            test_true_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_dataloader)
    test_true_labels = np.array(test_true_labels)
    test_pred_labels = np.array(test_pred_labels)

    test_metrics = calculate_metrics(test_true_labels, test_pred_labels)
    test_metrics['loss'] = test_loss
    return test_metrics


# Training and evaluation loop with logging
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    true_labels = []
    pred_labels = []

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).float()

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        pred_labels.extend((preds > 0.5).astype(int))
        true_labels.extend(labels.cpu().numpy())

    train_loss /= len(train_dataloader)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Calculate and log training metrics
    train_metrics = calculate_metrics(true_labels, pred_labels)
    logger.info(f"Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f}")
    logger.info(f"Training Accuracy: {train_metrics['accuracy']:.4f} | "
                f"Micro F1: {train_metrics['micro_f1']:.4f} | "
                f"Macro F1: {train_metrics['macro_f1']:.4f}")

    # Validation loop
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

    # Calculate and log validation metrics
    val_metrics = calculate_metrics(val_true_labels, val_pred_labels)
    logger.info(f"Validation Loss: {val_loss:.4f}")
    logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f} | "
                f"Validation Micro F1: {val_metrics['micro_f1']:.4f} | "
                f"Validation Macro F1: {val_metrics['macro_f1']:.4f}")

# Final test evaluation after training
test_metrics = evaluate_test_set(model, test_dataloader)
logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f} | "
            f"Test Micro F1: {test_metrics['micro_f1']:.4f} | "
            f"Test Macro F1: {test_metrics['macro_f1']:.4f}")

# Metrics summary
metrics_summary = {
    'train': train_metrics,
    'validation': val_metrics,
    'test': test_metrics
}
