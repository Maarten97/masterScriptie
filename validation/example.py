import logging
import numpy as np
from datasets import load_from_disk, disable_progress_bar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

# Setup logging configuration
logging.basicConfig(
    filename='training.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger()
disable_progress_bar()



# Model definition
class MultiLabelBERT(nn.Module):
    def __init__(self, num_labels, model_dir):
        super(MultiLabelBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_dir)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits


# Tokenization
def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)


# Multi-hot encode labels
def multi_hot_encode_labels(batch, num_labels):
    encoded_labels = torch.zeros((len(batch['labels']), num_labels), dtype=torch.float32)
    for i, labels in enumerate(batch['labels']):
        encoded_labels[i, labels] = 1.0
    batch['labels'] = encoded_labels
    return batch


# Calculate metrics
def calculate_metrics(true_labels, pred_labels):
    results = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels, average=None, zero_division=0),
        'recall': recall_score(true_labels, pred_labels, average=None, zero_division=0),
        'micro_f1': f1_score(true_labels, pred_labels, average='micro', zero_division=0),
        'macro_f1': f1_score(true_labels, pred_labels, average='macro', zero_division=0),
    }
    return results


# Training
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    train_loss = 0.0
    true_labels = []
    pred_labels = []

    for batch in tqdm(dataloader, desc="Training", disable=True):
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

    train_loss /= len(dataloader)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    return train_loss, true_labels, pred_labels


# Evaluation
def evaluate(model, dataloader, criterion, device):
    model.eval()
    eval_loss = 0.0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=True):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            eval_loss += loss.item()

            preds = torch.sigmoid(logits).cpu().numpy()
            pred_labels.extend((preds > 0.5).astype(int))
            true_labels.extend(labels.cpu().numpy())

    eval_loss /= len(dataloader)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    return eval_loss, true_labels, pred_labels


# Main method
def main():
    # Configuration
    model_dir = './bertje'
    dataset_path = './library'
    batch_size = 16
    epochs = 5
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load tokenizer and dataset
    tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
    dataset = load_from_disk(dataset_path)

    # # Subset before tokenization for faster testing
    # train_dataset = dataset['train'].select(range(100))
    # val_dataset = dataset['validation'].select(range(50))
    # test_dataset = dataset['test'].select(range(50))
    #
    # Subset before tokenization for faster testing
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    # Tokenize the subsets
    tokenized_train = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    tokenized_val = val_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    tokenized_test = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

    # Multi-hot encode labels
    num_labels = len(dataset['train'].features['labels'].feature.names)
    tokenized_train = tokenized_train.map(lambda batch: multi_hot_encode_labels(batch, num_labels), batched=True)
    tokenized_val = tokenized_val.map(lambda batch: multi_hot_encode_labels(batch, num_labels), batched=True)
    tokenized_test = tokenized_test.map(lambda batch: multi_hot_encode_labels(batch, num_labels), batched=True)

    # Convert to PyTorch format and remove unnecessary columns
    train_dataset = tokenized_train.remove_columns(['celex_id', 'text']).with_format('torch')
    val_dataset = tokenized_val.remove_columns(['celex_id', 'text']).with_format('torch')
    test_dataset = tokenized_test.remove_columns(['celex_id', 'text']).with_format('torch')

    # Create DataLoader instances
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model, optimizer, and scheduler
    model = MultiLabelBERT(num_labels, model_dir).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)

    # Training loop
    for epoch in range(epochs):
        train_loss, train_true, train_preds = train_epoch(
            model, train_dataloader, optimizer, scheduler, criterion, device
        )
        train_metrics = calculate_metrics(train_true, train_preds)
        logger.info(f"Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f}")
        logger.info(f"Training Metrics: {train_metrics}")

        val_loss, val_true, val_preds = evaluate(model, val_dataloader, criterion, device)
        val_metrics = calculate_metrics(val_true, val_preds)
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Metrics: {val_metrics}")

    # Final test evaluation
    test_loss, test_true, test_preds = evaluate(model, test_dataloader, criterion, device)
    test_metrics = calculate_metrics(test_true, test_preds)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Metrics: {test_metrics}")


if __name__ == "__main__":
    main()

