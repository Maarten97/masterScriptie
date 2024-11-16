import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from datasets import load_from_disk, disable_progress_bar
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import pandas as pd

# Setup logging configuration
logging.basicConfig(
    filename='validation.txt',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger()
disable_progress_bar()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_and_preprocess_dataset(dataset_path, model_dir):
    logger.info("Loading dataset...")
    dataset = load_from_disk(dataset_path)

    logger.info("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=1000)
    logger.info("Tokenized dataset")

    num_labels = len(dataset['train'].features['labels'].feature.names)
    logger.info(f"Number of labels: {num_labels}")

    def multi_hot_encode_labels(batch):
        encoded_labels = torch.zeros((len(batch['labels']), num_labels), dtype=torch.float32)
        for i, labels in enumerate(batch['labels']):
            encoded_labels[i, labels] = 1.0
        batch['labels'] = encoded_labels
        return batch

    logger.info("Encoding labels...")
    tokenized_datasets = tokenized_datasets.map(multi_hot_encode_labels, batched=True)
    logger.info("Labels encoded")

    return tokenized_datasets, num_labels

def create_dataloaders(tokenized_datasets, batch_size):
    logger.info("Creating dataloaders...")
    train_dataset = tokenized_datasets['train'].shuffle(seed=42).remove_columns(['celex_id', 'text']).with_format('torch')
    eval_dataset = tokenized_datasets['validation'].remove_columns(['celex_id', 'text']).with_format('torch')
    test_dataset = tokenized_datasets['test'].remove_columns(['celex_id', 'text']).with_format('torch')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader

class MultiLabelBERT(nn.Module):
    def __init__(self, num_labels):
        super(MultiLabelBERT, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_DIR)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits

def calculate_metrics(true_labels, pred_labels):
    results = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels, average=None, zero_division=0),
        "recall": recall_score(true_labels, pred_labels, average=None, zero_division=0),
        "micro_f1": f1_score(true_labels, pred_labels, average='micro'),
        "macro_f1": f1_score(true_labels, pred_labels, average='macro'),
    }
    return results

def compute_pos_weight(train_dataset, num_labels):
    logger.info("Computing pos_weight for class imbalance...")
    label_counts = np.zeros(num_labels)

    # Iterate through each sample's labels
    for labels in train_dataset['labels']:
        if isinstance(labels, list):  # Handle list of labels directly
            for label in labels:
                label_counts[label] += 1
        else:  # Already a tensor
            label_counts += labels.numpy()

    total_samples = len(train_dataset)
    pos_weight = torch.tensor((total_samples - label_counts) / label_counts, dtype=torch.float32).to(device)
    logger.info(f"Computed pos_weight: {pos_weight}")
    return pos_weight

def optimize_class_thresholds(y_true, y_pred_probs):
    logger.info("Optimizing per-class thresholds...")
    best_thresholds = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        best_threshold = 0.5
        best_score = 0
        for threshold in np.linspace(0.1, 0.9, 100):
            y_pred = (y_pred_probs[:, i] > threshold).astype(int)
            score = f1_score(y_true[:, i], y_pred)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        best_thresholds[i] = best_threshold
    logger.info(f"Optimal per-class thresholds: {best_thresholds}")
    return best_thresholds

def evaluate_model_with_thresholds(model, dataloader, criterion, thresholds=None):
    model.eval()
    total_loss = 0.0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits).cpu().numpy()
            if thresholds is not None:
                pred_labels.extend((preds > thresholds).astype(int))
            else:
                pred_labels.extend((preds > 0.5).astype(int))
            true_labels.extend(labels.cpu().numpy())

    total_loss /= len(dataloader)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    metrics = calculate_metrics(true_labels, pred_labels)
    return total_loss, metrics, true_labels, pred_labels

if __name__ == "__main__":
    MODEL_DIR = './bertje'
    DATASET_PATH = './library'
    BATCH_SIZE = 16

    tokenized_datasets, num_labels = load_and_preprocess_dataset(DATASET_PATH, MODEL_DIR)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(tokenized_datasets, BATCH_SIZE)

    pos_weight = compute_pos_weight(tokenized_datasets['train'], num_labels)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = MultiLabelBERT(num_labels=num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 5
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    for epoch in range(epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0.0

        for batch in train_dataloader:
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

        val_loss, val_metrics, val_true, val_preds = evaluate_model_with_thresholds(model, val_dataloader, criterion)
        thresholds = optimize_class_thresholds(val_true, val_preds)

        per_class_metrics = {
            "Threshold": thresholds,
            "Precision": precision_score(val_true, (val_preds > thresholds).astype(int), average=None, zero_division=0),
            "Recall": recall_score(val_true, (val_preds > thresholds).astype(int), average=None, zero_division=0),
        }
        per_class_df = pd.DataFrame(per_class_metrics)
        logger.info(f"\n{per_class_df.to_string(index=False)}")

        logger.info(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Micro F1: {val_metrics['micro_f1']:.4f} | Macro F1: {val_metrics['macro_f1']:.4f}")

    test_loss, test_metrics, _, _ = evaluate_model_with_thresholds(model, test_dataloader, criterion, thresholds)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Micro F1: {test_metrics['micro_f1']:.4f} | Test Macro F1: {test_metrics['macro_f1']:.4f}")
