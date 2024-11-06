import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_scheduler, Trainer, TrainingArguments
from datasets import load_dataset
import logging
import os
import json
from tqdm import tqdm
import evaluate

MODEL = 'bertje'
MODEL_DIR = 'C:/Users/looijengam/PycharmProjects/masterScriptie/bert/bertje'
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

# Set up logging
logging.basicConfig(filename='validation_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

def initialize_model():
    if not os.path.exists(MODEL_DIR):
        logger.error("BERT Model not found in the specified directory.")
        raise FileNotFoundError("BERT Model not in specified directory.")
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    logger.info('Model and tokenizer initialized.')
    return model

def initialize_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    logger.info(f'Device initialized on: {device}')
    return device

def prepare_data(split):
    dataset = load_dataset('coastalcph/multi_eurlex', 'nl', split=split, trust_remote_code=True)
    return dataset

def token_data(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')

def load_labels(label_level):
    with open('./eurovoc_concepts.json') as file:
        return {idx: concept for idx, concept in enumerate(json.load(file)[label_level])}

def token_label(batch, label_index):
    num_labels = len(label_index)
    token_labels = torch.zeros((len(batch['labels']), num_labels), dtype=torch.float32)
    for i, labels in enumerate(batch['labels']):
        token_labels[i, labels] = 1.0
    batch['labels'] = token_labels
    return batch

def train():
    model = initialize_model()
    device = initialize_device()

    # Load and tokenize data
    train_dataset = prepare_data('train')
    label_index = load_labels('level_1')
    train_dataset = train_dataset.map(token_data, batched=True)
    train_dataset = train_dataset.map(token_label, batched=True)

    eval_dataset = prepare_data('validation')
    eval_dataset = eval_dataset.map(lambda x: token_data(x, tokenizer), batched=True)
    eval_dataset = eval_dataset.map(lambda x: token_label(x, label_index), batched=True)

    print()



    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

#
# def valid():


def log_hyperparameters():
    hyperparameters = {
        'MODEL': MODEL,
        'MODEL_DIR': MODEL_DIR,
        'BATCH_SIZE': BATCH_SIZE,
        'EPOCHS': EPOCHS,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
    }
    logger.info(f'Hyperparameters: {hyperparameters}')

def main():
    log_hyperparameters()
    logger.info('Starting training process')
    train()
    logger.info('Starting validation process')
    # valid()

if __name__ == '__main__':
    main()
