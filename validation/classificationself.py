import torch
import logging
import os
from transformers import BertForPreTraining, BertTokenizer
from datasets import load_dataset
import evaluate
import json

MODEL = 'bertje'
MODEL_DIR = 'path'

EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

# Set up logging
logging.basicConfig(filename=f'validation_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

def initialize_model():
    """Prepare the data for training and validation"""
    # Create instance of Model
    if not os.path.exists(MODEL_DIR):
        logger.error("BERT Model not in folder in scratch root")
        raise FileNotFoundError
    model = BertForPreTraining.from_pretrained(MODEL_DIR, local_files_only=True)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    logger.info('Initialized model')
    return model, tokenizer

def initialize_device():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    logger.info(f'Initialized device on: {device}')
    return device

def prepare_data():
    """Prepare the data for training and validation"""
    dataset = load_dataset('coastalcph/multi_eurlex', 'nl', trust_remote_code=True)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    eval_dataset = dataset['validation']

    with open('eurovoc_concepts.json') as file:
        label_index = {idx: concept for idx, concept in enumerate(json.load(file)['level_1'])}
        logger.info(f'EUROVOC Concepts: {'level_1'} ({len(label_index)})')

    return train_dataset, test_dataset, eval_dataset

def token_data(dataset, tokenizer):
    """Tokenize the data"""
    tokenized = ''
    return tokenized

def train():
    """Train the data"""
    model, tokenizer = initialize_model()
    print('model initializd')
    train_dataset, test_dataset, eval_dataset = prepare_data()
    print('data prepared')
    train_token = token_data(train_dataset, tokenizer)
    test_token = token_data(test_dataset, tokenizer)
    eval_token = token_data(eval_dataset, tokenizer)
    print('data tokenized')

def valid():
    """Save the model output and metrics."""


def log_hyperparameters():
    """Log the hyperparameters used for the training."""
    hyperparameters = {
        'MODEL': MODEL,
        'PRETRAINED_MODEL_NAME': MODEL_DIR,
        'TOKENIZER_NAME': MODEL_DIR,
        'BATCH_SIZE': BATCH_SIZE,
        'EPOCHS': EPOCHS,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
    }
    logger.info(f'Hyperparameters: {hyperparameters}')

def main():
    # Set up and initialize Logging
    log_hyperparameters()
    # Load pretokenized dataset into Dataset
    logger.info('Start train method')
    print('Start train method')
    train()