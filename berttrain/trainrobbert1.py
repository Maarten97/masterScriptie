import os
import torch
import logging

from transformers import RobertaForMaskedLM, RobertaTokenizer

# Paths
STEP = 1
MODEL = 'robbert'
TOKEN_DIR = f'./output_file_{STEP}.pt'
CHECKPOINT_DIR = f'./{MODEL}_check{STEP}'
# LOCAL_MODEL_DIR = f'./{MODEL}_check{STEP - 1}'
LOCAL_MODEL_DIR = './robbert'

# TestPaths
# STEP = 2
# MODEL = 'robbert'
# TOKEN_DIR = './datasetTestkort.pt'
# CHECKPOINT_DIR = f'./{MODEL}_check{STEP}'
# # LOCAL_MODEL_DIR = 'C:/Users/looij/PycharmProjects/masterScriptie/bertmodel/robbert'
# # LOCAL_MODEL_DIR = f'./{MODEL}'
# LOCAL_MODEL_DIR = f'./{MODEL}_check{STEP - 1}'

# Training arguments
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
MLM_LOSS_WEIGHT = 1.25
SOP_LOSS_WEIGHT = 0.75

# Set up logging
logging.basicConfig(filename=f'training_log{STEP}.txt', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()


# Dataset Loader
class RechtDataset(torch.utils.data.Dataset):
    """Custom Dataset for loading and encoding text data with MLM and SOP objectives."""

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def train():
    """General def for training model on MLM and SOP objectives."""
    # Create instance of Model
    if not os.path.exists(LOCAL_MODEL_DIR):
        logger.error("BERT Model not in folder in scratch root")
    model = RobertaForMaskedLM.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
    logger.info('Initialized model')
    print('Initialized model')

    # Create an instance of RechtDataset
    dataset = RechtDataset(loading_data())
    logger.info(f'Loaded {len(dataset)} samples')
    print(f'Loaded {len(dataset)} samples')

    # Initialize Loader and Optimizer
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    logger.info(f'Loaded {len(loader)} batches with {BATCH_SIZE} samples')
    print(f'Loaded {len(loader)} batches with {BATCH_SIZE} samples')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    logger.info(f'Set up optimizer with learning rate of {LEARNING_RATE} and weight decay of {WEIGHT_DECAY}')
    print(f'Set up optimizer with learning rate of {LEARNING_RATE}')

    # Define loss functions
    mlm_loss_fn = torch.nn.CrossEntropyLoss()

    # Check for GPU availability
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    logger.info(f'Initialized device on: {device}')
    print(f'Initialized device on: {device}')

    # Move to GPU
    model.to(device)

    # Start Training Loop
    model.train()
    logger.info('Started training')
    print('Started training')

    for epoch in range(EPOCHS):
        total_epoch_loss = 0

        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            mlm_logits = outputs.logits
            # Compute MLM loss (CrossEntropyLoss is applied to the MLM logits and the labels)
            mlm_loss = mlm_loss_fn(mlm_logits.view(-1, mlm_logits.size(-1)), labels.view(-1))

            # Add losses to epoch logger
            total_epoch_loss += mlm_loss.item()

            # Add loss to logger
            logger.info(f'Batch MLM Loss: {mlm_loss.item()} ')

            # calculate loss for every parameter that needs grad update
            mlm_loss.backward()

            # update parameters
            optimizer.step()

        # Log epoch statistics
        avg_epoch_loss = total_epoch_loss / len(loader)

        logger.info(
            f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_epoch_loss:.4f}')
        print(
            f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_epoch_loss:.4f}')

    # Saving checkpoint
    save_model(model)

    logger.info('Finished training')
    logger.info('Model saved')


def loading_data(token_dir=TOKEN_DIR):
    """Loading pre-tokenized data into Dataloader"""
    if not os.path.exists(token_dir):
        raise FileNotFoundError(f"The file {token_dir} does not exist.")
    return torch.load(token_dir, map_location=torch.device('cpu'))


def save_model(model, checkpoint_dir=CHECKPOINT_DIR):
    """Save the model checkpoint."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model.save_pretrained(checkpoint_dir)
    tokenizer = RobertaTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    tokenizer.save_pretrained(checkpoint_dir)
    logger.info(f'Model checkpoint saved for step {STEP}')


def log_hyperparameters():
    """Log the hyperparameters used for the training."""
    hyperparameters = {
        'STEP': STEP,
        'MODEL': MODEL,
        'PRETRAINED_MODEL_NAME': LOCAL_MODEL_DIR,
        'TOKENIZER_NAME': LOCAL_MODEL_DIR,
        'BATCH_SIZE': BATCH_SIZE,
        'EPOCHS': EPOCHS,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'MLM_LOSS_WEIGHT': MLM_LOSS_WEIGHT,
        'SOP_LOSS_WEIGHT': SOP_LOSS_WEIGHT,
    }
    logger.info(f'Hyperparameters: {hyperparameters}')


def main():
    # Set up and initialize Logging
    log_hyperparameters()
    # Load pretokenized dataset into Dataset
    logger.info('Start train method')
    print('Start train method')
    train()


if __name__ == "__main__":
    main()
