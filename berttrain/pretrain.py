import os
import torch
import logging

from transformers import BertForPreTraining

# Paths
TOKEN_DIR = './tokenized_chunk_0.pt'
MODEL_OUTPUT_DIR = './bertje-mlm-sop-model'
CHECKPOINT_DIR = './model_checkpoints'
LOCAL_MODEL_DIR = './bertje'

# Training arguments
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
MLM_LOSS_WEIGHT = 1.3
SOP_LOSS_WEIGHT = 0.7
WARMUP_STEPS = 10000

# Set up logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')
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
    model = BertForPreTraining.from_pretrained(LOCAL_MODEL_DIR)
    logger.info('Initialized model')

    # Create an instance of RechtDataset
    dataset = RechtDataset(loading_data())
    logger.info(f'Loaded {len(dataset)} samples')

    # Initialize Loader and Optimizer
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    logger.info(f'Loaded {len(loader)} batches with {BATCH_SIZE} samples')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    logger.info(f'Set up optimizer with learning rate of {LEARNING_RATE} and weight decay of {WEIGHT_DECAY}')

    # Define loss functions
    mlm_loss_fn = torch.nn.CrossEntropyLoss()
    sop_loss_fn = torch.nn.CrossEntropyLoss()

    # Check for GPU availability
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    logger.info(f'Initialized device on: {device}')

    # Move to GPU
    model.to(device)

    #Start Training Loop
    model.train()
    logger.info('Started training')
    print('Started training')

    for epoch in range(EPOCHS):
        total_epoch_loss = 0
        mlm_epoch_loss = 0
        sop_epoch_loss = 0

        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sop_labels = batch['sop_labels'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            labels=labels, next_sentence_label=sop_labels)

            # # Extract MLM and SOP loss from the outputs
            # mlm_loss = outputs.loss  # Masked Language Modeling loss
            # sop_loss = outputs.next_sentence_loss  # Sentence Order Prediction loss

            # Extract logits for MLM and SOP
            mlm_logits = outputs.prediction_logits
            sop_logits = outputs.seq_relationship_logits

            # Compute MLM loss (CrossEntropyLoss is applied to the MLM logits and the labels)
            mlm_loss = mlm_loss_fn(mlm_logits.view(-1, mlm_logits.size(-1)), labels.view(-1))

            # Compute SOP loss (CrossEntropyLoss is applied to SOP logits and labels)
            sop_loss = sop_loss_fn(sop_logits.view(-1, 2), sop_labels.view(-1))

            # Combine losses
            loss = (MLM_LOSS_WEIGHT * mlm_loss) + (SOP_LOSS_WEIGHT * sop_loss)

            # Add losses to epoch logger
            total_epoch_loss += loss.item()
            mlm_epoch_loss += mlm_loss.item()
            sop_epoch_loss += sop_loss.item()

            # Add loss to logger
            logger.info(f'Batch Loss: {loss.item()} | MLM Loss: {mlm_loss.item()} | SOP Loss: {sop_loss.item()}')

            # calculate loss for every parameter that needs grad update
            loss.backward()

            # update parameters
            optimizer.step()

            # Clear cache and delete tensors
            del input_ids, token_type_ids, attention_mask, sop_labels, labels, outputs, mlm_logits, sop_logits, mlm_loss, sop_loss, loss
            torch.cuda.empty_cache()

        # Log epoch statistics
        avg_epoch_loss = total_epoch_loss / len(loader)
        avg_mlm_loss = mlm_epoch_loss / len(loader)
        avg_sop_loss = sop_epoch_loss / len(loader)

        logger.info(
            f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_epoch_loss:.4f}, '
            f'MLM Loss: {avg_mlm_loss:.4f}, SOP Loss: {avg_sop_loss:.4f}')
        print(
            f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_epoch_loss:.4f}, '
            f'MLM Loss: {avg_mlm_loss:.4f}, SOP Loss: {avg_sop_loss:.4f}')

        #Saving checkpoint for every Epoch
        save_model(model, epoch)

    logger.info('Finished training')
    logger.info('Model saved')


def loading_data(token_dir=TOKEN_DIR):
    """Loading pre-tokenized data into Dataloader"""
    if not os.path.exists(token_dir):
        raise FileNotFoundError(f"The file {token_dir} does not exist.")
    return torch.load(token_dir, map_location=torch.device('cpu'))


def save_model(model, epoch, checkpoint_dir=CHECKPOINT_DIR):
    """Save the model checkpoint."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model.save_pretrained(f'{checkpoint_dir}/model_epoch_{epoch}.bin')
    logger.info(f'Model checkpoint saved for epoch {epoch}')

def log_hyperparameters():
    """Log the hyperparameters used for the training."""
    hyperparameters = {
        'PRETRAINED_MODEL_NAME': LOCAL_MODEL_DIR,
        'TOKENIZER_NAME': LOCAL_MODEL_DIR,
        'MAX_LENGTH': MAX_LENGTH,
        'MASK_PROB': MASK_PROB,
        'BATCH_SIZE': BATCH_SIZE,
        'EPOCHS': EPOCHS,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'MLM_LOSS_WEIGHT': MLM_LOSS_WEIGHT,
        'SOP_LOSS_WEIGHT': SOP_LOSS_WEIGHT,
    }
    logger.info(f'Hyperparameters: {hyperparameters}')


def main():
    # Set up and initiale Logging
    log_hyperparameters()
    # Load pretokenized dataset into Dataset
    train()


if __name__ == "__main__":
    main()
