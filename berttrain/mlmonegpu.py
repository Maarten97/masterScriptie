import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForPreTraining, get_linear_schedule_with_warmup
from torch.optim import AdamW
import logging

# Paths
TEXT_DIR = './datasetTest.txt'
MODEL_OUTPUT_DIR = './mbert-mlm-sop-model'
CHECKPOINT_DIR = './model_checkpoints'
LOCAL_MODEL_DIR = './mbert'

# Training arguments
MAX_LENGTH = 256
MASK_PROB = 0.15
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
MLM_LOSS_WEIGHT = 1.5
SOP_LOSS_WEIGHT = 0.5
WARMUP_STEPS = 10000

# Set up logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

class RechtDataset(torch.utils.data.Dataset):
    """Custom Dataset for loading and encoding text data with MLM and SOP objectives."""
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def read_text_file(file_path, encoding='utf-8'):
    """Read text data from the given file path with specified encoding."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r', encoding=encoding) as fp:
        return fp.read().splitlines()

def train(file_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    logger.info('Initialized tokenizer')
    model = BertForPreTraining.from_pretrained(LOCAL_MODEL_DIR)
    logger.info('Initialized model')
    model = model.to(device)
    logger.info('Model moved to GPU')

    # Read and prepare the text data
    text = read_text_file(file_path)
    logger.info('Dataset has been read and prepared for training')

    # Split the text into pairs of sentences for SOP task
    sentences = [line for line in text if line.strip()]
    sentence_pairs = [(sentences[i], sentences[i + 1]) for i in range(0, len(sentences) - 1, 2)]
    logger.info('Text split into pairs of sentences for SOP task')

    # Shuffle sentence pairs
    shuffled_pairs = [(pair[1], pair[0]) if i % 2 != 0 else pair for i, pair in enumerate(sentence_pairs)]
    logger.info('Shuffled pairs of sentences')

    # Prepare inputs for MLM and SOP
    texts = [f"{pair[0]} {tokenizer.sep_token} {pair[1]}" for pair in shuffled_pairs]
    inputs = tokenizer(texts, return_tensors='pt', max_length=MAX_LENGTH, truncation=True, padding='max_length')
    inputs = create_mlm_sop_labels(inputs, mask_prob=MASK_PROB, sentence_pairs=shuffled_pairs, tokenizer=tokenizer)
    logger.info('Loaded inputs with labels')

    # Create dataset and dataloader
    dataset = RechtDataset(inputs)
    logger.info('Loaded dataset')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    logger.info('Loaded dataloader')

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    logger.info(f'Set up optimizer with learning rate of {LEARNING_RATE} and weight decay of {WEIGHT_DECAY}')
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)
    logger.info('Linear scheduler initialized')

    # Training loop
    model.train()
    logger.info('Started training')
    for epoch in range(EPOCHS):
        total_epoch_loss = 0
        mlm_epoch_loss = 0
        sop_epoch_loss = 0
        loop = tqdm(loader, leave=True)

        for step, batch in enumerate(loop):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            next_sentence_label = batch['next_sentence_label'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            prediction_logits = outputs.prediction_logits
            seq_relationship_logits = outputs.seq_relationship_logits

            # Calculate MLM loss
            mlm_loss_fn = torch.nn.CrossEntropyLoss()
            mlm_loss = mlm_loss_fn(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
            mlm_epoch_loss += mlm_loss.item()

            # Calculate SOP loss
            sop_loss_fn = torch.nn.CrossEntropyLoss()
            sop_loss = sop_loss_fn(seq_relationship_logits.view(-1, 2), next_sentence_label.view(-1))
            sop_epoch_loss += sop_loss.item()

            # Combine losses with custom weights
            total_loss = (MLM_LOSS_WEIGHT * mlm_loss) + (SOP_LOSS_WEIGHT * sop_loss)
            total_epoch_loss += total_loss.item()

            # Backpropagation and optimization
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            loop.set_description(f'Epoch {epoch + 1}')
            loop.set_postfix(loss=total_loss.item())

            # Log batch losses
            logger.info(f'Batch Loss: {total_loss.item()} | MLM Loss: {mlm_loss.item()} | SOP Loss: {sop_loss.item()}')

        # Log epoch losses
        logger.info(f'Epoch {epoch + 1} | Total Loss: {total_epoch_loss / len(loader)} | MLM Loss: {mlm_epoch_loss / len(loader)} | SOP Loss: {sop_epoch_loss / len(loader)}')

        # Save model checkpoint
        save_checkpoint(model, epoch)

    logger.info('Ended training')

def create_mlm_sop_labels(inputs, mask_prob=MASK_PROB, sentence_pairs=None, tokenizer=None):
    """Create masked language model labels and SOP labels."""
    inputs['labels'] = inputs.input_ids.clone().detach()
    rand = torch.rand(inputs.input_ids.shape)

    # Create mask
    mask_arr = (rand < mask_prob) & (inputs.input_ids != tokenizer.cls_token_id) & \
               (inputs.input_ids != tokenizer.sep_token_id) & (inputs.input_ids != tokenizer.pad_token_id)

    for i in range(inputs.input_ids.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        inputs.input_ids[i, selection] = tokenizer.mask_token_id

    # Create SOP labels
    sop_labels = torch.LongTensor([0 if i % 2 == 0 else 1 for i in range(len(sentence_pairs))])
    inputs['next_sentence_label'] = sop_labels
    logger.info("MLM SOP LABELS CREATED")

    return inputs

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

def save_checkpoint(model, epoch, checkpoint_dir=CHECKPOINT_DIR):
    """Save the model checkpoint."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model.save_pretrained(f'{checkpoint_dir}/model_epoch_{epoch}.bin')
    logger.info(f'Model checkpoint saved for epoch {epoch}')

def main():
    log_hyperparameters()
    train(TEXT_DIR)

if __name__ == "__main__":
    main()
