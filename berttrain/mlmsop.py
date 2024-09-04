import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForPreTraining, get_linear_schedule_with_warmup
from torch.optim import AdamW
import logging

# Paths
TEXT_DIR = './dataset.txt'
MODEL_OUTPUT_DIR = './bertje-mlm-sop-model'
CHECKPOINT_DIR = './model_checkpoints'
LOCAL_MODEL_DIR = './bertje'

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


def train(file_path):
    # Identify primary GPU (e.g., lowest available device ID)
    if torch.cuda.is_available():
        primary_gpu = torch.device("cuda")
    else:
        primary_gpu = torch.device("cpu")
        logger.info('No CUDA devices available, using CPU')
    device = primary_gpu
    logger.info(f'Using device: {device}')

    # Initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    logger.info('Initialized tokenizer')
    print('Initialized tokenizer')

    model = BertForPreTraining.from_pretrained(LOCAL_MODEL_DIR)
    logger.info('Initialized model')
    print('Initialized model')

    sop_pairs = create_input(file_path)
    print("SOP Text created")

    # Prepare inputs for MLM and SOP
    mlm_text = [f"{pair[0]} {tokenizer.sep_token} {pair[1]}" for pair in sop_pairs]
    print("MLM Text created")
    token_input = tokenizer(mlm_text, return_tensors='pt', max_length=MAX_LENGTH, truncation=True, padding='max_length')
    print("inputs created")
    token_input = create_mlm_sop_labels(token_input, mask_prob=MASK_PROB, sentence_pairs=sop_pairs, tokenizer=tokenizer)
    print("inputs labeled")
    logger.info('Loaded inputs with labels')

    # Clean variables to save RAM
    sop_pairs.clear()
    mlm_text.clear()
    print("Old variables cleaned")

    model = model.to(device)
    logger.info(f'Model moved to {device}')
    print(f'Model running on {device}')

    # Create dataset and dataloader
    dataset = RechtDataset(token_input)
    logger.info('Loaded dataset')
    print('Loaded dataset')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    logger.info('Loaded dataloader')
    print('Loaded dataloader')

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    logger.info(f'Set up optimizer with learning rate of {LEARNING_RATE} and weight decay of {WEIGHT_DECAY}')
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)
    logger.info('Linear scheduler initialized')

    # Training loop
    model.train()
    logger.info('Started training')
    print('Started training')
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

    #         # Log only on the main GPU
    #         if torch.cuda.current_device() == primary_gpu.index or not torch.cuda.is_available():
    #             logger.info(
    #                 f'Batch Loss: {total_loss.item()} | MLM Loss: {mlm_loss.item()} | SOP Loss: {sop_loss.item()} '
    #                 f'on main GPU {primary_gpu.index}')
    #         else:
    #             logger.info(
    #                 f'Batch Loss: {total_loss.item()} | MLM Loss: {mlm_loss.item()} | SOP Loss: {sop_loss.item()} '
    #                 f'on GPU {torch.cuda.current_device()}')
    #
    #     # Log epoch losses only on the main GPU
    #     if torch.cuda.current_device() == primary_gpu.index or not torch.cuda.is_available():
    #         logger.info(
    #             f'Epoch {epoch + 1} | Total Loss: {total_epoch_loss / len(loader)} | '
    #             f'MLM Loss: {mlm_epoch_loss / len(loader)} | SOP Loss: {sop_epoch_loss / len(loader)} on main GPU')
    #     else:
    #         logger.info(
    #             f'Epoch {epoch + 1} | Total Loss: {total_epoch_loss / len(loader)} | '
    #             f'MLM Loss: {mlm_epoch_loss / len(loader)} | SOP Loss: {sop_epoch_loss / len(loader)} '
    #             f'on GPU {torch.cuda.current_device()}')
    #
    #     # Save model checkpoint (from the main GPU)
    #     if torch.cuda.current_device() == primary_gpu.index or not torch.cuda.is_available():
    #         save_checkpoint(model, epoch)
    #         logger.info(f'Saving checkpoint to {CHECKPOINT_DIR}')
    #     else:
    #         logger.info(f'Trying to save from not the ')
    #
    # logger.info('Ended training')

            logger.info(
                f'Batch Loss: {total_loss.item()} | MLM Loss: {mlm_loss.item()} | SOP Loss: {sop_loss.item()} '
                f'on GPU {torch.cuda.current_device()}')

        logger.info(
            f'Epoch {epoch + 1} | Total Loss: {total_epoch_loss / len(loader)} | '
            f'MLM Loss: {mlm_epoch_loss / len(loader)} | SOP Loss: {sop_epoch_loss / len(loader)} '
            f'on GPU {torch.cuda.current_device()}')

        save_checkpoint(model, epoch)
        logger.info(f'Saving checkpoint to {CHECKPOINT_DIR}')

    logger.info('Ended training')


def get_gpu_id():
    # Obtain the list of GPUs assigned by SLURM
    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is None:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is not set. This script should be run under a SLURM job with GPUs allocated.")

    # Split the device IDs and select the first one as the primary GPU
    visible_device_ids = [int(x) for x in cuda_visible_devices.split(',')]
    primary_gpu = torch.device(f"cuda:{visible_device_ids[0]}")

    logger.info(f'Using GPUs: {cuda_visible_devices}')
    logger.info(f'Primary GPU set to cuda:{visible_device_ids[0]}')
    return primary_gpu


def create_input(file_path, encoding='utf-8'):
    """Read text data from the given file path with specified encoding."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r', encoding=encoding) as fp:
        temp = []
        pairs = []
        i = 0

        for line in fp.read().splitlines():
            if line.strip():
                temp.append(line)
                if len(temp) == 2:
                    if i % 2 != 0:
                        temp.reverse()
                    pairs.append(list(temp))
                    temp.clear()
                    i += 1
            else:
                temp.clear()
        if temp:
            # One sentence left
            temp.clear()
    logger.info(f'Read pairs lines from {file_path}')
    print(f'Read pairs lines from {file_path}')
    return pairs


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
    print("MLM SOP LABELS CREATED")

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
