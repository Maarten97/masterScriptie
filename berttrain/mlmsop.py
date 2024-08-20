import os
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForPreTraining
from torch.optim import AdamW

# Paths
TEXT_DIR = 'C:/Users/looijengam/Documents/datasetRandom4.txt'
MODEL_OUTPUT_DIR = './bertje-mlm-sop-model'

# Training arguments
PRETRAINED_MODEL_NAME = 'GroNLP/bert-base-dutch-cased'
TOKENIZER_NAME = 'GroNLP/bert-base-dutch-cased'
MAX_LENGTH = 512
MASK_PROB = 0.15
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
MLM_LOSS_WEIGHT = 1.5
SOP_LOSS_WEIGHT = 0.5

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

print(f'Using device: {device}, number of GPUs: {num_gpus}')

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
model = BertForPreTraining.from_pretrained(PRETRAINED_MODEL_NAME)

# Use DataParallel if multiple GPUs are available
if num_gpus > 1:
    model = torch.nn.DataParallel(model)

# Move model to the selected device
model.to(device)


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


def create_mlm_sop_labels(inputs, mask_prob=MASK_PROB, sentence_pairs=None):
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

    return inputs


def train_model(model, loader, device, epochs, lr, weight_decay, mlm_loss_weight, sop_loss_weight):
    """Train the BERT model with weighted MLM and SOP losses."""
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Define loss functions
    mlm_loss_fn = torch.nn.CrossEntropyLoss()  # Used for MLM
    sop_loss_fn = torch.nn.CrossEntropyLoss()  # Used for SOP

    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        for batch in loop:
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
            mlm_loss = mlm_loss_fn(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))

            # Calculate SOP loss
            sop_loss = sop_loss_fn(seq_relationship_logits.view(-1, 2), next_sentence_label.view(-1))

            # Combine losses with custom weights
            total_loss = (mlm_loss_weight * mlm_loss) + (sop_loss_weight * sop_loss)

            # Backpropagation and optimization
            total_loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch {epoch + 1}')
            loop.set_postfix(loss=total_loss.item())

def main():
    text = read_text_file(TEXT_DIR)

    # Split the text into pairs of sentences for SOP task
    sentences = [line for line in text if line.strip()]
    sentence_pairs = [(sentences[i], sentences[i + 1]) for i in range(0, len(sentences) - 1, 2)]

    # Shuffle sentence pairs
    shuffled_pairs = [(pair[1], pair[0]) if i % 2 != 0 else pair for i, pair in enumerate(sentence_pairs)]

    # Prepare inputs for MLM and SOP
    texts = [f"{pair[0]} {tokenizer.sep_token} {pair[1]}" for pair in shuffled_pairs]
    inputs = tokenizer(texts, return_tensors='pt', max_length=MAX_LENGTH, truncation=True, padding='max_length')
    inputs = create_mlm_sop_labels(inputs, mask_prob=MASK_PROB, sentence_pairs=shuffled_pairs)

    dataset = RechtDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_model(model, loader, device, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MLM_LOSS_WEIGHT, SOP_LOSS_WEIGHT)

    # Save model and tokenizer
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)


if __name__ == "__main__":
    main()