import os
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForPreTraining
from torch.optim import AdamW

# Paths
text_dir = 'C:/Users/looijengam/Documents/datasetRandom4.txt'
model_output_dir = './bertje-mlm-sop-model'

# Training arguments
pretrained_model_name = 'GroNLP/bert-base-dutch-cased'
tokenizer_name = 'GroNLP/bert-base-dutch-cased'
max_length = 512
mask_prob = 0.15
batch_size = 8
epochs = 2
learning_rate = 5e-5
weight_decay = 0.01
mlm_loss_weight = 1.5
sop_loss_weight = 0.5

# Check for process on GPU and count GPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f'Using device: {device}, number of GPUs: {num_gpus}')

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
model = BertForPreTraining.from_pretrained(pretrained_model_name)

# Use DataParallel if multiple GPUs are available
if num_gpus > 1:
    model = torch.nn.DataParallel(model)

# Move model to device
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
    try:
        with open(file_path, 'r', encoding=encoding) as fp:
            text = fp.read().split('\n')
        return text
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    except UnicodeDecodeError:
        raise UnicodeDecodeError(f"Cannot decode file {file_path} using {encoding} encoding.")


def create_mlm_sop_labels(inputs, mask_prob=0.15, sentence_pairs=None):
    """Create masked language model labels and SOP labels."""
    inputs['labels'] = inputs.input_ids.clone().detach()
    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < mask_prob) * (inputs.input_ids != tokenizer.cls_token_id) * \
               (inputs.input_ids != tokenizer.sep_token_id) * (inputs.input_ids != tokenizer.pad_token_id)
    selection = [torch.flatten(mask_arr[i].nonzero()).tolist() for i in range(inputs.input_ids.shape[0])]
    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = tokenizer.mask_token_id

    # Create SOP labels. 0 if in correct order, 1 if reversed
    labels = torch.LongTensor([0 if i % 2 == 0 else 1 for i in range(len(sentence_pairs))])
    inputs['next_sentence_label'] = labels

    return inputs


def train_model(model, loader, device, epochs, lr, weight_decay, mlm_loss_weight, sop_loss_weight):
    """Train the BERT model with weighted MLM and SOP losses."""
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            next_sentence_label = batch['next_sentence_label'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels,
                            next_sentence_label=next_sentence_label)

            # Extract the losses
            mlm_loss = outputs.prediction_loss  # Loss for MLM task
            sop_loss = outputs.seq_relationship_loss  # Loss for SOP task

            # Combine losses with custom weights
            total_loss = (mlm_loss_weight * mlm_loss) + (sop_loss_weight * sop_loss)

            # Backpropagation and optimization
            total_loss.backward()
            optim.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix_str(f'loss={total_loss.item()}')
def main():
    text = read_text_file(text_dir)

    # Split the text into pairs of sentences for SOP task
    sentences = [line for line in text if line.strip() != '']
    sentence_pairs = [(sentences[i], sentences[i + 1]) for i in range(0, len(sentences) - 1, 2)]

    #shuffle sentence pairs
    shuffled_pairs = [(pair[1], pair[0]) if i % 2 != 0 else pair for i, pair in enumerate(sentence_pairs)]

    # Prepare inputs for MLM and SOP
    texts = [pair[0] + " " + tokenizer.sep_token + " " + pair[1] for pair in shuffled_pairs]
    inputs = tokenizer(texts, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    inputs = create_mlm_sop_labels(inputs, mask_prob=mask_prob, sentence_pairs=shuffled_pairs)

    dataset = RechtDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_model(model, loader, device, epochs, learning_rate, weight_decay, mlm_loss_weight, sop_loss_weight)

    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)


if __name__ == "__main__":
    main()
