import os
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForPreTraining, AdamW

# Paths
text_dir = 'M:/BIT/datasetRandom.txt'
model_output_dir = './bertje-mlm-sop-model'

# Training arguments
pretrained_model_name = 'GroNLP/bert-base-dutch-cased'
tokenizer_name = 'bert-base-dutch'
max_length = 512
mask_prob = 0.15
batch_size = 8
epochs = 2
learning_rate = 5e-5
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
model = BertForPreTraining.from_pretrained(pretrained_model_name)


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


def create_mlm_sop_labels(inputs, mask_prob=0.15):
    """Create masked language model labels and SOP labels."""
    inputs['labels'] = inputs.input_ids.clone().detach()
    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < mask_prob) * (inputs.input_ids != tokenizer.cls_token_id) * \
               (inputs.input_ids != tokenizer.sep_token_id) * (inputs.input_ids != tokenizer.pad_token_id)
    selection = [torch.flatten(mask_arr[i].nonzero()).tolist() for i in range(inputs.input_ids.shape[0])]
    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = tokenizer.mask_token_id

    # Create SOP labels
    inputs['next_sentence_label'] = torch.LongTensor([1] * inputs.input_ids.shape[0])

    return inputs


def train_model(model, loader, device, epochs, lr):
    """Train the BERT model with the given data loader."""
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            next_sentence_label = batch['next_sentence_label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels,
                            next_sentence_label=next_sentence_label)
            loss = outputs.loss
            loss.backward()
            optim.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())


def main():
    text = read_text_file(text_dir)

    # Split the text into pairs of sentences for SOP task
    sentences = [line for line in text if line.strip() != '']
    sentence_pairs = [(sentences[i], sentences[i + 1]) for i in range(0, len(sentences) - 1, 2)]

    # Prepare inputs for MLM and SOP
    texts = [pair[0] + " " + tokenizer.sep_token + " " + pair[1] for pair in sentence_pairs]
    inputs = tokenizer(texts, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    inputs = create_mlm_sop_labels(inputs, mask_prob=mask_prob)

    dataset = RechtDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_model(model, loader, device, epochs, learning_rate)

    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)


if __name__ == "__main__":
    main()
