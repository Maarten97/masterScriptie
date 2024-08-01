import os
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, AdamW

# Set the path to the text file
text_dir = 'output.txt'

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')


class RechtDataset(torch.utils.data.Dataset):
    """Custom Dataset for loading and encoding text data."""

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


def create_mlm_labels(inputs, mask_prob=0.15):
    """Create masked language model labels."""
    inputs['labels'] = inputs.input_ids.clone().detach()
    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < mask_prob) * (inputs.input_ids != tokenizer.cls_token_id) * \
               (inputs.input_ids != tokenizer.sep_token_id) * (inputs.input_ids != tokenizer.pad_token_id)
    selection = [torch.flatten(mask_arr[i].nonzero()).tolist() for i in range(inputs.input_ids.shape[0])]
    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = tokenizer.mask_token_id
    return inputs


def train_model(model, loader, device, epochs=2, lr=5e-5):
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
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())


def main():
    text = read_text_file(text_dir)
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    inputs = create_mlm_labels(inputs)
    dataset = RechtDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_model(model, loader, device, epochs=2, lr=5e-5)
    model.save_pretrained('./bert-mlm-model')
    tokenizer.save_pretrained('./bert-mlm-model')


if __name__ == "__main__":
    main()
