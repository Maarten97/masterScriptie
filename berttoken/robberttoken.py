import os
import random
import torch
import logging
from transformers import RobertaTokenizer, BatchEncoding

TEXT_DIR = './output'
TOKENIZED_CHUNKS_DIR = './tokenized_robbert'
LOCAL_MODEL_DIR = 'bertmodel/robbert'
MAX_LENGTH = 100
MASK_PROB = 0.15


def check_dir():
    # Ensure the input directory for dataset exists
    if not os.path.exists(TEXT_DIR):
        raise FileNotFoundError(f"The file {TEXT_DIR} does not exist.")
    if not os.path.exists(LOCAL_MODEL_DIR):
        raise FileNotFoundError(f"The file {LOCAL_MODEL_DIR} does not exist.")
    # Ensure the output directory for tokenized chunks exists
    os.makedirs(TOKENIZED_CHUNKS_DIR, exist_ok=True)


def read_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def tokenize_chunk(chunk, tokenizer, max_length=MAX_LENGTH, mask_prob=MASK_PROB):
    """Tokenize a chunk of text data."""
    mlm_inputs = BatchEncoding({'input_ids': [], 'attention_mask': [], 'labels': []})

    for sentence in chunk:
        if not sentence:
            continue

        # Tokenize sentence for MLM
        mlm_input = tokenizer(sentence, return_tensors='pt', max_length=max_length, truncation=True,
                              padding='max_length')

        mlm_inputs['input_ids'].append(mlm_input['input_ids'].squeeze(0))  # Remove extra dimension
        mlm_inputs['attention_mask'].append(mlm_input['attention_mask'].squeeze(0))

    mlm_inputs['input_ids'] = torch.stack(mlm_inputs['input_ids'])
    mlm_inputs['attention_mask'] = torch.stack(mlm_inputs['attention_mask'])

    # Create MLM labels: clone input_ids for labels
    mlm_inputs['labels'] = mlm_inputs.input_ids.detach().clone()
    rand = torch.rand(mlm_inputs.input_ids.shape)

    # Create mask array for MLM, avoiding special tokens
    mask_arr = (rand < mask_prob) * (mlm_inputs.input_ids != tokenizer.cls_token_id) * \
               (mlm_inputs.input_ids != tokenizer.sep_token_id) * \
               (mlm_inputs.input_ids != tokenizer.pad_token_id)

    # Apply mask to input_ids
    for i in range(mlm_inputs.input_ids.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        mlm_inputs.input_ids[i, selection] = tokenizer.mask_token_id

    logger.info("MLM Labels created for a Chunk")

    return mlm_inputs


def save_tokenized_data_pt(output_file, tokenized_data):
    """Save tokenized data directly to a .pt file."""
    # Save the tokenized tensor directly to .pt file
    torch.save(tokenized_data, output_file)
    logger.info(f"Saved tokenized data to {output_file}")


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(filename='tokenization_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()

    # Check if all dirs are correct
    check_dir()
    logger.info('Directories checked')

    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    logger.info('Initialized tokenizer')

    for files in os.listdir(TEXT_DIR):
        file_path = os.path.join(TEXT_DIR, files)

        chunk = read_file(file_path)
        logger.info('Loaded file {}'.format(files))

        tokenized_data = tokenize_chunk(chunk, tokenizer)
        logger.info('Tokenized data of file {}'.format(files))

        output_dir = os.path.join(TOKENIZED_CHUNKS_DIR, os.path.splitext(files)[0] + '.pt')
        save_tokenized_data_pt(os.path.join(output_dir), tokenized_data)
        logger.info(f'Saved tokenized data to {output_dir}')
    logger.info('Done')



