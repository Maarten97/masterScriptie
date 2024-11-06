import os
import random
import torch
import logging
from transformers import BertTokenizer, BatchEncoding

TEXT_DIR = './output'
# TEXT_DIR = './tokenized_chunksnew'
TOKENIZED_CHUNKS_DIR = './tokenized_mbert'
# LOCAL_MODEL_DIR = './mbert'
LOCAL_MODEL_DIR = 'C:/Users/looij/PycharmProjects/masterScriptie/bertmodel/mbert'
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
    sop_inputs = BatchEncoding({'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'sop_labels': []})

    for i in range(0, len(chunk) - 1, 2):
        if not chunk[i]:
            i = i + 1

        first_sentence = chunk[i]
        if i + 1 <= len(chunk):
            second_sentence = chunk[i + 1]
        else:
            second_sentence = ''

        # SOP Labelling
        if random.random() > 0.5 or not second_sentence:
            label = 0
            sop_pair = [f'{second_sentence} {tokenizer.sep_token} {first_sentence}']
        else:
            label = 1
            sop_pair = [f'{first_sentence} {tokenizer.sep_token} {second_sentence}']

        sop_input = tokenizer(sop_pair, return_tensors='pt', max_length=max_length, truncation=True,
                              padding='max_length')

        sop_inputs['input_ids'].append(sop_input['input_ids'].squeeze(0))  # Remove extra dimension
        sop_inputs['attention_mask'].append(sop_input['attention_mask'].squeeze(0))
        sop_inputs['token_type_ids'].append(sop_input['token_type_ids'].squeeze(0))
        sop_inputs['sop_labels'].append(torch.tensor(label))  # Convert label to tensor

    sop_inputs['input_ids'] = torch.stack(sop_inputs['input_ids'])
    sop_inputs['attention_mask'] = torch.stack(sop_inputs['attention_mask'])
    sop_inputs['token_type_ids'] = torch.stack(sop_inputs['token_type_ids'])
    sop_inputs['sop_labels'] = torch.stack(sop_inputs['sop_labels'])  # Fixed this line

    # Logging info
    logger.info("SOP Labels created for a Chunk")

    # MLM Labeling: create a copy of input_ids for MLM labels
    sop_inputs['labels'] = sop_inputs.input_ids.detach().clone()
    rand = torch.rand(sop_inputs.input_ids.shape)

    # Create mask array for MLM
    mask_arr = (rand < mask_prob) * (sop_inputs.input_ids != tokenizer.cls_token_id) * \
               (sop_inputs.input_ids != tokenizer.sep_token_id) * \
               (sop_inputs.input_ids != tokenizer.pad_token_id)

    # Apply mask
    for i in range(sop_inputs.input_ids.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        sop_inputs.input_ids[i, selection] = tokenizer.mask_token_id
    logger.info("MLM Labels created for a Chunk")

    return sop_inputs


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
    tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_DIR)
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



