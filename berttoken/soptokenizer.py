import os
import random
import torch
import logging
import mmap
from transformers import BertTokenizer, BatchEncoding
from multiprocessing import Pool, cpu_count, get_context

TEXT_DIR = './dataset.txt'
TOKENIZED_CHUNKS_DIR = './tokenized_chunks'
LOCAL_MODEL_DIR = './mbert'
CHUNK_SIZE = 500000
MAX_LENGTH = 256
MASK_PROB = 0.15
SKIP = 0
CORES = 16

# Ensure the output directory for tokenized chunks exists
os.makedirs(TOKENIZED_CHUNKS_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(filename='tokenization_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()


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
            random_i = random.randint(0, len(chunk) - 1)
            sop_pair = [f'{first_sentence} {tokenizer.sep_token} {chunk[random_i]}']
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


def memory_map_file(file_path):
    """Map the file into memory using mmap."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r+b') as f:
        # Memory-map the file, size 0 means whole file
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    return mmapped_file


def read_chunk_from_memory(mmapped_file, chunk_size):
    """Read a chunk of data from the memory-mapped file."""
    lines = []
    line = mmapped_file.readline().decode('utf-8')
    while line and len(lines) < chunk_size:
        lines.append(line.strip())
        line = mmapped_file.readline().decode('utf-8')
    return lines


def save_tokenized_data_pt(output_file, tokenized_data):
    """Save tokenized data directly to a .pt file."""
    # Save the tokenized tensor directly to .pt file
    torch.save(tokenized_data, output_file)
    logger.info(f"Saved tokenized data to {output_file}")


def parallel_tokenization_with_mmap(file_path, tokenizer, chunk_size=CHUNK_SIZE):
    """Tokenize the text data in parallel using memory-mapped file and multiple CPU cores."""
    mmapped_file = memory_map_file(file_path)
    logger.info(f"Memory-mapped {file_path} for tokenization")

    input_chunks = []
    while True:
        chunk = read_chunk_from_memory(mmapped_file, chunk_size)
        if not chunk:
            break
        input_chunks.append((chunk, tokenizer))

    logger.info(f"File split into {len(input_chunks)} chunks for parallel processing")

    # Removing first Chunks due to previous error
    if SKIP != 0:
        del input_chunks[:SKIP]
        logger.info(f"Now {len(input_chunks)} chunks after removing {SKIP} chuncks")

    if not os.path.exists(TOKENIZED_CHUNKS_DIR):
        os.makedirs(TOKENIZED_CHUNKS_DIR)

    # Use multiprocessing Pool to tokenize in parallel with 'spawn' context
    logger.info(f"Creating Pool with cores = {CORES}, Official count: {cpu_count()}")

    with get_context("spawn").Pool(CORES) as pool:
        tokenized_chunk = pool.starmap(tokenize_chunk, input_chunks)
        for i, tokenized_chunk in enumerate(tokenized_chunk):

            # Save each tokenized chunk directly to .pt file
            chunk_output_path = os.path.join(TOKENIZED_CHUNKS_DIR, f"tokenized_chunk_{i}.pt")
            logger.info(f"Attempting to save tokenized chunk {i} to {chunk_output_path}")
            save_tokenized_data_pt(chunk_output_path, tokenized_chunk[i])
            logger.info(f"Saved tokenized chunk {i} to {chunk_output_path}")

    # Close the memory-mapped file after use to release resources
    mmapped_file.close()


def main():
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    logger.info('Initialized tokenizer')

    # Start the tokenization process with memory-mapped input
    logger.info("Processing from memory-mapped text")
    parallel_tokenization_with_mmap(TEXT_DIR, tokenizer)

    logger.info("Tokenization completed")


if __name__ == '__main__':
    main()
