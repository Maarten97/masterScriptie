import os
import torch
import logging
import mmap
from transformers import BertTokenizer
from multiprocessing import Pool, cpu_count

TEXT_DIR = './dataset.txt'
TOKENIZED_DATA_PATH = './tokenized_dataset.pt'
LOCAL_MODEL_DIR = './mbert'
CHUNK_SIZE = 100000

# Set up logging
logging.basicConfig(filename='tokenization_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()


def tokenize_chunk(chunk, tokenizer, max_length=128):
    """Tokenize a chunk of text data."""
    tokenized_output = tokenizer(chunk, return_tensors='pt', max_length=max_length, truncation=True,
                                 padding='max_length')
    return tokenized_output


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


def parallel_tokenization_with_mmap(file_path, tokenizer, chunk_size=CHUNK_SIZE):
    """Tokenize the text data in parallel using memory-mapped file and multiple CPU cores."""
    mmapped_file = memory_map_file(file_path)
    logger.info(f"Memory-mapped {file_path} for tokenization")

    input_chunks = []
    while True:
        chunk = read_chunk_from_memory(mmapped_file, chunk_size)
        if not chunk:
            break
        input_chunks.append(chunk)

    logger.info(f"File split into {len(input_chunks)} chunks for parallel processing")

    # Use multiprocessing Pool to tokenize in parallel
    with Pool(cpu_count()) as pool:
        tokenized_chunks = pool.starmap(tokenize_chunk, [(chunk, tokenizer) for chunk in input_chunks])

    # Combine tokenized outputs
    combined_tokenized_data = {key: torch.cat([chunk[key] for chunk in tokenized_chunks], dim=0) for key in
                               tokenized_chunks[0]}

    logger.info(f"Tokenization complete. {len(combined_tokenized_data['input_ids'])} samples tokenized.")
    return combined_tokenized_data


def main():
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    logger.info('Initialized tokenizer')

    # Check if tokenized data exists
    if os.path.exists(TOKENIZED_DATA_PATH):
        logger.info("Loading tokenized dataset from disk")
        tokenized_data = torch.load(TOKENIZED_DATA_PATH)
    else:
        logger.info("Tokenized dataset not found, processing from memory-mapped text")
        tokenized_data = parallel_tokenization_with_mmap(TEXT_DIR, tokenizer)

        # Save the tokenized data to disk
        torch.save(tokenized_data, TOKENIZED_DATA_PATH)
        logger.info(f"Tokenized dataset saved to {TOKENIZED_DATA_PATH}")


if __name__ == '__main__':
    main()
