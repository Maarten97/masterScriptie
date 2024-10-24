import os
import torch
import logging
import mmap
from transformers import BertTokenizer
from multiprocessing import Pool, cpu_count, get_context

TEXT_DIR = '../berttoken/datasetTest.txt'
TOKENIZED_CHUNKS_DIR = 'output'
MERGED_DATA_PATH = './merged_tokenized_data.pt'
LOCAL_MODEL_DIR = '../berttoken/mbert'
CHUNK_SIZE = 5000
MAX_LENGTH = 256

# Ensure the output directory for tokenized chunks exists
os.makedirs(TOKENIZED_CHUNKS_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(filename='../berttoken/tokenization_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()


def tokenize_chunk(chunk, tokenizer, max_length=MAX_LENGTH):
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
        input_chunks.append(chunk)

    logger.info(f"File split into {len(input_chunks)} chunks for parallel processing")

    # Use multiprocessing Pool to tokenize in parallel with 'spawn' context
    logger.info(f"Creating Pool with cores = {cpu_count()}")
    with get_context("spawn").Pool(cpu_count()) as pool:
        for i, chunk in enumerate(input_chunks):
            tokenized_chunk = pool.starmap(tokenize_chunk, [(chunk, tokenizer)])

            # Save each tokenized chunk directly to .pt file
            chunk_output_path = os.path.join(TOKENIZED_CHUNKS_DIR, f"tokenized_chunk_{i}.pt")
            save_tokenized_data_pt(chunk_output_path, tokenized_chunk[0])
            logger.info(f"Saved tokenized chunk {i} to {chunk_output_path}")

            # Remove the processed chunk to free up memory
            del input_chunks[i]
            tokenized_chunk = None  # Also free the tokenized chunk
            logger.info(f"Tokenized chunk {i} done")

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
