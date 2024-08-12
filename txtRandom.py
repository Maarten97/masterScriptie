import os

# Usage
input_file = 'C:/Users/looijengam/Documents/Final/dataset.txt'
output_file = 'C:/Users/looijengam/Documents/datasetRandom2.txt'
skip_lines = 5000000  # Number of lines to skip
num_lines = 10000  # Number of lines to extract after skippiif __name__ == '__main__':

def save_first_and_last_10000_lines(input_file_path, output_file_path):
    first_lines = []
    last_lines = []

    # Read the first 10,000 lines
    with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for _ in range(10000):
            line = file.readline()
            if not line:
                break
            first_lines.append(line)

    # Read the last 10,000 lines
    with open(input_file_path, 'rb') as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        buffer_size = 1024
        buffer = bytearray()
        lines = 0

        while lines < 10000 and file_size > 0:
            to_read = min(buffer_size, file_size)
            file.seek(file_size - to_read)
            buffer = file.read(to_read) + buffer
            lines = buffer.count(b'\n')
            file_size -= to_read

        last_lines = buffer.split(b'\n')[-10000:]

    # Decode last_lines from bytes to str and ignore errors
    last_lines = [line.decode('utf-8', errors='ignore') + '\n' for line in last_lines]

    # Write the lines to a new output file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.writelines(first_lines)
        output_file.writelines(last_lines)


def skip_and_extract_lines(input_file, output_file, skip_lines, num_lines):
    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            # Skip the first `skip_lines` lines
            for _ in range(skip_lines):
                line = infile.readline()
                if not line:
                    print(f"File ended before skipping {skip_lines} lines.")
                    return

            # Extract the next `num_lines` lines
            for _ in range(num_lines):
                line = infile.readline()
                if not line:
                    break  # Stop if we reach the end of the file
                outfile.write(line)

        print(f"Successfully skipped {skip_lines} lines and extracted {num_lines} lines to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    skip_and_extract_lines(input_file, output_file, skip_lines, num_lines)
    # save_first_and_last_10000_lines(input_file, output_file)
