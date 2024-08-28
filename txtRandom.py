import os


def save_first_lines(input_file_path, num_lines):
    first_lines = []

    # Read the first 10,000 lines
    with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for _ in range(num_lines):
            line = file.readline()
            if not line:
                break
            first_lines.append(line)
    print(f"Successfully extracted {num_lines} lines.")
    return first_lines


def skip_and_extract_lines(input_file, skip_lines, num_lines):
    last_lines = []
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile:
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
            last_lines.append(line)
    print(f"Successfully skipped {skip_lines} lines and extracted {num_lines} lines.")
    return last_lines


def print_lines(input_file, output_file, skip_lines, num_lines):
    first_line = save_first_lines(input_file, num_lines)
    last_line = skip_and_extract_lines(input_file, skip_lines, num_lines)
    all_lines = first_line + last_line

    with open(output_file, 'w', encoding='utf-8', errors='ignore') as outfile:
        for line in all_lines:
            outfile.write(line)


if __name__ == '__main__':
    input_file = 'E:/BIT/dataset.txt'
    output_file = 'E:/BIT/datasetTest.txt'
    skip_lines = 5000000  # Number of lines to skip
    num_lines = 10000  # Number of lines for both first and last

    print_lines(input_file, output_file, skip_lines, num_lines)
