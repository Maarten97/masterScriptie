import csv
import os
import pandas as pd

# Global variables for filename and number of lines per file
BASE_FILENAME = "output_file"
LINES_PER_FILE = 2000000
OUTPUT_FOLDER = "output"
INPUT_FILE = "dataset.txt"
LIMIT_LARGE = False


def split_file(input_file):
    """
    Split the input file into smaller files, each containing LINES_PER_FILE lines.
    The output files will be named BASE_FILENAME_1.txt, BASE_FILENAME_2.txt, etc.,
    and will be saved in the OUTPUT_FOLDER.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    file_counter = 1  # Tracks the current output file number
    line_counter = 0  # Tracks the current line in the current output file

    # Open the input file and read line by line
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile:
        outfile = None  # Placeholder for the current output file
        for line in infile:
            if line_counter == 0:
                # Start a new output file
                if outfile:
                    outfile.close()  # Close the previous file
                output_path = os.path.join(OUTPUT_FOLDER, f"{BASE_FILENAME}_{file_counter}.txt")
                outfile = open(output_path, 'w', encoding='utf-8', errors='ignore')
                file_counter += 1

            # Write the current line to the current output file
            outfile.write(line)
            line_counter += 1

            # If the current file reaches the limit, reset the line counter
            if line_counter >= LINES_PER_FILE:
                line_counter = 0
                print(f'Finished generating TXT file # {file_counter - 1}')

        # Close the last output file if still open
        if outfile:
            outfile.close()


def count_lines(input_file, output_csv, output2_csv):
    """
    Parses the input file and writes the line number and word count to a CSV file.
    """
    line_counter = 0
    with (open(input_file, 'r', encoding='utf-8', errors='ignore') as infile,
          open(output_csv, 'w', newline='', encoding='utf-8') as csvfile,
          open(output2_csv, 'w', newline='', encoding='utf-8') as largefile):
        csv_writer = csv.writer(csvfile)
        csv_writer2 = csv.writer(largefile)
        # Write header for CSV
        csv_writer.writerow(['sentence_length'])
        csv_writer2.writerow(['line', 'sentence_length'])

        for line in infile:
            line_counter += 1
            # Count words by splitting the line based on whitespace
            word_count = len(line.split())
            # Write the line number and word count to the CSV
            if LIMIT_LARGE:
                if word_count != 0 and word_count < 101:
                    csv_writer.writerow([word_count])
                if word_count > 100:
                    csv_writer.writerow([101])
                    csv_writer2.writerow([line_counter, word_count])
            else:
                csv_writer.writerow([word_count])
        print(f'{line_counter} lines parsed')


def find_lines(input_file, row_number):
    line_counter = 0
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile:
        for line in infile:
            if line_counter == row_number:
                print(f'Printing row: {row_number}')
                print(line)
                break
            line_counter += 1


def print_lines(input_file, output_txt):
    line_counter = 0
    with (open(input_file, 'r', encoding='utf-8', errors='ignore') as infile,
          open(output_txt, 'w', newline='', encoding='utf-8') as txtfile):
        for line in infile:
            line_counter += 1
            word_count = len(line.split())
            if word_count > 500:
                txtfile.write(f'At {line_counter} lines following sentence of {word_count} words:' + "\n")
                txtfile.write(line + '\n')


def frequence_to_csv(csv_file_path, output_csv_file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Assuming the column containing sentence lengths is named 'sentence_length'
    # If the column has a different name, replace 'sentence_length' with the actual column name
    sentence_lengths = df['sentence_length']

    if LIMIT_LARGE:
        # Set all sentence lengths greater than 100 to 101
        sentence_lengths = sentence_lengths.apply(lambda x: 101 if x > 100 else x)

    # Count the frequency of each sentence length
    frequency = sentence_lengths.value_counts().sort_index()

    # Create a new DataFrame to store the frequency data
    frequency_df = pd.DataFrame({'Length': frequency.index, 'Frequency': frequency.values})

    # Save the frequency data to a new CSV file
    frequency_df.to_csv(output_csv_file_path, index=False)

    print(f"Frequency data has been saved to {output_csv_file_path}")


# Call the function with the path to your large txt file
if __name__ == '__main__':
    count_lines(INPUT_FILE, "logall.csv", 'loglarge102.csv')
    # find_lines(INPUT_FILE, 1824223)
    # print_lines(INPUT_FILE, 'lines.txt')
    # split_file(INPUT_FILE)
    # frequence_to_csv('log101.csv', 'frequence.csv')
