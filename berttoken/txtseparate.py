import os

# Global variables for filename and number of lines per file
BASE_FILENAME = "output_file"
LINES_PER_FILE = 4000
OUTPUT_FOLDER = "output"


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

        # Close the last output file if still open
        if outfile:
            outfile.close()


# Call the function with the path to your large txt file
if __name__ == '__main__':
    split_file("datasetTest.txt")
