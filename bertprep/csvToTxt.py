import csv
import re

input_dir = 'C:/Users/looijengam/Documents/DatasetFinal/random.csv'
output_dir = 'output.txt'


def split_into_sentences(text):
    # Split text by sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences


def process_csv_to_txt(input_dir, output_dir):
    with open(input_dir, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';', quoting=csv.QUOTE_ALL)
        with open(output_dir, mode='w', encoding='utf-8') as txt_file:
            for row in csv_reader:
                if 'text' in row:
                    text = row['text']
                    sentences = split_into_sentences(text)
                    for sentence in sentences:
                        txt_file.write(sentence + '\n')
                    txt_file.write('\n\n')  # Add white lines between entries


# Example usage
if __name__ == '__main__':
    process_csv_to_txt(input_dir, output_dir)
