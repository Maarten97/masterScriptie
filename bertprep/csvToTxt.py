import csv
import os
import re


# Code to check the mega full txt file. With great inspiration from the World Wide Web.

def merge_csv(inputdirs, tempdir):
    csv.field_size_limit(10**9)
    fieldnames = ["id", "text"]
    with open(tempdir, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, quotechar='"', delimiter=';')
        writer.writeheader()
        for inputdir in inputdirs:
            if inputdir.endswith('.csv'):
                with open(inputdir, mode='r', encoding='utf-8') as csvinput:
                    reader = csv.DictReader(csvinput, quotechar='"', delimiter=';')
                    for row in reader:
                        writer.writerow({'id': row['id'], 'text': row['text']})
            else:
                raise TypeError("Input type not supported, non .csv file")


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
                        check = sentence.strip().split()
                        if len(check) > 1:
                            txt_file.write(sentence + '\n')
                    if txt_file:
                        txt_file.write('\n')  # Add white lines between entries


def main(inputs, outputs, tempdir):
    if isinstance(inputs, list):
        merge_csv(inputs, tempdir)
        process_csv_to_txt(input_dir=tempdir, output_dir=outputs)
    elif isinstance(inputs, str):
        process_csv_to_txt(input_dir=inputs, output_dir=outputs)
    else:
        raise TypeError("Inputs should be a string or a list of strings.")


# Example usage
if __name__ == '__main__':
    input_dir = 'C:/Users/looijengam/Documents/DatasetFinal/random.csv'
    output_dir = 'output.txt'
    main(inputs=input_dir, outputs=output_dir, tempdir=None)

