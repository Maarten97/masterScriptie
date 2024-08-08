import csv
import os
import re


def clean_text(fulltext):
    newtext = []
    lines = fulltext.split('\n')

    # If line is empty, it is skipped
    for line in lines:
        if not line:
            continue

        # Verwijder hoofdstuknummering
        if line[0].isdigit():
            while line and (line[0].isdigit() or line[0].isspace() or line[0] == '.'):
                line = line[1:]
            if line:
                if not line.endswith('.'):
                    line += '.'
            else:
                continue

        # Removes any whitespaces at beginning or end of line
        line = line.strip()

        # Remove words within square brackets
        line = re.sub(r'\[.*?]', '', line)
        # Remove all parentheses, brackets, and other non-alphabetic/numeric characters except space and punctuation
        line = re.sub(r'[^\w\s,.:/;]', '', line)
        # Replace multiple whitespaces with a single whitespace
        line = re.sub(r'\s+', ' ', line)
        # Remove all citation characters ('')
        line = line.replace("'", "")
        # Replace semicolons with commas
        line = line.replace(';', ',')

        title_skipping = False
        title_lowercase = False
        newline = []
        line = line.strip()
        words = line.split()
        for word in words:

            # Delete all mr.
            if title_skipping:
                if word.count('.') > 1:
                    continue
                elif re.match(r"^[A-Za-z]\.", word):
                    continue

                elif title_lowercase:
                    if word[0].isupper():
                        title_lowercase = False
                    continue
                else:
                    if word[0].isupper():
                        continue

            # If word is fully written in capitals, convert to lower case
            if word.isupper():
                word = word.lower()

            if len(word) == 1:
                continue

            if re.search(r'\b(b\.v\.|n\.v\.|c\.s\.|v\.o\.f\.)\b,', word, re.IGNORECASE):
                continue

            if word == ',.':
                continue

            # If word contains more than 3 numbers, remove the word
            if len(re.findall(r'\d', word)) > 3:
                if word.endswith('.'):
                    word = '.'
                else:
                    continue
            title_pattern = r'\b(Mr\.|Mrs\.|Dr\.|Drs\.)'
            if re.match(title_pattern, word, re.IGNORECASE):
                title_skipping, title_lowercase = True, True
                continue

            if word == 'art.':
                word = 'artikel'
            if word == 'nr.':
                word = 'nummer'
            if word == 'p.':
                word = 'paragraaf'
            if word == 'd.d.':
                word = 'de dato'

            # If word contains more than one dot, skip the word.
            if word.count('.') > 1:
                continue

            # Remove the ' [] ' at the beginning and ending of every word
            if word.startswith('[') and word.endswith(']'):
                word = word[1:-1]
            if word == '.' and newline:
                newline[-1] += "."
            else:
                newline.append(word)

        # Add the cleaned line to the new text if it's not empty
        if newline and len(newline) > 1:
            newtext.append(' '.join(newline))
    # Join all sentences together
    returnline = ' '.join(newtext)

    # Remove multiple points and comma's.
    returnline = re.sub(r'\.+', '.', returnline)
    returnline = re.sub(r',+', ',', returnline)

    # If starts with space, remove space
    if returnline and returnline[0] == ' ':
        returnline = returnline[1:]
    # Find the first dot and remove everything in front of it.

    dot_index = returnline.find('.')
    if dot_index != -1:
        returnline = returnline[dot_index + 1:]

    if returnline.startswith('Datum uitspraak'):
        dot_index = returnline.find('.')
        if dot_index != -1:
            returnline = returnline[dot_index + 1:]

    if returnline.startswith('Partijen'):
        dot_index = returnline.find('.')
        if dot_index != -1:
            returnline = returnline[dot_index + 1:]

    return returnline


def open_text(input_dir, output_dir):
    csv.field_size_limit(10 ** 9)
    fieldnames = ["id", "text"]
    with open(output_dir, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, quotechar='"', delimiter=';')
        writer.writeheader()

        for csvfile in os.listdir(input_dir):
            if csvfile.endswith('.csv'):
                newdir = os.path.join(input_dir, csvfile)
                print(f'Now processing {newdir}')

                with open(newdir, mode='r', encoding='utf-8') as csvinput:
                    reader = csv.DictReader(csvinput, quotechar='"', delimiter=';')

                    for row in reader:
                        court_text = row['text']
                        if court_text:
                            court_text = clean_text(court_text)
                            if len(court_text) == 0:
                                print(f'No court text cleaned for {row['id']}')
                            else:
                                writer.writerow({'id': row['id'], 'text': court_text})
                        else:
                            print(f'No court text found for {row['id']}')


def test_text(input_dir):
    csv.field_size_limit(10 ** 9)
    for csvfile in os.listdir(input_dir):
        if csvfile.endswith('.csv'):
            newdir = os.path.join(input_dir, csvfile)
            with (open(newdir, mode='r', encoding='utf-8') as csvinput):
                reader = csv.DictReader(csvinput, quotechar='"', delimiter=';')
                for row in reader:
                    court_text = row['text']
                    if court_text:
                        court_text = clean_text(court_text)
                        print(f'id: {row['id']}, court_text: {court_text}')
                    else:
                        print(row[0:20])


def main(inputdr, output):
    open_text(inputdr, output)
    # test_text(inputdr)


if __name__ == '__main__':
    input_dir = 'C:/Programming/Dataset/Test'
    output_dir = 'C:/Programming/Dataset/TestOut/rechtspraaktst.csv'
    main(input_dir, output_dir)
