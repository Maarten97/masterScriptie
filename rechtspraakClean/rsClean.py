import csv, re, os, sys

input_dir = 'C:/Programming/Dataset/Test'
output_dir = 'C:/Programming/Dataset/TestOut/rechtspraak.csv'


def clean_text(fulltext):
    newtext = []
    lines = fulltext.split('\n')

    # Remove heading of each Court Decision
    start = False
    for line in lines:
        if not line:
            continue
        if not start:
            if line.startswith('1'):
                start = True
            else:
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

        line = line.strip()

        # if line.endswith('.'):
        #     end_of_line = True

        print(f'LINE: {line}')
        # Here checks for sentences

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

            # If word does not contain normal characters or numbers with commas, remove it
            # if not re.match(r'^[a-zA-Z0-9,]+$', word):
            #     continue
            # If word contains more than 3 numbers, remove the word
            if len(re.findall(r'\d', word)) > 3:
                if word.endswith('.'):
                    word = '.'
                else:
                    continue
            title_pattern = r"(?:Mr\.|Mrs\.)"
            if re.match(title_pattern, word, re.IGNORECASE):
                title_skipping, title_lowercase = True, True
                continue

            #remove this word and next words
            # If word is two digits with one being a '.', remove
            # if re.match(r'^\d\.\d$', word):
            #     continue
            # Remove the ' [] ' at the beginning and ending of every word
            if word.startswith('[') and word.endswith(']'):
                word = word[1:-1]
            if word == '.' and newline:
                newline[-1] += "."
            else:
                newline.append(word)

        print(f'NEWLINE: {newline}')

        # Add the cleaned line to the new text if it's not empty
        if newline:
            newtext.append(' '.join(newline))
    # Join all sentences together
    returnline = ' '.join(newtext)

    # Remove double points and comma's.
    returnline = re.sub(r'\.+', '.', returnline)
    returnline = re.sub(r',+', ',', returnline)
    return returnline


def open_text():
    csv.field_size_limit(10**9)
    header = False
    fieldnames = ["id", "text"]
    with open(output_dir, mode='w', encoding='utf-8') as outfile:

        for csvfile in os.listdir(input_dir):
            if csvfile.endswith('.csv'):
                newdir = os.path.join(input_dir, csvfile)

                with open(newdir, mode='r', encoding='utf-8') as csvinput:
                    reader = csv.DictReader(csvinput, quotechar='"', delimiter=';')
                    writer = csv.DictWriter(outfile, fieldnames=fieldnames, quotechar='"', delimiter=';')
                    if not header:
                        writer.writeheader()
                        header = True

                    for row in reader:
                        court_text = row['text']
                        if court_text:
                            court_text = clean_text(court_text)
                        writer.writerow({'id': row['id'], 'text': court_text})


def test_text():
    with (open(input_dir, mode='r', encoding='utf-8') as csvinput):
        reader = csv.DictReader(csvinput)
        for row in reader:
            court_text = row['full_text']
            if court_text:
                court_text = clean_text(court_text)
            print(f'id: {row['ecli']}, court_text: {court_text}')


if __name__ == '__main__':
    open_text()
    # test_text()