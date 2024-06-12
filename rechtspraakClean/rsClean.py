import csv, re, os

input_dir = 'C:/Programming/Dataset/Test'
output_dir = 'C:/Programming/Dataset/Test/testcsv.csv'


def clean_text(fulltext):
    newtext = []
    lines = fulltext.split('\n')
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

        end_of_line = False

        line = line.strip()

        # if line.endswith('.'):
        #     end_of_line = True

        print(f'END: {end_of_line}, LINE: {line}')
        # Here checks for sentences

        # Remove all parentheses, brackets, and other non-alphabetic/numeric characters except space and punctuation
        line = re.sub(r'[^\w\s,.:/;]', '', line)
        # Replace multiple whitespaces with a single whitespace
        line = re.sub(r'\s+', ' ', line)
        # Remove all citation characters ('')
        line = line.replace("'", "")
        # Replace semicolons with commas
        line = line.replace(';', ',')

        newline = []
        words = line.split()
        for word in words:
            # If word is fully written in capitals, convert to lower case
            if word.isupper():
                word = word.lower()
            if word == '-':
                continue
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

            if word == 'mr.' or word == 'mrs.':
                #remove this word and next words

            # If word is two digits with one being a '.', remove
            # if re.match(r'^\d\.\d$', word):
            #     continue
            # Remove the ' [] ' at the beginning and ending of every word
            if word.startswith('[') and word.endswith(']'):
                word = word[1:-1]
            if word == '.':
                newline[-1] += "."
            else:
                newline.append(word)

        print(f'NEWLINE: {newline}')
        # Add the cleaned line to the new text if it's not empty
        if newline:
            if end_of_line:
                newtext.append('. '.join(newline))
            else:
                newtext.append(' '.join(newline))
    return ' '.join(newtext)


def open_text(path):
    with (open(input_dir, mode='r', encoding='utf-8') as csvinput, open(output_dir, mode='w',
                                                                        encoding='utf-8') as outfile):
        reader = csv.DictReader(csvinput)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            court_text = row['text']
            court_text = clean_text(court_text)
            writer.writerow({'id': row['id'], 'text': court_text})


def test_text(path):
    with (open(path, mode='r', encoding='utf-8') as csvinput):
        reader = csv.DictReader(csvinput)
        for row in reader:
            court_text = row['full_text']
            if court_text:
                court_text = clean_text(court_text)
            print(f'id: {row['ecli']}, court_text: {court_text}')


if __name__ == '__main__':
    for csvfile in os.listdir(input_dir):
        newdir = os.path.join(input_dir, csvfile)
        print(newdir)
        if os.path.exists(newdir):
            # open_text(input_dir)
            test_text(newdir)
