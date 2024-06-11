import csv, re, os

input_dir = ''
output_dir = ''


def clean_text(fulltext):
    newtext = []
    lines = fulltext.split('\n')
    for line in lines:

        # Here checks for sentences

        # Remove all parentheses, brackets, and other non-alphabetic/numeric characters except space and punctuation
        line = re.sub(r'[^\w\s,.;]', '', line)
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
            # If word does not contain normal characters or numbers with commas, remove it
            if not re.match(r'^[a-zA-Z0-9,]+$', word):
                continue
            # If word contains more than 3 numbers, remove the word
            if len(re.findall(r'\d', word)) > 3:
                continue
            # If word is two digits with one being a '.', remove
            if re.match(r'^\d\.\d$', word):
                continue
            # Remove the ' [] ' at the beginning and ending of every word
            if word.startswith('[') and word.endswith(']'):
                word = word[1:-1]

            newline.append(word)

        # Add the cleaned line to the new text if it's not empty
        if newline:
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


def test_text():
    testdir = ''
    with (open(testdir, mode='r', encoding='utf-8') as csvinput):
        reader = csv.DictReader(csvinput)
        for row in reader:
            court_text = row['text']
            court_text = clean_text(court_text)
            print(f'id: {row['id']}, court_text: {court_text}')


if __name__ == '__main__':
    open_text(input_dir)
