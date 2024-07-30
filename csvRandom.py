import csv
import random

newdir = 'C:/Users/looijengam/Documents/DatasetFinal/random.csv'
fileone = 'C:/Users/looijengam/Documents/DatasetFinal/bwb.csv'
filetwo = 'C:/Users/looijengam/Documents/DatasetFinal/rechtspraak.csv'

csv.field_size_limit(10 ** 9)


# Function to read random rows from a CSV file
def read_random_rows(filename, num_rows):
    with open(filename, 'r', newline='', encoding="UTF-8") as file:
        reader = list(csv.DictReader(file, quotechar='"', delimiter=';'))
        return random.sample(reader, num_rows)


# Create a new CSV file with columns 'id' and 'text'
with open(newdir, 'w', newline='') as new_file:
    fieldnames = ['id', 'text']
    writer = csv.DictWriter(new_file, fieldnames=fieldnames, delimiter=';', quoting=csv.QUOTE_ALL)
    writer.writeheader()

    # Read and write 10 random rows from rech.csv
    rech_rows = read_random_rows(filetwo, 10)
    for row in rech_rows:
        writer.writerow({'id': row['id'], 'text': row['text']})

    # Read and write 10 random rows from bwb.csv
    bwb_rows = read_random_rows(fileone, 10)
    for row in bwb_rows:
        writer.writerow({'id': row['id'], 'text': row['text']})



print("CSV file created successfully!")
