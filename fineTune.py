# This script will load and preprocess my dataset.
import csv
import os
import xml.etree.ElementTree as ET

# For testing, does not write to CSV file when set to False
csvWrite = True

# Often reoccuring strings in Dutch Lawbooks, removed for better accuracy
invalid_stings = [
    "bevat overgangsrecht m.b.t. deze wijziging",
    "bevatten overgangsrecht m.b.t. deze wijziging",
    "Tekstplaatsing met vernummering",
    "De gegevens van inwerkingtreding zijn ontleend aan de bron van aankondiging van de tekstplaatsing",
    "De datum van inwerkingtreding is ontleend aan de bron van aankondiging van de tekstplaatsing",
    "Abusievelijk is een wijzigingsopdracht geformuleerd die niet geheel juist is",
    "Vervalt behoudens voor zover het betreft de toepassing of overeenkomstige toepassing van deze artikelen "
]


# Function to retrieve all XML files from Directory
def load_data_from_xml(data_directory, csv_file):
    data = []
    empty_printer = 0
    line_counter = 0
    vervallen_counter = 0
    invalid_counter = 0
    word_counter = 0
    write_header = False

    # Create and Initialze CSV file
    if csvWrite:
        output_dir = os.path.dirname(csv_file)
        os.makedirs(output_dir, exist_ok=True)

    # Read from XML and write to CSV
    for filename in os.listdir(data_directory):
        if filename.endswith(".xml"):
            print("Currently loading XML file: " + filename)
            filepath = os.path.join(data_directory, filename)
            tree = ET.parse(filepath)
            root = tree.getroot()

            if csvWrite:
                with open(csv_file, "a", newline="") as csvfile:
                    fieldnames = ["bron", "text"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                    if not write_header:
                        writer.writeheader()
                        write_header = True

                    for item in root.findall(".//al"):
                        al_text = process_xml_text(item)
                        if al_text == "Vervallen" or al_text == "Vervallen.":
                            vervallen_counter += 1
                        elif al_text == "Invalid":
                            invalid_counter += 1
                        elif al_text is not None:
                            writer.writerow({"bron": os.path.splitext(filename)[0], "text": al_text})
                            word_counter += word_count(al_text)
                            line_counter += 1
                        else:
                            empty_printer += 1

            else:
                for item in root.findall(".//al"):
                    al_text = process_xml_text(item)
                    if al_text == "Vervallen" or al_text == "Vervallen.":
                        vervallen_counter += 1
                    elif al_text == "Invalid":
                        invalid_counter += 1
                    elif al_text is not None:
                        data.append(al_text)
                        word_counter += word_count(al_text)
                        line_counter += 1
                    else:
                        empty_printer += 1

            print("Finished loading XML file: " + filename)
    print(f"Aantal lege velden (errors): {empty_printer}")
    print(f"Aantal opgeslagen regels: {line_counter}")
    print(f"Aantal artikelen die reeds vervallen zijn: {vervallen_counter}")
    print(f"Aantal artikelen die manueel verwijderd zijn: {invalid_counter}")
    print(f"Aantal opgeslagen woorden: {word_counter}")
    return data


def process_xml_text(item):
    al_text = ''.join(item.itertext())

    # Removes all breaklines
    if "\n" in al_text:
        al_text = al_text.replace('\n', '')
    if "\r" in al_text:
        al_text = al_text.replace('\r', '')

    # Remove all extra whitespaces at the beginning or the end of the String
    al_text = al_text.strip()

    # Remove multiple whitespaces
    al_text = ' '.join(al_text.split())

    # Remove the ' " ' at the beginning or ending of every String
    if al_text.startswith('"'):
        al_text = al_text[1:]
    if al_text.endswith('"'):
        al_text = al_text[:-1]

    # Check if String is staring with '['
    if al_text.startswith('[') and al_text.endswith(']'):
        al_text = "Invalid"

    # Check if String is invalid according to definition
    for i in invalid_stings:
        if i in al_text:
            al_text = "Invalid"

    return al_text


def word_count(string):
    words_list = string.strip().split(" ")
    return len(words_list)


# INITIAL
if __name__ == "__main__":
    dataDirectory = "Data/Wetboeken"
    csvFile = "Data/Output/BurgelijkWetboekCSVall.csv"
    dataSet = load_data_from_xml(dataDirectory, csvFile)
    # if not csvWrite:
    #     print(dataSet)
