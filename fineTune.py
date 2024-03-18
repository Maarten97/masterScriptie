## This script will load and preprocess my dataset.
import csv
import os
import xml.etree.ElementTree as ET

csvWrite = False


##Function to retrieve all XML files from Directory
def load_data_from_xml(data_directory, csvFile):
    data = []
    emptyPrinter = 0
    lineCounter = 0
    vervallenCounter = 0

    ##Create and Initialze CSV file
    if csvWrite:
        output_dir = os.path.dirname(csvFile)
        os.makedirs(output_dir, exist_ok=True)

    ##Read from XML and write to CSV
    for filename in os.listdir(data_directory):
        print(filename)
        if filename.endswith(".xml"):
            print("Currently loading XML file: " + filename)
            filepath = os.path.join(data_directory, filename)
            tree = ET.parse(filepath)
            root = tree.getroot()

            if csvWrite:
                with open(csvFile, "a", newline="") as csvfile:
                    fieldnames = ["bron", "text"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                    writer.writeheader()

                    for item in root.findall(".//al"):
                        al_text = process_xml_text(item)
                        if al_text is not None and al_text is not "Vervallen":
                            writer.writerow({"bron": os.path.splitext(filename)[0], "text": al_text})
                            lineCounter += 1
                        elif al_text is "Vervallen":
                            print("Artikel is vervallen")
                            vervallenCounter += 1
                        else:
                            print("al_text looks to be empty")
                            emptyPrinter += 1

            else:
                for item in root.findall(".//al"):
                    al_text = process_xml_text(item)
                    if al_text is not None and al_text is not "Vervallen":
                        data.append(al_text)
                        lineCounter += 1
                    elif al_text is "Vervallen":
                        print("Artikel is vervallen")
                        vervallenCounter += 1
                    else:
                        print("al_text looks to be empty")
                        emptyPrinter += 1

            print("Finished loading XML file: " + filename)
    print(f"EmptyPrinter: {emptyPrinter}")
    print(f"LineCounter: {lineCounter}")
    print(f"vervallenCounter: {vervallenCounter}")
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

    ##BUGFIXING
    if "Vervallen" in al_text:
        print("test1")
    return al_text

### INITIAL
if __name__ == "__main__":
    dataDirectory = "C:/Users/looij/PycharmProjects/masterScriptie/Data/Wetboeken"
    csvFile = "Data/Output/BurgelijkWetboekCSV.csv"
    dataSet = load_data_from_xml(dataDirectory, csvFile)
    # if not csvWrite:
        # print(dataSet)