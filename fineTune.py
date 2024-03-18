## This script will load and preprocess my dataset.
import csv
import os
import xml.etree.ElementTree as ET

csvWrite = True


##Function to retrieve all XML files from Directory
def load_data_from_xml(data_directory, csvFile):
    data = []
    emptyPrinter = 0
    lineCounter = 0

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
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    for item in root.findall(".//al"):
                        al_text = item.text
                        if al_text is not None:
                                writer.writerow({"bron": os.path.splitext(filename)[0], "text": al_text.strip()})
                                lineCounter += 1
                        else:
                            print("al_text looks to be empty")
                            emptyPrinter += 1

            else:
                for item in root.findall(".//al"):
                    al_text = item.text
                    if al_text is not None:
                            data.append(al_text.strip())
                            lineCounter += 1
                    else:
                        print("al_text looks to be empty")
                        emptyPrinter += 1

            print("Finished loading XML file: " + filename)
    print(f"EmptyPrinter: {emptyPrinter}")
    print(f"LineCounter: {lineCounter}")
    return data

if __name__ == "__main__":
    dataDirectory = "C:/Users/looij/PycharmProjects/masterScriptie/Data/Wetboeken"
    csvFile = "Data/Output/BurgelijkWetboekCSV.csv"
    dataSet = load_data_from_xml(dataDirectory, csvFile)
    # print(dataSet)
