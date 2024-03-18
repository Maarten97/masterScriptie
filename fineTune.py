## This script will load and preprocess my dataset.
import os
import xml.etree.ElementTree as ET


##Function to retrieve all XML files from Directory
def load_data_from_xml(data_directory):
    data = []
    emptyPrinter = 0
    lineCounter = 0
    for filename in os.listdir(data_directory):
        print(filename)
        if filename.endswith(".xml"):
            print("Currently loading XML file: " + filename)
            filepath = os.path.join(data_directory, filename)
            tree = ET.parse(filepath)
            root = tree.getroot()

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
    dataSet = load_data_from_xml(dataDirectory)
    # print(dataSet)
