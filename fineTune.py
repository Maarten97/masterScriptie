## This script will load and preprocess my dataset.
import os
import xml.etree.ElementTree as ET


##Function to retrieve all XML files from Directory
def load_data_from_xml(data_directory):
    data = []
    for filename in os.listdir(data_directory):
        print(filename)
        if filename.endswith(".xml"):
            print("Currently loading XML file: " + filename)
            filepath = os.path.join(data_directory, filename)
            tree = ET.parse(filepath)
            root = tree.getroot()

            for item in root.findall("al"):
                text = item.find("al").text.strip()
                data.append(text)


            print("Finished loading XML file: " + filename)
    return data

if __name__ == "__main__":
    dataDirectory = "C:/Users/looij/PycharmProjects/masterScriptie/Data/Wetboeken"
    dataSet = load_data_from_xml(dataDirectory)
    print(dataSet)
