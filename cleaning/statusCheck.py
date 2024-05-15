import os
import csv
import xml.etree.ElementTree as ET

root_dir = 'D:/BWB/Subset'
output_dir = 'D:/BWB/Output/statusCheck.csv'
root_folder = ['202210_BWB_1', '202210_BWB_2', '202210_BWB_3', '202210_BWB_4']

write_header = False


def manifest_lookup():
    for folder in root_folder:
        for root, dirs, files in os.walk(os.path.join(root_dir, folder)):
            for filename in files:
                if filename == 'manifest.xml':
                    data = []
                    manifest_path = os.path.join(root, filename)

                    data.append(os.path.basename(os.path.dirname(manifest_path)))
                    data.extend(manifest_reading(manifest_path))
                    print(data)
                    write_to_csv(data)


def manifest_reading(manifest_path):
    data = []
    tree = ET.parse(manifest_path)
    root = tree.getroot()

    primary_metadata = root.find("./metadata")
    data.append(primary_metadata.find("datum_inwerkingtreding").text)
    # status = ""
    datum_intrekking = primary_metadata.find("datum_intrekking")
    if datum_intrekking is not None:
        datum_intrekking = datum_intrekking.text
        status = "Vervallen"
    else:
        datum_intrekking = "N/A"
        status = "Geldend"
    data.append(status)
    data.append(datum_intrekking)
    return data


def write_to_csv(data):
    output_file = os.path.dirname(output_dir)
    os.makedirs(output_file, exist_ok=True)

    with open(output_file, "a", newline="") as csvfile:
        fieldnames = ["id", "Datum inwerking", "Status", "Datum intrekking"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)

        if not write_header:
            writer.writeheader()
            modify_bool()

        writer.writerows({"id": data[0], "Datum inwerking": data[1], "Status": data[2], "Datum intrekking": data[3]})


def modify_bool():
    global write_header
    write_header = True


if __name__ == "__main__":
    manifest_lookup()
    print("Code statusCheck executed")
