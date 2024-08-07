import os
import csv
import xml.etree.ElementTree as ET

root_dir = 'M:/BWB/Origineel'
output_dir = "M:/BWB/Output/Status"
file_name = "statusCheckall.csv"
root_folder = ['202210_BWB_1', '202210_BWB_2', '202210_BWB_3', '202210_BWB_4']


def manifest_lookup():
    for folder in root_folder:
        for root, dirs, files in os.walk(os.path.join(root_dir, folder)):
            for filename in files:
                if filename == 'manifest.xml':
                    data = []
                    manifest_path = os.path.join(root, filename)

                    data.append(os.path.basename(os.path.dirname(manifest_path)))
                    data.extend(manifest_reading(manifest_path))
                    write_to_csv(data)


def manifest_reading(manifest_path):
    data = []
    tree = ET.parse(manifest_path)
    root = tree.getroot()

    primary_metadata = root.find("./metadata")

    datum_inwerkingtreding = primary_metadata.find("datum_inwerkingtreding")
    if datum_inwerkingtreding is not None:
        data.append(datum_inwerkingtreding.text)
    else:
        data.append("ERROR")
        write_to_txt("ERROR AT: " + manifest_path + ". No attribute datum_inwerkingtreding.")

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
    fieldnames = ["id", "Datum inwerking", "Status", "Datum intrekking"]
    if not os.path.exists(os.path.join(output_dir, file_name)):
        with open(os.path.join(output_dir, file_name), "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"id": data[0], "Datum inwerking": data[1], "Status": data[2], "Datum intrekking": data[3]})
    else:
        with open(os.path.join(output_dir, file_name), "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({"id": data[0], "Datum inwerking": data[1], "Status": data[2], "Datum intrekking": data[3]})


def write_to_txt(text):
    error_dir = '../Data/logfile/statusCheck.txt'
    if not os.path.exists(error_dir):
        with open(error_dir, "w") as txtfile:
            txtfile.write(text + "\n")
    else:
        with open(error_dir, "a") as txtfile:
            txtfile.write(text + "\n")


if __name__ == "__main__":
    manifest_lookup()
    print("Code statusCheck executed")
