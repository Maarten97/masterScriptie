import csv, os
from datetime import datetime


def write_general(path, bwbid, text):
    output_dir = path
    if not path.endswith(".csv"):
        output_file = bwbid + ".csv"
        output_dir = os.path.join(path, output_file)
    fieldnames = ["id", "text"]
    if not os.path.exists(output_dir):
        with open(output_dir, "w", newline="", encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerow({"id": bwbid, "text": text})
    else:
        with open(output_dir, "a", newline="", encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';', quoting=csv.QUOTE_ALL)
            writer.writerow({"id": bwbid, "text": text})


def write_error(file, text):
    current_datetime = datetime.now()
    formatted_date_time = current_datetime.strftime("%Y%m%d_%H%M")
    output_file = formatted_date_time + ".txt"

    parent_dir = os.path.dirname(os.getcwd())
    output_path = os.path.join(parent_dir, "Data", "logfile", output_file)

    if not os.path.isfile(output_path):
        with open(output_path, "w") as txtfile:
            txtfile.write(file + ": " + text + "\n")
    else:
        with open(output_path, "a") as txtfile:
            txtfile.write(file + ": " + text + "\n")
