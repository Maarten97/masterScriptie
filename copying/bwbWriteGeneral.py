import csv, os


def write_general(path, bwbid, text):
    input_file = bwbid + ".xml"
    input_dir = os.path.join(path, input_file)
    output_file = bwbid + ".csv"
    output_dir = os.path.join(path, output_file)
    fieldnames = ["id", "text"]
    if os.path.isdir(input_dir):
        if not os.path.isdir(os.path.join(path, output_file)):
            with open(output_dir, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({"id": bwbid, "text": text})
        else:
            with open(output_dir, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({"id": bwbid, "text": text})
