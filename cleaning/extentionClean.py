import os
import csv

root_dir = 'D:/BWB/Origineel'
output_dir = 'D:/BWB/Output/ExtentionCleaned.csv'
okay_extensions = ['.xml', '.xsd', '.wti']


def lookup():
    result = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            _, file_ext = os.path.splitext(filename)
            if file_ext.lower() not in okay_extensions:
                folder_name = extract_folder_name(dirpath)
                file_size = os.path.getsize(file_path)
                result.append((file_path, file_ext, folder_name, file_size))
    return result


def extract_folder_name(dirpath):
    components = dirpath.split(os.path.sep)  # Split the path into components
    for component in components:
        if component.startswith("BWBR") or component.startswith("BWBV") or component.startswith("BWBW"):
            return component  # Return the subfolder name if it matches the pattern
    return None


def csv_write():
    result_list = lookup()
    with open(output_dir, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["File Path", "Extension", "Folder Name", "File Size (bytes)"])
        for file_path, file_ext, folder_name, file_size in result_list:
            writer.writerow([file_path, file_ext, folder_name, file_size])
    print("List saved in CSV")


# INITIAL
if __name__ == "__main__":
    csv_write()
    print("Code executed")
