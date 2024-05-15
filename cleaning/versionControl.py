import datetime
import os
import shutil

root_dir = 'D:/BWB/Subset'
output_dir = 'D:/BWB/Output/VersionControl'
root_folder = ['202210_BWB_1', '202210_BWB_2', '202210_BWB_3', '202210_BWB_4']


def folder_lookup():
    for folder in root_folder:
        input_path = os.path.join(root_dir, folder)
        for bwbid in os.listdir(input_path):
            if os.path.isdir(os.path.join(input_path, bwbid)):
                bwb_path = os.path.join(input_path, bwbid)
                output_path = os.path.join(output_dir, bwbid)

                date = newest_date(bwb_path)
                date_path = os.path.join(bwb_path, date)
                copy_xml(date_path, output_path)


def newest_date(input_path):
    folder_names = []
    for folder in os.listdir(input_path):
        if os.path.isdir(os.path.join(input_path, folder)):
            # Split the folder name by underscores
            parts = folder.split("_")
            if len(parts) >= 2:
                try:
                    # Extract the date part (first element)
                    date_str = parts[0]
                    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")

                    # Extract the version part (last element)
                    version = int(parts[-1])

                    folder_names.append((date_obj, version, folder))
                except ValueError:
                    # Handle invalid date format (if any)
                    print("File with bwbid " + input_path + " skipped.")
                    pass
    sorted_folders = sorted(folder_names, key=lambda x: (x[0], x[1]), reverse=True)
    if sorted_folders:
        newest_folder = sorted_folders[0][2]
        return newest_folder
    else:
        print("No date found in folder " + input_path)
        exit(-2)


def copy_xml(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for root, _, files in os.walk(input_path):
        for filename in files:
            if filename.lower().endswith(".xml"):
                source_path = os.path.join(root, filename)
                destination_path = os.path.join(output_path, filename)
                shutil.copy2(source_path, destination_path)
                print(f"Copied {source_path} to {destination_path}")


# INITIAL
if __name__ == "__main__":
    folder_lookup()
    print("Code versionControl executed")