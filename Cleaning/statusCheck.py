import os
import csv

root_dir = 'D:/BWB/Subset'
output_dir = 'D:/BWB/Output/VersionControl'
root_folder = ['202210_BWB_1', '202210_BWB_2', '202210_BWB_3', '202210_BWB_4']

def manifest_lookup():
    for folder in root_folder:
        for root, dirs, files in os.walk(os.path.join(root_dir, folder)):
            for filename in files:
                if filename == 'manifest.xml':
                    manifest_path = os.path.join(root, filename)
                    subfolder_name = os.path.basename(os.path.dirname(manifest_path))




if __name__ == "__main__":
    manifest_lookup()
    print("Code statusCheck executed")