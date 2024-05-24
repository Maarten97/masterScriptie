import os
import csv
import xml.etree.ElementTree as ET
import bwbWriteGeneral

# root_dir = 'D:/BWB/Origineel/202210_BWB_4'
root_dir = 'C:/Users/looij/Documents/BWB/Subset2/202210_BWB_4'
output_dir = 'D:/BWB/Output/Verdrag'


# Requires already that are not more versions in the file
def folder_lookup():
    dirlist = []
    for bwbid in os.listdir(root_dir):
        if bwbid.startswith('BWBV'):
            if os.path.isdir(os.path.join(root_dir, bwbid)):
                dirlist.append(os.path.join(root_dir, bwbid))
    return dirlist


def language_check(dlist):
    for direc in dlist:
        for filename in os.listdir(direc):
            if filename.endswith(".xml"):
                print("Currently loading XML file: " + filename)
                bwbWriteGeneral.write_error("bwbVerdragSelect", "Error in file: " + filename + ". No more code.")


if __name__ == "__main__":
    dirlist = folder_lookup()
    language_check(dirlist)
    print("Code statusCheck executed")
