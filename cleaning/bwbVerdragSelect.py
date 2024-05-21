import os
import csv

root_dir = 'D:/BWB/Origineel/202210_BWB_4'
output_dir = 'D:/BWB/Output/Verdrag'
error_file = 'D:/BWB/Output/Verdrag/error.txt'


def folder_lookup():
    dirlist = []
    for bwbid in os.listdir(root_dir):
        if bwbid.startswith('BWBV'):
            if os.path.isdir(os.path.join(root_dir, bwbid)):
                dirlist.append(os.path.join(root_dir, bwbid))
    return dirlist

def status_check(dirlist):
    for direc in dirlist:
        return None




if __name__ == "__main__":
    dirlist = folder_lookup()
    status_check(dirlist)
    print("Code statusCheck executed")
